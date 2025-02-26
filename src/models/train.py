import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path
import yaml
import logging
import sys
import argparse
from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_auc_score

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.gnn_lstm_fusion import GNNLSTMFusion
from src.models.federated_client import FederatedClient
from src.data.ingestion import DataIngestion
from tests.test_data_generator import TestDataGenerator
from src.models.model_config import ModelConfig

class DrugResponseDataset(TensorDataset):
    def __init__(self, data_dict: Dict[str, torch.Tensor]):
        logger.debug(f"Initializing dataset with keys: {data_dict.keys()}")
        self.mol_features = data_dict['mol_features']
        self.mol_adj = data_dict['mol_adj']
        self.clinical_features = data_dict['clinical_features']
        self.clinical_lengths = data_dict['clinical_lengths']
        self.labels = data_dict['labels']
        logger.debug(f"Dataset shapes: mol_features={self.mol_features.shape}, "
                    f"clinical_features={self.clinical_features.shape}, "
                    f"labels={self.labels.shape}")
        
    def __getitem__(self, index):
        return {
            'mol_features': self.mol_features[index],
            'mol_adj': self.mol_adj[index],
            'clinical_features': self.clinical_features[index],
            'clinical_lengths': self.clinical_lengths[index],
            'labels': self.labels[index]
        }
    
    def __len__(self):
        return len(self.labels)

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2) -> tuple:
    """Performs mixup on the input batch"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y

class Trainer:
    def __init__(self, config_path: str = "configs/default_config.yaml"):
        logger.debug(f"Loading config from {config_path}")
        self.config = ModelConfig.load_from_file(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize data processing
        self.data_ingestion = DataIngestion()
        
        self.mixup_alpha = 0.2
        self.grad_accum_steps = 4
        self.best_val_auc = 0.0
        
    def prepare_data(self, use_test_data: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data loaders for training"""
        logger.info("Preparing datasets...")
        
        if use_test_data:
            # Generate synthetic test data
            logger.debug("Generating synthetic test data")
            generator = TestDataGenerator(num_samples=1000)
            test_data = generator.generate_dataset()
            logger.debug(f"Generated {len(test_data['drug_data'])} drug samples and "
                        f"{len(test_data['clinical_data'])} clinical samples")
            data_dict = self.data_ingestion.prepare_data(
                drug_data=test_data['drug_data'],
                clinical_data=test_data['clinical_data']
            )
        else:
            # Use real data
            logger.debug("Loading real data")
            data_dict = self.data_ingestion.prepare_data()
        
        # Create dataset
        dataset = DrugResponseDataset(data_dict)
        
        # Split sizes
        total_size = len(dataset)
        train_size = int(self.config.train_split * total_size)
        val_size = int(self.config.val_split * total_size)
        test_size = total_size - train_size - val_size
        
        logger.debug(f"Splitting dataset: train={train_size}, val={val_size}, test={test_size}")
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size
        )
        
        logger.debug("Data loaders created successfully")
        return train_loader, val_loader, test_loader
    
    def train_centralized(self, train_loader: DataLoader, val_loader: DataLoader):
        """Centralized training with GNN-LSTM fusion model"""
        logger.info("Starting centralized training...")
        
        model = GNNLSTMFusion(self.config.__dict__).to(self.device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Cosine annealing with warmup
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            epochs=self.config.num_epochs,
            steps_per_epoch=len(train_loader) // self.grad_accum_steps,
            pct_start=0.1,
            anneal_strategy='cos',
            cycle_momentum=True
        )
        
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        scaler = torch.cuda.amp.GradScaler()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        try:
            for epoch in range(self.config.num_epochs):
                # Training
                model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0
                optimizer.zero_grad()
                
                for batch_idx, batch in enumerate(train_loader):
                    # Move data to device
                    mol_features = batch['mol_features'].to(self.device)
                    mol_adj = batch['mol_adj'].to(self.device)
                    clinical_features = batch['clinical_features'].to(self.device)
                    clinical_lengths = batch['clinical_lengths'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Apply mixup augmentation during training
                    if model.training:
                        mol_features, mixed_labels = mixup_data(mol_features, labels, self.mixup_alpha)
                        clinical_features, _ = mixup_data(clinical_features, labels, self.mixup_alpha)
                        labels = mixed_labels
                    
                    logger.debug(f"Batch {batch_idx}: Processing batch with shapes "
                               f"mol={mol_features.shape}, clinical={clinical_features.shape}")
                    
                    # Forward pass with mixed precision
                    with torch.cuda.amp.autocast():
                        outputs = model(mol_features, mol_adj, clinical_features, clinical_lengths)
                        loss = criterion(outputs['predictions'].squeeze(), labels)
                        
                        # Scale loss for gradient accumulation
                        loss = loss / self.grad_accum_steps
                    
                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % self.grad_accum_steps == 0:
                        # Gradient clipping
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        # Optimizer step with gradient scaling
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()
                    
                    # Update metrics
                    train_loss += loss.item() * self.grad_accum_steps
                    predictions = (outputs['predictions'].squeeze() > 0.0).float()
                    train_correct += (predictions == labels).sum().item()
                    train_total += labels.size(0)
                
                # Validation
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        mol_features = batch['mol_features'].to(self.device)
                        mol_adj = batch['mol_adj'].to(self.device)
                        clinical_features = batch['clinical_features'].to(self.device)
                        clinical_lengths = batch['clinical_lengths'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        
                        outputs = model(mol_features, mol_adj, clinical_features, clinical_lengths)
                        loss = criterion(outputs['predictions'].squeeze(), labels)
                        val_loss += loss.item()
                        
                        # Calculate accuracy
                        predictions = (outputs['predictions'].squeeze() > 0.5).float()
                        val_correct += (predictions == labels).sum().item()
                        val_total += labels.size(0)
                
                # Calculate metrics
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                train_accuracy = 100 * train_correct / train_total
                val_accuracy = 100 * val_correct / val_total
                
                # Log progress
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
                logger.info(f"Train Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
                logger.info(f"Val Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model
                    model_path = self.model_dir / "fusion_model.pt"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': best_val_loss,
                        'config': self.config.__dict__
                    }, model_path)
                    logger.info(f"Saved best model checkpoint to {model_path}")
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        logger.info("Early stopping triggered")
                        break
                        
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def train_federated(self, train_loader: DataLoader, val_loader: DataLoader):
        """Federated training with multiple clients"""
        logger.info("Starting federated training...")
        
        # Create federated clients
        clients = []
        for i in range(self.config.min_clients):
            client = FederatedClient(
                f"client_{i}",
                self.config.__dict__,
                privacy_budget=self.config.privacy_budget,
                noise_multiplier=self.config.noise_multiplier,
                max_grad_norm=self.config.max_grad_norm
            )
            clients.append(client)
        
        # Federated training rounds
        for round_num in range(self.config.num_federated_rounds):
            logger.info(f"Federated Round {round_num + 1}")
            
            # Train each client locally
            client_weights = []
            for client in clients:
                metrics = client.train(
                    train_loader,
                    val_loader,
                    num_epochs=self.config.local_epochs
                )
                client_weights.append(client.get_weights())
                logger.info(f"Client {client.client_id} metrics: {metrics}")
            
            # Aggregate weights
            aggregated_weights = FederatedClient.aggregate_weights(client_weights)
            
            # Update all clients with aggregated weights
            for client in clients:
                client.set_weights(aggregated_weights)
            
            # Save aggregated model
            model_path = self.model_dir / f"federated_model_round_{round_num + 1}.pt"
            torch.save({
                'round': round_num + 1,
                'state_dict': aggregated_weights,
                'config': self.config.__dict__
            }, model_path)
            logger.info(f"Saved federated model checkpoint to {model_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['centralized', 'federated'], default='centralized')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml')
    parser.add_argument('--use-test-data', action='store_true', help='Use synthetic test data')
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        logger.debug("Initializing trainer...")
        trainer = Trainer(args.config)
        
        # Prepare data
        train_loader, val_loader, test_loader = trainer.prepare_data(use_test_data=args.use_test_data)
        
        # Train model
        if args.mode == 'centralized':
            trainer.train_centralized(train_loader, val_loader)
        else:
            trainer.train_federated(train_loader, val_loader)
            
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()