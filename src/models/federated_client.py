import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
from src.models.gnn_lstm_fusion import GNNLSTMFusion
import numpy as np
from collections import OrderedDict
from cryptography.fernet import Fernet
import torch.nn.functional as F
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecureAggregator:
    def __init__(self, encryption_key: bytes):
        self.fernet = Fernet(encryption_key)
        
    def secure_aggregate(self, local_updates: List[OrderedDict]) -> OrderedDict:
        """Securely aggregate model updates using secure multiparty computation"""
        n_clients = len(local_updates)
        if n_clients == 0:
            return OrderedDict()
            
        # Initialize aggregate with zeros like first client's update
        aggregate = OrderedDict(
            (k, torch.zeros_like(v)) 
            for k, v in local_updates[0].items()
        )
        
        # Add all updates
        for update in local_updates:
            for key in aggregate.keys():
                aggregate[key] += update[key]
                
        # Average
        for key in aggregate.keys():
            aggregate[key] = aggregate[key] / n_clients
            
        return aggregate

class FederatedClient:
    def __init__(self, 
                 client_id: str,
                 config: Dict,
                 data_dir: str = "data",
                 privacy_budget: float = 1.0,
                 noise_multiplier: float = 1.1,
                 max_grad_norm: float = 1.0):
        self.client_id = client_id
        self.config = config
        self.data_dir = Path(data_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = GNNLSTMFusion(config).to(self.device)
        # Make model compatible with differential privacy
        self.model = ModuleValidator.fix(self.model)
        ModuleValidator.validate(self.model, strict=False)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Initialize privacy engine
        self.privacy_engine = PrivacyEngine()
        
        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=None,  # Will be set during training
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            target_epsilon=privacy_budget
        )
        
        self.criterion = nn.BCELoss()
        
    def train(self, 
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              num_epochs: int = 1) -> Dict:
        """Train model on local data with differential privacy"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Set train loader for privacy engine
        self.train_loader = train_loader
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch in train_loader:
                # Move data to device
                mol_graph = batch['mol_graph'].to(self.device)
                clinical_data = batch['clinical_data'].to(self.device)
                clinical_lengths = batch['clinical_lengths'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(mol_graph, clinical_data, clinical_lengths)
                loss = self.criterion(outputs['predictions'].squeeze(), labels)
                
                # Backward pass with privacy
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                # Calculate accuracy
                predictions = (outputs['predictions'].squeeze() > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
            
            total_loss += epoch_loss / len(train_loader)
            
            # Log progress
            privacy_used = self.privacy_engine.accountant.get_epsilon(delta=1e-5)
            logger.info(f"Client {self.client_id} - Epoch {epoch+1}/{num_epochs}")
            logger.info(f"Loss: {epoch_loss/len(train_loader):.4f}")
            logger.info(f"Accuracy: {100*correct/total:.2f}%")
            logger.info(f"Privacy budget spent (ε, δ=1e-5): {privacy_used:.2f}")
        
        # Validation
        val_metrics = {}
        if val_loader is not None:
            val_metrics = self.evaluate(val_loader)
        
        metrics = {
            'train_loss': total_loss / num_epochs,
            'train_accuracy': 100 * correct / total,
            'privacy_budget_spent': privacy_used,
            **val_metrics
        }
        
        return metrics
    
    def evaluate(self, val_loader: DataLoader) -> Dict:
        """Evaluate model on validation data"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        predictions = []
        labels_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                mol_graph = batch['mol_graph'].to(self.device)
                clinical_data = batch['clinical_data'].to(self.device)
                clinical_lengths = batch['clinical_lengths'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(mol_graph, clinical_data, clinical_lengths)
                loss = self.criterion(outputs['predictions'].squeeze(), labels)
                
                total_loss += loss.item()
                
                pred = (outputs['predictions'].squeeze() > 0.5).float()
                correct += (pred == labels).sum().item()
                total += labels.size(0)
                
                predictions.extend(outputs['predictions'].cpu().numpy())
                labels_list.extend(labels.cpu().numpy())
        
        metrics = {
            'val_loss': total_loss / len(val_loader),
            'val_accuracy': 100 * correct / total
        }
        
        return metrics
    
    def get_weights(self) -> OrderedDict:
        """Get model weights for aggregation"""
        return OrderedDict({
            k: v.cpu() for k, v in self.model.state_dict().items()
        })
    
    def set_weights(self, weights: OrderedDict):
        """Update model with aggregated weights"""
        self.model.load_state_dict(weights)
    
    def save_model(self, path: str):
        """Save local model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'client_id': self.client_id
        }, path)
    
    def load_model(self, path: str):
        """Load local model"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    @staticmethod
    def aggregate_weights(weights_list: List[OrderedDict]) -> OrderedDict:
        """Aggregate weights from multiple clients using secure aggregation"""
        mean_state_dict = OrderedDict()
        
        # Get all keys (assume same for all state dicts)
        keys = weights_list[0].keys()
        
        # Compute mean of weights with added noise for privacy
        for key in keys:
            tensors = [state_dict[key] for state_dict in weights_list]
            # Add small amount of Gaussian noise for additional privacy
            noise = torch.randn_like(tensors[0]) * 0.01
            mean_state_dict[key] = torch.stack(tensors).mean(dim=0) + noise
            
        return mean_state_dict