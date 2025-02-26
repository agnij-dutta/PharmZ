from dataclasses import dataclass
from typing import Optional
import yaml

@dataclass
class ModelConfig:
    # Molecular graph parameters
    mol_dim: int = 7  # Number of atom features
    graph_hidden_dim: int = 64
    graph_out_dim: int = 128
    graph_layers: int = 3
    
    # Clinical data parameters
    clinical_dim: int = 3  # Number of clinical features
    clinical_hidden_dim: int = 64
    clinical_layers: int = 2
    
    # Fusion parameters
    fusion_dim: int = 256
    num_attention_heads: int = 4
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    dropout: float = 0.1
    num_epochs: int = 100
    early_stopping_patience: int = 10
    
    # Data parameters
    max_atoms: int = 50
    max_seq_length: int = 100
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Federated learning parameters
    min_clients: int = 3
    num_federated_rounds: int = 50
    local_epochs: int = 5
    privacy_budget: float = 1.0
    noise_multiplier: float = 1.1
    max_grad_norm: float = 1.0
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ModelConfig':
        return cls(**config_dict)
    
    @classmethod
    def load_from_file(cls, config_path: str) -> 'ModelConfig':
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)['model_params']
        return cls.from_dict(config_dict)
    
    def save_to_file(self, config_path: str):
        config_dict = {
            'model_params': self.__dict__
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)