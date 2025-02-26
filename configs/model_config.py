from dataclasses import dataclass
from typing import Dict, Any
import yaml
import os

@dataclass
class ModelConfig:
    # Model architecture
    drug_input_dim: int = 128
    drug_hidden_dim: int = 256
    drug_num_layers: int = 5
    ehr_input_dim: int = 768
    ehr_hidden_dim: int = 128
    fusion_dim: int = 512
    num_classes: int = 2
    dropout: float = 0.3

    # Training parameters
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    early_stopping_patience: int = 10

    # Federated learning
    num_federated_rounds: int = 10
    min_clients: int = 3
    local_epochs: int = 5

    # Clinical parameters
    confidence_threshold: float = 0.8
    pk_base_clearance: float = 1.0
    pk_base_volume: float = 1.0
    ec50: float = 0.5
    hill_coefficient: float = 1.2

    @classmethod
    def load_from_file(cls, config_path: str) -> 'ModelConfig':
        """Load configuration from YAML file"""
        if not os.path.exists(config_path):
            return cls()
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def save_to_file(self, config_path: str):
        """Save configuration to YAML file"""
        config_dict = self.__dict__
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self.__dict__

# Default configuration file
default_config = """
# Model architecture
drug_input_dim: 128
drug_hidden_dim: 256
drug_num_layers: 5
ehr_input_dim: 768
ehr_hidden_dim: 128
fusion_dim: 512
num_classes: 2
dropout: 0.3

# Training parameters
batch_size: 32
learning_rate: 0.001
num_epochs: 100
early_stopping_patience: 10

# Federated learning
num_federated_rounds: 10
min_clients: 3
local_epochs: 5

# Clinical parameters
confidence_threshold: 0.8
pk_base_clearance: 1.0
pk_base_volume: 1.0
ec50: 0.5
hill_coefficient: 1.2
"""