import unittest
import torch
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
src_path = str(Path(__file__).parent.parent)
sys.path.append(src_path)

# Create mock for dgl
class MockGraph:
    def __init__(self):
        self.ndata = {}

mock_dgl = Mock()
mock_dgl.graph = Mock(return_value=MockGraph())
mock_dgl.batch = Mock(return_value=MockGraph())
sys.modules['dgl'] = mock_dgl

from src.models.fusion_model import FusionModel, DrugEncoder, EHREncoder

class TestFusionModel(unittest.TestCase):
    def setUp(self):
        self.config = {
            'drug_input_dim': 32,
            'drug_hidden_dim': 64,
            'drug_num_layers': 3,
            'ehr_input_dim': 128,
            'ehr_hidden_dim': 64,
            'fusion_dim': 128,
            'num_classes': 2,
            'dropout': 0.1
        }
        self.model = FusionModel(self.config)
        
    def test_drug_encoder(self):
        encoder = DrugEncoder(
            self.config['drug_input_dim'],
            self.config['drug_hidden_dim'],
            self.config['drug_num_layers']
        )
        
        # Test with dummy input
        batch_size = 3
        dummy_input = torch.randn(batch_size, self.config['drug_input_dim'])
        output = encoder(dummy_input)
        
        self.assertEqual(output.shape, (batch_size, self.config['drug_hidden_dim']))
        
    def test_ehr_encoder(self):
        batch_size = 2
        seq_len = 5
        x = torch.randn(batch_size, seq_len, self.config['ehr_input_dim'])
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        encoder = EHREncoder(
            self.config['ehr_input_dim'],
            self.config['ehr_hidden_dim']
        )
        output = encoder(x, mask)
        
        self.assertEqual(output.shape, (batch_size, self.config['ehr_hidden_dim'] * 2))
        
    def test_forward_pass(self):
        batch_size = 2
        seq_len = 5
        
        batch = {
            'drug_graph': torch.randn(batch_size, self.config['drug_input_dim']),  # Using tensor input for testing
            'ehr_data': torch.randn(batch_size, seq_len, self.config['ehr_input_dim']),
            'ehr_mask': torch.ones(batch_size, seq_len, dtype=torch.bool)
        }
        
        logits, features = self.model(batch)
        
        self.assertEqual(logits.shape, (batch_size, self.config['num_classes']))
        self.assertIn('drug_features', features)
        self.assertIn('ehr_features', features)
        self.assertEqual(features['drug_features'].shape, (batch_size, self.config['drug_hidden_dim']))
        self.assertEqual(features['ehr_features'].shape, (batch_size, self.config['ehr_hidden_dim'] * 2))

if __name__ == '__main__':
    unittest.main()