import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
src_path = str(Path(__file__).parent.parent)
sys.path.append(src_path)

from src.models.baseline import BaselineModel, MoleculeFeaturizer

class TestBaselineModel(unittest.TestCase):
    def setUp(self):
        self.featurizer = MoleculeFeaturizer()
        self.model = BaselineModel()
        
        # Create more realistic test data with multiple compounds
        self.mol_data = pd.DataFrame({
            'smiles': [
                'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
                'CC1=CC=C(C=C1)NC(=O)CN2CCN(CC2)CC3=CC=C(C=C3)OCC4=CC=CC=C4',  # Example drug
                'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
                'CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C',  # Testosterone
                'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Duplicate for testing
                'CC(=O)OC1=CC=CC=C1C(=O)O'  # Duplicate for testing
            ],
            'compound_id': ['2244', '11957529', '2519', '6013', '2519', '2244']
        })
        
        # Create clinical features with multiple samples
        num_samples = len(self.mol_data)
        num_features = 10
        self.clinical_data = pd.DataFrame(
            np.random.randn(num_samples, num_features),
            columns=[f'feature_{i}' for i in range(num_features)]
        )
        
        # Create balanced binary labels
        self.labels = np.array([0, 1, 0, 1, 1, 0])
        
    def test_morgan_fingerprint(self):
        fp = self.featurizer.get_morgan_fingerprint(self.mol_data['smiles'].iloc[0])
        self.assertEqual(len(fp), 2048)
        self.assertTrue(isinstance(fp, np.ndarray))
        
    def test_pubchem_features(self):
        features = self.featurizer.get_pubchem_features(self.mol_data['compound_id'].iloc[0])
        self.assertTrue(all(k in features for k in [
            'molecular_weight', 'xlogp', 'rotatable_bond_count',
            'h_bond_donor_count', 'h_bond_acceptor_count', 'tpsa'
        ]))
        
    def test_model_training(self):
        metrics = self.model.train(self.mol_data, self.clinical_data, self.labels)
        self.assertIn('auc', metrics)
        self.assertIn('average_precision', metrics)
        self.assertTrue(0 <= metrics['auc'] <= 1)
        self.assertTrue(0 <= metrics['average_precision'] <= 1)
        
    def test_model_prediction(self):
        # Train model first
        self.model.train(self.mol_data, self.clinical_data, self.labels)
        
        # Get predictions for the same data
        preds = self.model.predict(self.mol_data, self.clinical_data)
        self.assertEqual(len(preds), len(self.mol_data))
        self.assertTrue(all(0 <= p <= 1 for p in preds))
        
        # Test prediction shape for new data
        new_mol_data = pd.DataFrame({
            'smiles': ['CC(=O)OC1=CC=CC=C1C(=O)O'],
            'compound_id': ['2244']
        })
        new_clinical_data = pd.DataFrame(
            np.random.randn(1, 10),
            columns=[f'feature_{i}' for i in range(10)]
        )
        new_preds = self.model.predict(new_mol_data, new_clinical_data)
        self.assertEqual(len(new_preds), 1)
        self.assertTrue(all(0 <= p <= 1 for p in new_preds))

if __name__ == '__main__':
    unittest.main()