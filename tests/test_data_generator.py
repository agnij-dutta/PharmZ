import torch
import numpy as np
from typing import Dict
import pandas as pd
import json

class TestDataGenerator:
    def __init__(self, num_samples: int = 100):
        self.num_samples = num_samples
        
    def generate_drug_data(self) -> pd.DataFrame:
        """Generate synthetic drug data"""
        # Common drug SMILES for testing
        drug_smiles = [
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
            'CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C',  # Testosterone
            'CC(C)(C)NCC(O)C1=CC(O)=CC(O)=C1',  # Albuterol
            'CC1=C(C)C=C2C=CC(C)(C)OC2=C1'  # Example drug
        ]
        
        # Generate dataframe with drug info
        return pd.DataFrame({
            'smiles': np.random.choice(drug_smiles, size=self.num_samples),
            'target': ['TEST'] * self.num_samples,
            'pubchem_id': [str(i) for i in range(self.num_samples)]
        })
    
    def generate_clinical_data(self) -> pd.DataFrame:
        """Generate synthetic clinical data"""
        clinical_data = []
        
        for i in range(self.num_samples):
            # Generate random time series length
            num_timepoints = np.random.randint(5, 15)
            timestamps = sorted(np.random.uniform(0, 100, num_timepoints))
            
            # Generate measurements
            measurements = []
            for _ in range(num_timepoints):
                measurement = {
                    'hr': float(np.random.normal(80, 10)),  # Heart rate
                    'bp': float(np.random.normal(120, 15)),  # Blood pressure
                    'temp': float(np.random.normal(37, 0.5))  # Temperature
                }
                measurements.append(measurement)
            
            clinical_data.append({
                'patient_id': i,
                'measurements': json.dumps(measurements),  # Store as JSON string
                'timestamp': json.dumps(list(timestamps)),  # Convert numpy array to list before JSON dump
                'outcome': float(np.random.randint(0, 2))  # Binary outcome as float
            })
        
        return pd.DataFrame(clinical_data)
    
    def generate_dataset(self) -> Dict[str, pd.DataFrame]:
        """Generate complete synthetic dataset"""
        return {
            'drug_data': self.generate_drug_data(),
            'clinical_data': self.generate_clinical_data()
        }

if __name__ == "__main__":
    generator = TestDataGenerator(num_samples=100)
    dataset = generator.generate_dataset()
    print(f"Generated {len(dataset['drug_data'])} drug samples")
    print(f"Generated {len(dataset['clinical_data'])} clinical samples")
    
    # Print example data
    print("\nExample drug data:")
    print(dataset['drug_data'].head())
    print("\nExample clinical data:")
    print(dataset['clinical_data'].head())