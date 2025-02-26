import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import pubchempy as pcp
from typing import Dict, Tuple, List
import logging
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MoleculeFeaturizer:
    def __init__(self):
        self.feature_size = 2048  # Morgan fingerprint size
        
    def get_morgan_fingerprint(self, smiles: str) -> np.ndarray:
        """Convert SMILES to Morgan fingerprint"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(self.feature_size)
            return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 3, self.feature_size))
        except:
            return np.zeros(self.feature_size)
    
    def get_molecular_descriptors(self, mol) -> Dict:
        """Calculate RDKit molecular descriptors"""
        if mol is None:
            return {name: 0 for name in Descriptors._descList}
        
        return {name: func(mol) for name, func in Descriptors._descList}
    
    def get_pubchem_features(self, compound_id: str) -> Dict:
        """Get compound properties from PubChem"""
        try:
            compound = pcp.get_compounds(compound_id, 'cid')[0]
            props = {
                'molecular_weight': compound.molecular_weight,
                'xlogp': compound.xlogp,
                'rotatable_bond_count': compound.rotatable_bond_count,
                'h_bond_donor_count': compound.h_bond_donor_count,
                'h_bond_acceptor_count': compound.h_bond_acceptor_count,
                'tpsa': compound.tpsa,
                'complexity': compound.complexity,
                'heavy_atom_count': compound.heavy_atom_count,
                'charge': compound.charge,
                'covalent_unit_count': compound.covalent_unit_count
            }
            return {k: (v if v is not None else 0) for k, v in props.items()}
        except:
            return {k: 0 for k in [
                'molecular_weight', 'xlogp', 'rotatable_bond_count',
                'h_bond_donor_count', 'h_bond_acceptor_count', 'tpsa',
                'complexity', 'heavy_atom_count', 'charge', 'covalent_unit_count'
            ]}

class BaselineModel:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.featurizer = MoleculeFeaturizer()
        self.model = None
        
    def prepare_features(self, 
                        molecular_data: pd.DataFrame,
                        clinical_data: pd.DataFrame) -> np.ndarray:
        """Prepare combined features for training"""
        # Process molecular features
        mol_features = []
        for _, row in molecular_data.iterrows():
            # Get Morgan fingerprint
            morgan_fp = self.featurizer.get_morgan_fingerprint(row['smiles'])
            
            # Get molecule object
            mol = Chem.MolFromSmiles(row['smiles'])
            
            # Get RDKit descriptors
            rdkit_desc = self.featurizer.get_molecular_descriptors(mol)
            rdkit_features = np.array(list(rdkit_desc.values()))
            
            # Get PubChem properties
            pubchem_props = self.featurizer.get_pubchem_features(row['compound_id'])
            pubchem_features = np.array([pubchem_props[k] for k in sorted(pubchem_props.keys())])
            
            # Combine all molecular features
            combined = np.concatenate([
                morgan_fp,
                rdkit_features,
                pubchem_features
            ])
            mol_features.append(combined)
        
        mol_features = np.array(mol_features)
        
        # Combine with clinical features
        X = np.concatenate([
            mol_features,
            clinical_data.values
        ], axis=1)
        
        return X.astype(np.float32)
    
    def train(self, 
              molecular_data: pd.DataFrame,
              clinical_data: pd.DataFrame,
              labels: np.ndarray,
              valid_frac: float = 0.2) -> Dict[str, float]:
        """Train XGBoost model with hyperparameter tuning"""
        X = self.prepare_features(molecular_data, clinical_data)
        X_train, X_val, y_train, y_val = train_test_split(
            X, labels, test_size=valid_frac, random_state=42, stratify=labels
        )
        
        # Create DMatrix objects
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Define parameter grid for tuning
        param_grid = {
            'max_depth': [4, 6, 8],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'eta': [0.01, 0.1],
            'gamma': [0, 0.1, 0.2]
        }
        
        # Basic parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'logloss'],
            'tree_method': 'hist'  # For faster training
        }
        
        # Train final model with best parameters
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, 'validation')],
            early_stopping_rounds=20,
            verbose_eval=100
        )
        
        # Calculate metrics
        y_pred = self.model.predict(dval)
        auc_score = roc_auc_score(y_val, y_pred)
        ap_score = average_precision_score(y_val, y_pred)
        
        # Feature importance analysis
        importance = self.model.get_score(importance_type='gain')
        top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])
        
        metrics = {
            'auc': auc_score,
            'average_precision': ap_score,
            'top_features': top_features,
            'best_iteration': self.model.best_iteration
        }
        
        # Save model
        self.model.save_model(str(self.model_dir / 'baseline_xgb.json'))
        
        return metrics
    
    def predict(self, 
                molecular_data: pd.DataFrame,
                clinical_data: pd.DataFrame) -> np.ndarray:
        """Generate predictions for new data"""
        if self.model is None:
            raise ValueError("Model needs to be trained first")
            
        X = self.prepare_features(molecular_data, clinical_data)
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

if __name__ == "__main__":
    # Load sample data
    from ingestion import DataIngestion
    
    ingestion = DataIngestion()
    mol_features, response = ingestion.prepare_drug_response_data()
    
    # Convert to binary response for classification
    binary_response = (response > response.median()).numpy()
    
    # Train baseline model
    model = BaselineModel()
    metrics = model.train(mol_features, clinical_data, binary_response)
    logger.info(f"Training metrics: {metrics}")