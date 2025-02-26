import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from typing import Dict, List, Tuple
import torch
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClinicalDataProcessor:
    def __init__(self, config_path: str = "configs/data_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.data_dir = Path(self.config['data_sources']['mimic']['local_path'])
        self.scaler = StandardScaler()
        
    def load_clinical_tables(self) -> Dict[str, pd.DataFrame]:
        """Load clinical tables from MIMIC-IV format"""
        tables = {}
        for table in self.config['data_sources']['mimic']['tables']:
            try:
                path = self.data_dir / f"{table}.csv.gz"
                if path.exists():
                    tables[table] = pd.read_csv(path)
                    logger.info(f"Loaded {table} with {len(tables[table])} records")
            except Exception as e:
                logger.error(f"Error loading {table}: {str(e)}")
        return tables
    
    def process_lab_events(self, lab_events: pd.DataFrame) -> pd.DataFrame:
        """Process laboratory test results"""
        vital_signs = self.config['preprocessing']['clinical_data']['vital_signs']
        lab_tests = self.config['preprocessing']['clinical_data']['lab_tests']
        
        # Filter relevant labs and vitals
        relevant_items = vital_signs + lab_tests
        events = lab_events[lab_events['itemid'].isin(relevant_items)]
        
        # Pivot to create feature matrix
        features = events.pivot_table(
            index='hadm_id',
            columns='itemid',
            values='valuenum',
            aggfunc='mean'
        ).reset_index()
        
        return features
    
    def process_medications(self, prescriptions: pd.DataFrame) -> pd.DataFrame:
        """Process medication orders"""
        meds = prescriptions.groupby(['hadm_id', 'drug'])['dose_val_rx'].agg(['mean', 'count']).reset_index()
        med_matrix = meds.pivot_table(
            index='hadm_id',
            columns='drug',
            values='mean'
        ).fillna(0)
        return med_matrix
    
    def create_patient_timeline(self, 
                              tables: Dict[str, pd.DataFrame]) -> Tuple[torch.Tensor, List[int]]:
        """Create temporal features for each patient admission"""
        features = []
        admission_ids = []
        
        lab_features = self.process_lab_events(tables['lab_events'])
        med_features = self.process_medications(tables['prescriptions'])
        
        # Merge features
        combined = pd.merge(lab_features, med_features, on='hadm_id', how='inner')
        
        # Scale features
        feature_matrix = self.scaler.fit_transform(combined.drop('hadm_id', axis=1))
        admission_ids = combined['hadm_id'].tolist()
        
        return torch.FloatTensor(feature_matrix), admission_ids
    
    def get_diagnoses(self, tables: Dict[str, pd.DataFrame]) -> Dict[int, List[str]]:
        """Get diagnosis codes for each admission"""
        diagnoses = tables['diagnoses_icd'].groupby('hadm_id')['icd_code'].agg(list).to_dict()
        return diagnoses
    
    def process_clinical_data(self) -> Tuple[torch.Tensor, Dict[int, List[str]]]:
        """Main processing pipeline for clinical data"""
        logger.info("Processing clinical data...")
        tables = self.load_clinical_tables()
        features, admission_ids = self.create_patient_timeline(tables)
        diagnoses = self.get_diagnoses(tables)
        
        logger.info(f"Processed {len(features)} patient admissions")
        logger.info(f"Feature dimension: {features.shape[1]}")
        
        return features, diagnoses

if __name__ == "__main__":
    processor = ClinicalDataProcessor()
    features, diagnoses = processor.process_clinical_data()