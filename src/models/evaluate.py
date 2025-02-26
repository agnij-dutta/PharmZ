import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve
import logging
from typing import Dict, Tuple
from baseline import BaselineModel
from ingestion import DataIngestion
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model_path: str = "models/baseline_xgb.json"):
        self.model = BaselineModel()
        if Path(model_path).exists():
            self.model.model = xgb.Booster()
            self.model.model.load_model(model_path)
        else:
            logger.warning(f"Model not found at {model_path}, will need to train first")
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
        """Plot calibration curve"""
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('Predicted Probability')
        plt.ylabel('True Probability')
        plt.title('Calibration Curve')
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def analyze_feature_importance(self, feature_names: list = None) -> Dict[str, float]:
        """Analyze and visualize feature importance"""
        importance = self.model.model.get_score(importance_type='gain')
        if feature_names:
            importance = {feature_names[int(k.replace('f', ''))]: v 
                        for k, v in importance.items()}
        
        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        # Plot top 20 features
        plt.figure(figsize=(12, 6))
        plt.bar(list(importance.keys())[:20], list(importance.values())[:20])
        plt.xticks(rotation=45, ha='right')
        plt.title('Top 20 Most Important Features')
        plt.tight_layout()
        plt.savefig('models/feature_importance.png')
        plt.close()
        
        return importance
    
    def evaluate_model(self, 
                      test_mol_data: pd.DataFrame,
                      test_clinical_data: pd.DataFrame,
                      test_labels: np.ndarray) -> Dict:
        """Comprehensive model evaluation"""
        # Generate predictions
        y_pred = self.model.predict(test_mol_data, test_clinical_data)
        
        # Calculate metrics
        metrics = {
            'auc': roc_auc_score(test_labels, y_pred),
            'average_precision': average_precision_score(test_labels, y_pred)
        }
        
        # Generate plots
        self.plot_roc_curve(test_labels, y_pred, 'models/roc_curve.png')
        self.plot_calibration_curve(test_labels, y_pred, 'models/calibration_curve.png')
        
        # Analyze feature importance
        importance = self.analyze_feature_importance()
        metrics['feature_importance'] = importance
        
        # Save metrics
        with open('models/evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        return metrics

def main():
    # Load and prepare data
    ingestion = DataIngestion()
    mol_features, response = ingestion.prepare_drug_response_data()
    
    # Convert to binary response
    binary_response = (response > response.median()).numpy()
    
    # Split data
    test_size = 0.2
    indices = np.arange(len(binary_response))
    np.random.shuffle(indices)
    split_idx = int(len(indices) * (1 - test_size))
    
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    # Train model if needed
    model_path = "models/baseline_xgb.json"
    if not Path(model_path).exists():
        logger.info("Training new model...")
        model = BaselineModel()
        metrics = model.train(
            mol_features.iloc[train_idx],
            clinical_data.iloc[train_idx],
            binary_response[train_idx]
        )
        logger.info(f"Training metrics: {metrics}")
    
    # Evaluate model
    evaluator = ModelEvaluator(model_path)
    metrics = evaluator.evaluate_model(
        mol_features.iloc[test_idx],
        clinical_data.iloc[test_idx],
        binary_response[test_idx]
    )
    
    logger.info(f"Evaluation metrics: {metrics}")

if __name__ == "__main__":
    main()