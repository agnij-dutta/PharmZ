import torch
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import h5py
from scipy import sparse
import anndata as ad
from bioservices import KEGG
from cmapPy.pandasGEXpress import parse
import logging
from rdkit import Chem
from rdkit.Chem import AllChem
import pubchempy as pcp
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MolecularGraphBuilder:
    def __init__(self, max_atoms: int = 50):
        self.max_atoms = max_atoms
        
    def smiles_to_graph(self, smiles: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert SMILES to molecular graph tensors"""
        # Create RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return self._empty_graph()
        
        # Get atom features
        atoms = mol.GetAtoms()
        atom_features = []
        for atom in atoms:
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetNumRadicalElectrons(),
                atom.GetIsAromatic(),
                atom.GetHybridization(),
                atom.GetTotalNumHs()
            ]
            atom_features.append(features)
            
        # Pad atom features if needed
        num_atoms = len(atoms)
        if num_atoms > self.max_atoms:
            atom_features = atom_features[:self.max_atoms]
        else:
            padding = [[0] * len(atom_features[0])] * (self.max_atoms - num_atoms)
            atom_features.extend(padding)
            
        # Create adjacency matrix
        adj_matrix = np.zeros((self.max_atoms, self.max_atoms))
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            if i < self.max_atoms and j < self.max_atoms:
                bond_type = bond.GetBondTypeAsDouble()
                adj_matrix[i, j] = bond_type
                adj_matrix[j, i] = bond_type
                
        return (
            torch.FloatTensor(atom_features),
            torch.FloatTensor(adj_matrix)
        )
    
    def _empty_graph(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return empty graph tensors for invalid SMILES"""
        return (
            torch.zeros(self.max_atoms, 7),
            torch.zeros(self.max_atoms, self.max_atoms)
        )

class ClinicalDataProcessor:
    def __init__(self, max_seq_length: int = 100):
        self.max_seq_length = max_seq_length
        
    def process_timeseries(self, 
                          measurements: List[Dict],
                          timestamps: List[float]) -> Tuple[torch.Tensor, int]:
        """Process clinical time-series data"""
        # Sort by timestamp
        sorted_data = sorted(zip(measurements, timestamps), key=lambda x: x[1])
        measurements = [x[0] for x in sorted_data]
        
        # Create feature matrix
        seq_length = min(len(measurements), self.max_seq_length)
        features = np.zeros((self.max_seq_length, len(measurements[0])))
        
        # Fill features
        for i, measurement in enumerate(measurements[:seq_length]):
            features[i] = list(measurement.values())
            
        return torch.FloatTensor(features), seq_length

class DataIngestion:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.kegg = KEGG()
        self.mol_graph_builder = MolecularGraphBuilder()
        self.clinical_processor = ClinicalDataProcessor()
        
    def download_ccle_data(self) -> pd.DataFrame:
        """Download CCLE gene expression data"""
        logger.info("Downloading CCLE data...")
        ccle_url = "https://ndownloader.figshare.com/files/13515395"
        ccle_data = pd.read_csv(ccle_url, index_col=0)
        ccle_data.to_csv(self.data_dir / "ccle_expression.csv")
        return ccle_data

    def download_gdsc_data(self) -> pd.DataFrame:
        """Download GDSC drug response data"""
        logger.info("Downloading GDSC data...")
        gdsc_url = "https://www.cancerrxgene.org/downloads/bulk_download"
        gdsc_data = pd.read_csv(gdsc_url, index_col=0)
        gdsc_data.to_csv(self.data_dir / "gdsc_response.csv")
        return gdsc_data

    def process_molecular_data(self, expression_data: pd.DataFrame) -> torch.Tensor:
        """Process molecular data into pathway-level features"""
        logger.info("Processing molecular data...")
        # Get KEGG pathway information
        pathways = self.kegg.pathwayIds("hsa")
        pathway_genes = {p: self.kegg.link("hsa", p) for p in pathways}
        
        # Convert gene expression to pathway activity scores
        pathway_scores = []
        for pathway, genes in pathway_genes.items():
            if genes is None:
                continue
            pathway_genes = [g.split(":")[-1] for g in genes]
            pathway_expr = expression_data[expression_data.index.isin(pathway_genes)]
            pathway_scores.append(pathway_expr.mean(axis=0))
            
        pathway_matrix = np.stack(pathway_scores)
        return torch.FloatTensor(pathway_matrix)

    def prepare_drug_response_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare drug response data for training"""
        logger.info("Preparing drug response data...")
        ccle_data = pd.read_csv(self.data_dir / "ccle_expression.csv")
        gdsc_data = pd.read_csv(self.data_dir / "gdsc_response.csv")
        
        # Process molecular features
        mol_features = self.process_molecular_data(ccle_data)
        
        # Process drug response
        response = torch.FloatTensor(gdsc_data["IC50"].values)
        
        return mol_features, response

    def download_drug_data(self) -> pd.DataFrame:
        """Download and process drug data"""
        logger.info("Downloading drug data...")
        try:
            drug_data = pd.read_csv(self.data_dir / "drug_data.csv")
        except FileNotFoundError:
            # Example drug data structure
            drug_data = pd.DataFrame({
                'smiles': ['CC(=O)OC1=CC=CC=C1C(=O)O'],  # Example SMILES
                'target': ['COX-2'],
                'pubchem_id': ['2244']
            })
            drug_data.to_csv(self.data_dir / "drug_data.csv", index=False)
        return drug_data
    
    def download_clinical_data(self) -> pd.DataFrame:
        """Download and process clinical data"""
        logger.info("Downloading clinical data...")
        try:
            clinical_data = pd.read_csv(self.data_dir / "clinical_data.csv")
        except FileNotFoundError:
            # Example clinical data structure
            clinical_data = pd.DataFrame({
                'patient_id': [1],
                'measurements': [{'hr': 80, 'bp': 120, 'temp': 37.0}],
                'timestamp': [0.0],
                'outcome': [1]
            })
            clinical_data.to_csv(self.data_dir / "clinical_data.csv", index=False)
        return clinical_data
    
    def prepare_molecular_graphs(self, smiles_list: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare molecular graph data"""
        logger.info("Processing molecular graphs...")
        features_list = []
        adj_list = []
        
        for smiles in smiles_list:
            features, adj = self.mol_graph_builder.smiles_to_graph(smiles)
            features_list.append(features)
            adj_list.append(adj)
            
        return (
            torch.stack(features_list),
            torch.stack(adj_list)
        )
    
    def prepare_clinical_sequences(self, 
                                 measurements_list: List[List[Dict]],
                                 timestamps_list: List[List[float]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare clinical sequence data"""
        logger.info("Processing clinical sequences...")
        features_list = []
        lengths_list = []
        
        for measurements, timestamps in zip(measurements_list, timestamps_list):
            features, length = self.clinical_processor.process_timeseries(
                measurements, timestamps
            )
            features_list.append(features)
            lengths_list.append(length)
            
        return (
            torch.stack(features_list),
            torch.tensor(lengths)
        )
    
    def prepare_data(self,
                    drug_data: Optional[pd.DataFrame] = None,
                    clinical_data: Optional[pd.DataFrame] = None) -> Dict[str, torch.Tensor]:
        """Prepare complete dataset"""
        # Load or use provided data
        if drug_data is None:
            drug_data = self.download_drug_data()
        if clinical_data is None:
            clinical_data = self.download_clinical_data()
        
        # Process molecular data
        mol_features, mol_adj = self.prepare_molecular_graphs(drug_data['smiles'])
        
        # Process clinical data
        clinical_features, seq_lengths = self.prepare_clinical_sequences(
            clinical_data['measurements'].apply(self._parse_measurements).tolist(),
            clinical_data['timestamp'].apply(self._parse_timestamps).tolist()
        )
        
        # Prepare labels
        labels = torch.FloatTensor(clinical_data['outcome'].values)
        
        return {
            'mol_features': mol_features,
            'mol_adj': mol_adj,
            'clinical_features': clinical_features,
            'clinical_lengths': seq_lengths,
            'labels': labels
        }

    def _parse_measurements(self, measurements_str: str) -> List[Dict]:
        """Parse measurements from string format"""
        if isinstance(measurements_str, str):
            try:
                return json.loads(measurements_str)
            except json.JSONDecodeError:
                return [{'hr': 80, 'bp': 120, 'temp': 37.0}]
        return measurements_str if isinstance(measurements_str, list) else [measurements_str]
    
    def _parse_timestamps(self, timestamps_str: str) -> List[float]:
        """Parse timestamps from string format"""
        if isinstance(timestamps_str, str):
            try:
                return json.loads(timestamps_str)
            except json.JSONDecodeError:
                return [0.0]
        return [float(timestamps_str)] if not isinstance(timestamps_str, list) else timestamps_str

    def prepare_molecular_graphs(self, smiles_list: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert SMILES to molecular graph tensors"""
        max_atoms = 50  # Maximum number of atoms to consider
        num_samples = len(smiles_list)
        
        # Initialize tensors
        features = torch.zeros(num_samples, max_atoms, 7)  # 7 atom features
        adj_matrices = torch.zeros(num_samples, max_atoms, max_atoms)
        
        for idx, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
                
            # Get atom features
            atoms = mol.GetAtoms()
            num_atoms = min(len(atoms), max_atoms)
            
            for atom_idx in range(num_atoms):
                atom = atoms[atom_idx]
                features[idx, atom_idx] = torch.tensor([
                    atom.GetAtomicNum(),
                    atom.GetDegree(),
                    atom.GetFormalCharge(),
                    atom.GetNumRadicalElectrons(),
                    int(atom.GetIsAromatic()),
                    atom.GetHybridization(),
                    atom.GetTotalNumHs()
                ])
            
            # Create adjacency matrix
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                if i < max_atoms and j < max_atoms:
                    bond_type = bond.GetBondTypeAsDouble()
                    adj_matrices[idx, i, j] = bond_type
                    adj_matrices[idx, j, i] = bond_type
        
        return features.float(), adj_matrices.float()
    
    def prepare_clinical_sequences(self, 
                                measurements_list: List[List[Dict]],
                                timestamps_list: List[List[float]],
                                max_seq_length: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process clinical time-series data"""
        num_samples = len(measurements_list)
        
        # Get number of features from first non-empty measurement
        first_measurement = next(
            (m[0] for m in measurements_list if m), 
            {'hr': 0, 'bp': 0, 'temp': 0}
        )
        num_features = len(first_measurement)
        
        # Initialize tensors
        features = torch.zeros(num_samples, max_seq_length, num_features)
        lengths = []
        
        for idx, (measurements, timestamps) in enumerate(zip(measurements_list, timestamps_list)):
            # Handle empty or invalid data
            if not measurements or not timestamps:
                lengths.append(1)
                continue
                
            # Sort by timestamp
            sorted_data = sorted(zip(measurements, timestamps), key=lambda x: x[1])
            measurements = [x[0] for x in sorted_data]
            
            # Process sequence
            seq_length = min(len(measurements), max_seq_length)
            lengths.append(seq_length)
            
            for t, measurement in enumerate(measurements[:seq_length]):
                if isinstance(measurement, dict):
                    features[idx, t] = torch.tensor(list(measurement.values()))
        
        return features.float(), torch.tensor(lengths)
    
    def download_drug_data(self) -> pd.DataFrame:
        """Download and process drug data"""
        logger.info("Downloading drug data...")
        try:
            drug_data = pd.read_csv(self.data_dir / "drug_data.csv")
        except FileNotFoundError:
            # Example drug data structure
            drug_data = pd.DataFrame({
                'smiles': ['CC(=O)OC1=CC=CC=C1C(=O)O'],  # Aspirin
                'target': ['COX-2'],
                'pubchem_id': ['2244']
            })
            drug_data.to_csv(self.data_dir / "drug_data.csv", index=False)
        return drug_data
    
    def download_clinical_data(self) -> pd.DataFrame:
        """Download and process clinical data"""
        logger.info("Downloading clinical data...")
        try:
            clinical_data = pd.read_csv(self.data_dir / "clinical_data.csv")
        except FileNotFoundError:
            # Example clinical data structure
            clinical_data = pd.DataFrame({
                'patient_id': [1],
                'measurements': [json.dumps([{'hr': 80, 'bp': 120, 'temp': 37.0}])],
                'timestamp': [json.dumps([0.0])],
                'outcome': [1]
            })
            clinical_data.to_csv(self.data_dir / "clinical_data.csv", index=False)
        return clinical_data

if __name__ == "__main__":
    ingestion = DataIngestion()
    ccle_data = ingestion.download_ccle_data()
    gdsc_data = ingestion.download_gdsc_data()
    mol_features, response = ingestion.prepare_drug_response_data()
    logger.info(f"Processed {len(mol_features)} molecular features")
    logger.info(f"Processed {len(response)} drug responses")
    data = ingestion.prepare_data()
    logger.info("Data preparation completed!")
    for k, v in data.items():
        logger.info(f"{k}: {v.shape}")

def load_genomic_data(path: str) -> pd.DataFrame:
    """Load and preprocess genomic data from various formats"""
    file_type = Path(path).suffix
    
    if file_type == '.h5':
        with h5py.File(path, 'r') as f:
            data = pd.DataFrame(f['data'][:], columns=f['gene_names'][:])
    elif file_type == '.mtx':
        # Load sparse matrix
        mtx = sparse.load_mtx(path)
        genes_path = Path(path).parent / 'genes.tsv'
        genes = pd.read_csv(genes_path, sep='\t', header=None)[0]
        data = pd.DataFrame.sparse.from_spmatrix(mtx, columns=genes)
    else:  # CSV
        data = pd.read_csv(path, index_col=0)
    
    # Basic QC and normalization
    adata = ad.AnnData(data)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    
    return pd.DataFrame(adata.X, columns=adata.var_names, index=adata.obs_names)

def load_clinical_data(path: str) -> pd.DataFrame:
    """Load and preprocess clinical data"""
    data = pd.read_csv(path)
    
    # Handle missing values
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    
    # Impute numerical with median
    data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())
    
    # Impute categorical with mode
    for col in categorical_cols:
        data[col] = data[col].fillna(data[col].mode()[0])
    
    # Encode categorical variables
    data = pd.get_dummies(data, columns=categorical_cols)
    
    # Normalize numerical columns
    data[numerical_cols] = (data[numerical_cols] - data[numerical_cols].mean()) / data[numerical_cols].std()
    
    return data

def fetch_drug_properties(drug_name: str) -> Dict:
    """Fetch drug properties from PubChem"""
    import pubchempy as pcp
    
    try:
        compound = pcp.get_compounds(drug_name, 'name')[0]
        properties = {
            'molecular_weight': compound.molecular_weight,
            'xlogp': compound.xlogp,
            'rotatable_bond_count': compound.rotatable_bond_count,
            'h_bond_donor_count': compound.h_bond_donor_count,
            'h_bond_acceptor_count': compound.h_bond_acceptor_count,
            'tpsa': compound.tpsa,
            'complexity': compound.complexity,
            'fingerprint': compound.fingerprint,
            'inchi': compound.inchi,
            'smiles': compound.canonical_smiles
        }
        return properties
    except:
        return None

def process_batch_data(
    genomic_data: pd.DataFrame,
    clinical_data: pd.DataFrame,
    drug_properties: Dict,
    response_data: Optional[pd.Series] = None
) -> Dict[str, torch.Tensor]:
    """Process a batch of data for model input"""
    
    # Align genomic and clinical data
    common_patients = genomic_data.index.intersection(clinical_data.index)
    genomic = genomic_data.loc[common_patients]
    clinical = clinical_data.loc[common_patients]
    
    # Convert to tensors
    processed = {
        'genomic': torch.FloatTensor(genomic.values),
        'clinical': torch.FloatTensor(clinical.values),
        'drug': {k: torch.tensor(v) for k, v in drug_properties.items() 
                if isinstance(v, (int, float, np.number))}
    }
    
    if response_data is not None:
        response = response_data.loc[common_patients]
        processed['response'] = torch.FloatTensor(response.values)
    
    return processed

def load_and_preprocess_data(
    genomic_path: str,
    clinical_path: str,
    drug_name: str,
    response_path: Optional[str] = None
) -> Tuple[Dict[str, torch.Tensor], Optional[None]]:
    """Load and preprocess all data modalities"""
    
    # Load data
    genomic_data = load_genomic_data(genomic_path)
    clinical_data = load_clinical_data(clinical_path)
    drug_props = fetch_drug_properties(drug_name)
    
    response_data = None
    if response_path:
        response_data = pd.read_csv(response_path, index_col=0)['response']
    
    # Process data
    processed_data = process_batch_data(
        genomic_data,
        clinical_data,
        drug_props,
        response_data
    )
    
    return processed_data, None