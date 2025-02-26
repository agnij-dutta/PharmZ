import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
try:
    import dgl
    from dgl.nn.pytorch.conv import GINConv
    from dgl.nn.pytorch import GATConv
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False
    dgl = None
    GINConv = None
    GATConv = None
import scanpy as sc

class AttentionBlock(nn.Module):
    def __init__(self, input_dim: int, n_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, n_heads)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.ReLU(),
            nn.Linear(4 * input_dim, input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + attended)
        return self.norm2(x + self.ffn(x))

class PathwayContextualizer(nn.Module):
    def __init__(self, n_genes: int = 20000, n_pathways: int = 137):
        super().__init__()
        self.gene_embed = nn.Linear(n_genes, 256)
        self.pathway_attention = AttentionBlock(256)
        self.pathway_classifier = nn.Linear(256, n_pathways)
        
    def forward(self, gene_expr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.gene_embed(gene_expr)
        x = self.pathway_attention(x)
        pathway_scores = self.pathway_classifier(x)
        return x, pathway_scores

class DrugEffectSimulator(nn.Module):
    def __init__(self, mol_dim: int = 128):
        super().__init__()
        if not DGL_AVAILABLE:
            # Fallback to simple MLP if DGL is not available
            self.mol_embed = nn.Sequential(
                nn.Linear(mol_dim, mol_dim),
                nn.ReLU(),
                nn.Linear(mol_dim, mol_dim)
            )
        else:
            self.mol_embed = nn.ModuleList([
                GATConv(mol_dim, mol_dim // 4, num_heads=4),
                nn.ReLU(),
                GATConv(mol_dim, mol_dim // 4, num_heads=4)
            ])
        
        self.kinetics_predictor = nn.Sequential(
            nn.Linear(mol_dim, mol_dim),
            nn.ReLU(),
            nn.Linear(mol_dim, 3)  # Binding, permeability, toxicity
        )
        
    def forward(self, mol_graph) -> Tuple[torch.Tensor, torch.Tensor]:
        if not DGL_AVAILABLE:
            x = self.mol_embed(mol_graph)
        else:
            x = mol_graph
            for layer in self.mol_embed:
                if isinstance(layer, GATConv):
                    x = layer(x).mean(1)  # Average over heads
                else:
                    x = layer(x)
        kinetics = self.kinetics_predictor(x)
        return x, kinetics

class EnvironmentIntegrator(nn.Module):
    def __init__(self, n_covariates: int = 78):
        super().__init__()
        self.covariate_embed = nn.Linear(n_covariates, 128)
        self.attention = AttentionBlock(128)
        self.effect_predictor = nn.Linear(128, 1)
        
    def forward(self, covariates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.covariate_embed(covariates)
        x = self.attention(x)
        effects = self.effect_predictor(x)
        return x, effects

class FusionModel(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.pathway_net = PathwayContextualizer(
            n_genes=config.get('n_genes', 20000),
            n_pathways=config.get('n_pathways', 137)
        )
        self.drug_net = DrugEffectSimulator(
            mol_dim=config.get('mol_dim', 128)
        )
        self.env_net = EnvironmentIntegrator(
            n_covariates=config.get('n_covariates', 78)
        )
        
        fusion_dim = 256 + 128 + 128
        self.fusion_attention = AttentionBlock(fusion_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                gene_expr: torch.Tensor,
                mol_graph,
                covariates: torch.Tensor
               ) -> Dict[str, torch.Tensor]:
        # Process each modality
        pathway_feats, pathway_scores = self.pathway_net(gene_expr)
        drug_feats, kinetics = self.drug_net(mol_graph)
        env_feats, env_effects = self.env_net(covariates)
        
        # Fusion
        fusion_input = torch.cat([pathway_feats, drug_feats, env_feats], dim=-1)
        fused = self.fusion_attention(fusion_input)
        response_prob = self.classifier(fused)
        
        return {
            'response_probability': response_prob,
            'pathway_scores': pathway_scores,
            'kinetics': kinetics,
            'environmental_effects': env_effects
        }
    
    def predict(self,
               gene_expr: np.ndarray,
               mol_graph,
               covariates: np.ndarray,
               threshold: float = 0.5
              ) -> Dict[str, np.ndarray]:
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                torch.FloatTensor(gene_expr),
                mol_graph,
                torch.FloatTensor(covariates)
            )
            predictions = {k: v.numpy() for k, v in outputs.items()}
            predictions['binary_response'] = predictions['response_probability'] > threshold
        return predictions