import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any

class GraphEncoder(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.mol_dim = config['mol_dim']
        self.hidden_dim = config['graph_hidden_dim']
        self.out_dim = config['graph_out_dim']
        self.num_layers = config['graph_layers']
        self.dropout = config['dropout']
        
        # Node feature embedding
        self.node_embedding = nn.Linear(self.mol_dim, self.hidden_dim)
        
        # Graph convolutional layers
        self.conv_layers = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim)
            for _ in range(self.num_layers)
        ])
        
        # Output projection
        self.output_layer = nn.Linear(self.hidden_dim, self.out_dim)
        
        self.dropout_layer = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        
    def message_passing(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Perform message passing between nodes"""
        # Compute messages
        messages = torch.matmul(adj, x)
        
        # Add self-connection
        return messages + x
    
    def forward(self, node_features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: Tensor of shape [batch_size, num_nodes, num_features]
            adj_matrix: Tensor of shape [batch_size, num_nodes, num_nodes]
        """
        # Initial node embedding
        x = self.node_embedding(node_features)
        
        # Graph convolution layers
        for conv in self.conv_layers:
            # Message passing
            messages = self.message_passing(x, adj_matrix)
            
            # Update node features
            x = conv(messages)
            x = F.relu(x)
            x = self.dropout_layer(x)
            x = self.layer_norm(x)
        
        # Global pooling (mean of node features)
        x = torch.mean(x, dim=1)
        
        # Project to output dimension
        return self.output_layer(x)

class ClinicalEncoder(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.clinical_dim = config['clinical_dim']
        self.hidden_dim = config['clinical_hidden_dim']
        self.num_layers = config['clinical_layers']
        self.dropout = config['dropout']
        self.max_seq_len = config.get('max_seq_len', 100)
        
        # Input embedding with positional encoding
        self.input_embedding = nn.Sequential(
            nn.Linear(self.clinical_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # Positional encoding
        self.register_buffer(
            'pos_encoding',
            self._create_positional_encoding(self.max_seq_len, self.hidden_dim)
        )
        
        # Bi-LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Transformer encoder layers for sequence modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=2 * self.hidden_dim,  # 2x for bidirectional
            nhead=8,
            dim_feedforward=4 * self.hidden_dim,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN transformer
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )
        
        # Attention pooling
        self.attention_pooling = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encodings"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
        
    def forward(self, 
                clinical_data: torch.Tensor, 
                lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clinical_data: Tensor of shape [batch_size, seq_len, features]
            lengths: Tensor of shape [batch_size] with sequence lengths
        """
        batch_size, seq_len = clinical_data.shape[:2]
        
        # Input embedding + positional encoding
        x = self.input_embedding(clinical_data)
        x = x + self.pos_encoding[:seq_len]
        
        # Pack sequence for LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            x, 
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # Process through LSTM
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        # Create padding mask for transformer
        # Generate mask based on sequence lengths
        padding_mask = torch.arange(lstm_out.size(1), device=lstm_out.device)[None, :] >= lengths[:, None]
        
        # Process through transformer
        transformed = self.transformer(lstm_out, src_key_padding_mask=padding_mask)
        
        # Attention pooling with mask
        attention_weights = self.attention_pooling(transformed)
        attention_weights = attention_weights.masked_fill(padding_mask.unsqueeze(-1), 0)
        attention_weights = attention_weights / attention_weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
        
        # Weighted sum
        pooled = torch.bmm(attention_weights.transpose(1, 2), transformed).squeeze(1)
        
        # Final projection
        return self.output_projection(pooled)

class MultiHeadAttention(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.fusion_dim = config['fusion_dim']
        self.num_heads = config['num_attention_heads']
        self.head_dim = self.fusion_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.fusion_dim, "Fusion dimension must be divisible by number of heads"
        
        self.q_linear = nn.Linear(self.fusion_dim, self.fusion_dim)
        self.k_linear = nn.Linear(self.fusion_dim, self.fusion_dim)
        self.v_linear = nn.Linear(self.fusion_dim, self.fusion_dim)
        
        self.output_linear = nn.Linear(self.fusion_dim, self.fusion_dim)
        self.layer_norm = nn.LayerNorm(self.fusion_dim)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        batch_size = q.size(0)
        
        # Linear projections and reshape for attention heads
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())
        attn = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape and project to output dimension
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.fusion_dim)
        out = self.output_linear(out)
        
        # Add residual connection and layer normalization
        return self.layer_norm(out)

class GNNLSTMFusion(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.graph_encoder = GraphEncoder(config)
        self.clinical_encoder = ClinicalEncoder(config)
        
        # Project encoded features to common fusion space
        self.graph_projection = nn.Linear(config['graph_out_dim'], config['fusion_dim'])
        self.clinical_projection = nn.Linear(config['clinical_hidden_dim'], config['fusion_dim'])
        
        # Multihead attention for feature fusion
        self.attention = MultiHeadAttention(config)
        
        # Output layers
        self.fusion_mlp = nn.Sequential(
            nn.Linear(config['fusion_dim'], config['fusion_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['fusion_dim'] // 2, config['fusion_dim'] // 4),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['fusion_dim'] // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                mol_features: torch.Tensor,
                mol_adj: torch.Tensor,
                clinical_features: torch.Tensor,
                clinical_lengths: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the fusion model
        
        Args:
            mol_features: Molecular graph node features [batch_size, num_nodes, node_features]
            mol_adj: Molecular graph adjacency matrices [batch_size, num_nodes, num_nodes]
            clinical_features: Clinical time series data [batch_size, seq_len, features]
            clinical_lengths: Sequence lengths [batch_size]
            
        Returns:
            Dictionary containing model outputs and attention weights
        """
        # Encode molecular graphs
        graph_encoding = self.graph_encoder(mol_features, mol_adj)
        graph_features = self.graph_projection(graph_encoding)
        
        # Encode clinical sequences
        clinical_encoding = self.clinical_encoder(clinical_features, clinical_lengths)
        clinical_features = self.clinical_projection(clinical_encoding)
        
        # Reshape for attention
        graph_features = graph_features.unsqueeze(1)
        clinical_features = clinical_features.unsqueeze(1)
        
        # Apply multi-head attention for feature fusion
        fused_features = self.attention(
            graph_features,
            clinical_features,
            clinical_features
        )
        
        # Generate predictions
        predictions = self.fusion_mlp(fused_features.squeeze(1))
        
        return {
            'predictions': predictions,
            'graph_features': graph_features,
            'clinical_features': clinical_features,
            'fused_features': fused_features
        }