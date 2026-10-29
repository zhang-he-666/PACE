import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data


class ForegroundExtractor(nn.Module):
    """Extract foreground action features using attention-based separation"""
    
    def __init__(self, in_dim=1024, hidden_dim=512, out_dim=256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
    
    def forward(self, stage_feature):
 
        # Foreground-background attention
        fg_attention = self.attention(stage_feature)  # [B, 1]
        
        # Apply attention to extract foreground
        weighted_feature = stage_feature * fg_attention
        F_k = self.feature_extractor(weighted_feature)
        
        return F_k


class BodyFeatureExtractor(nn.Module):

    
    def __init__(self, in_dim=1024, hidden_dim=512, out_dim=256):
        super().__init__()
        self.in_dim = in_dim
        
        # Channel attention for body-relevant features (without AdaptiveAvgPool1d)
        # Simplified channel attention to avoid dimension issues
        self.channel_gate = nn.Sequential(
            nn.Linear(in_dim, in_dim // 16),
            nn.ReLU(),
            nn.Linear(in_dim // 16, in_dim),
            nn.Sigmoid()
        )
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
    
    def forward(self, stage_feature):
 
        # Channel attention (directly on the feature vector)
        attn = self.channel_gate(stage_feature)  # [B, in_dim]
        weighted_feature = stage_feature * attn
        
        B_k = self.feature_extractor(weighted_feature)
        return B_k


class AppearanceExtractor(nn.Module):

    
    def __init__(self, in_dim=1024, hidden_dim=512, out_dim=256):
        super().__init__()
        # Appearance features are typically from higher semantic levels
        self.appearance_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
    
    def forward(self, stage_feature):

        A_k = self.appearance_encoder(stage_feature)
        return A_k


class MotionExtractor(nn.Module):

    
    def __init__(self, in_dim=1024, hidden_dim=512, out_dim=256):
        super().__init__()
        # Motion encoder with temporal awareness
        self.motion_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        
        # Optional: Temporal difference weights
        self.temporal_weight = nn.Parameter(torch.ones(1))
    
    def forward(self, stage_feature, prev_stage_feature=None):

        if prev_stage_feature is not None:
            # Temporal difference as motion cue
            temporal_diff = stage_feature - prev_stage_feature
            motion_input = stage_feature + self.temporal_weight * temporal_diff
        else:
            motion_input = stage_feature
        
        M_k = self.motion_encoder(motion_input)
        return M_k


class SaliencyScorer(nn.Module):

    
    def __init__(self, in_dim=1024, hidden_dim=256):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output probability [0, 1]
        )
    
    def forward(self, stage_feature):

        S_k = self.scorer(stage_feature)
        return S_k


class CausalGNN(nn.Module):

    
    def __init__(self, node_dim=256, hidden_dim=256, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        
        # GAT layers for causal message passing
        self.gat_layers = nn.ModuleList([
            GATv2Conv(
                in_channels=node_dim if i == 0 else hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                concat=True,
                add_self_loops=False  # We define edges explicitly
            )
            for i in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, nodes, edge_index):

        B, num_nodes, node_dim = nodes.shape
        
        # Flatten batch for GNN processing
        x = nodes.view(B * num_nodes, node_dim)
        
        # Adjust edge_index for batched processing
        batch_edge_index = []
        for b in range(B):
            batch_edge_index.append(edge_index + b * num_nodes)
        batch_edge_index = torch.cat(batch_edge_index, dim=1)
        
        # Multi-layer GNN with causal message passing
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            x_new = gat(x, batch_edge_index)
            x_new = F.elu(x_new)
            x_new = self.dropout(x_new)
            x_new = norm(x_new)
            
            # Residual connection (if dimensions match)
            if x.size(-1) == x_new.size(-1):
                x = x + x_new
            else:
                x = x_new
        
        # Reshape back to [B, num_nodes, hidden_dim]
        refined_nodes = x.view(B, num_nodes, -1)
        
        return refined_nodes


class IntraStageSubgraph(nn.Module):

    
    def __init__(
        self,
        in_dim=1024,
        node_dim=256,
        hidden_dim=256,
        num_heads=4,
        num_gnn_layers=2,
        dropout=0.1,
        use_foreground=True,  # Ablation flag: w/o F_k
        use_body=True,  # Ablation flag: w/o B_k
        use_appearance=True,  # Ablation flag: w/o A_k
        use_motion=True,  # Ablation flag: w/o M_k
        use_saliency=True,  # Ablation flag: w/o S_k
        use_directed_graph=True,  # Ablation flag: directed vs undirected
    ):
        super().__init__()
        
        self.use_foreground = use_foreground
        self.use_body = use_body
        self.use_appearance = use_appearance
        self.use_motion = use_motion
        self.use_saliency = use_saliency
        self.use_directed_graph = use_directed_graph
        self.node_dim = node_dim  # Store for later use
        
        # 5 explicit node extractors (can be disabled for ablation)
        if use_foreground:
            self.foreground_branch = ForegroundExtractor(in_dim, hidden_dim, node_dim)
        else:
            self.foreground_branch = None
        
        if use_body:
            self.body_branch = BodyFeatureExtractor(in_dim, hidden_dim, node_dim)
        else:
            self.body_branch = None
        
        if use_appearance:
            self.appearance_branch = AppearanceExtractor(in_dim, hidden_dim, node_dim)
        else:
            self.appearance_branch = None
        
        if use_motion:
            self.motion_branch = MotionExtractor(in_dim, hidden_dim, node_dim)
        else:
            self.motion_branch = None
        
        if use_saliency:
            self.saliency_branch = SaliencyScorer(in_dim, hidden_dim)
        else:
            self.saliency_branch = None
        
        self.saliency_projection = nn.Linear(1, node_dim)

        if use_directed_graph:
            # Directed causal graph (DAG) with improved structure
            causal_edges = [
                [2, 0], 
                [2, 1],  
                [0, 1], 
                [1, 0], 
                [3, 4], 
                [3, 0], 
                [3, 1], 
                [1, 4], 
            ]
        else:
            # Undirected graph (for ablation: undirected vs directed)
            causal_edges = [
                [2, 0], [0, 2], 
                [2, 1], [1, 2],  
                [0, 1], [1, 0],  
                [3, 4], [4, 3], 
                [3, 0], [0, 3],  
                [3, 1], [1, 3], 
                [1, 4], [4, 1],  
            ]
        
        self.register_buffer(
            'causal_edge_index',
            torch.tensor(causal_edges, dtype=torch.long).t().contiguous()
        )
        
        # Causal GNN
        self.causal_gnn = CausalGNN(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_gnn_layers,
            dropout=dropout
        )
        
        self.output_projection = nn.Linear(hidden_dim, node_dim)
        

        self.node_attention = nn.Sequential(
            nn.Linear(node_dim, node_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim // 2, 1),
            nn.Softmax(dim=1)  # Softmax over 5 nodes
        )
    
    def forward(self, stage_feature, prev_stage_feature=None):

        B = stage_feature.size(0)
        device = stage_feature.device
        
        # Step 1: Extract 5 explicit nodes (with ablation support)
        node_list = []
        node_indices = []
        
        if self.use_foreground and self.foreground_branch is not None:
            F_k = self.foreground_branch(stage_feature)  # [B, node_dim]
            node_list.append(F_k)
            node_indices.append(0)  # F_k index
        else:
            F_k = torch.zeros(B, self.node_dim if hasattr(self, 'node_dim') else 256, device=device)
            node_list.append(F_k)
            node_indices.append(0)
        
        if self.use_body and self.body_branch is not None:
            B_k = self.body_branch(stage_feature)  # [B, node_dim]
            node_list.append(B_k)
            node_indices.append(1)  # B_k index
        else:
            B_k = torch.zeros(B, self.node_dim if hasattr(self, 'node_dim') else 256, device=device)
            node_list.append(B_k)
            node_indices.append(1)
        
        if self.use_appearance and self.appearance_branch is not None:
            A_k = self.appearance_branch(stage_feature)  # [B, node_dim]
            node_list.append(A_k)
            node_indices.append(2)  # A_k index
        else:
            A_k = torch.zeros(B, self.node_dim if hasattr(self, 'node_dim') else 256, device=device)
            node_list.append(A_k)
            node_indices.append(2)
        
        if self.use_motion and self.motion_branch is not None:
            M_k = self.motion_branch(stage_feature, prev_stage_feature)  # [B, node_dim]
            node_list.append(M_k)
            node_indices.append(3)  # M_k index
        else:
            M_k = torch.zeros(B, self.node_dim if hasattr(self, 'node_dim') else 256, device=device)
            node_list.append(M_k)
            node_indices.append(3)
        
        if self.use_saliency and self.saliency_branch is not None:
            S_k = self.saliency_branch(stage_feature)  # [B, 1]
            # Project saliency to node_dim for graph processing
            S_k_proj = self.saliency_projection(S_k)  # [B, node_dim]
            node_list.append(S_k_proj)
            node_indices.append(4)  # S_k index
        else:
            S_k = torch.zeros(B, 1, device=device)
            S_k_proj = torch.zeros(B, self.node_dim if hasattr(self, 'node_dim') else 256, device=device)
            node_list.append(S_k_proj)
            node_indices.append(4)
        

        nodes = torch.stack(node_list, dim=1)  # [B, 5, node_dim]
        
    
        refined_nodes = self.causal_gnn(nodes, self.causal_edge_index)  # [B, 5, hidden_dim]
        

        refined_nodes = self.output_projection(refined_nodes)  # [B, 5, node_dim]
        
   
        F_k_refined = refined_nodes[:, 0]  # [B, node_dim]
        B_k_refined = refined_nodes[:, 1]
        A_k_refined = refined_nodes[:, 2]
        M_k_refined = refined_nodes[:, 3]
        S_k_refined = refined_nodes[:, 4]
        
 
        all_nodes = torch.stack([
            F_k_refined,  # [B, node_dim]
            B_k_refined,
            A_k_refined,
            M_k_refined,
            S_k_refined
        ], dim=1)  # [B, 5, node_dim]
        
   
        node_attn_weights = self.node_attention(all_nodes)  # [B, 5, 1]
        
     
        stage_graph = (all_nodes * node_attn_weights).sum(dim=1)  # [B, node_dim]
        
 
        saliency_gate = S_k.unsqueeze(-1)  # [B, 1, 1]
        motion_enhanced = M_k_refined * saliency_gate.squeeze(-1)  # [B, node_dim]
        
 
        stage_graph = stage_graph + 0.1 * motion_enhanced
        
        # Return as dictionary for easy access
        node_dict = {
            'F_k': F_k_refined,
            'B_k': B_k_refined,
            'A_k': A_k_refined,
            'M_k': M_k_refined,
            'S_k': S_k_refined,
        }
        
        return node_dict, stage_graph, S_k


class MultiStageSubgraphReasoner(nn.Module):

    
    def __init__(
        self,
        in_dim=1024,
        node_dim=256,
        hidden_dim=256,
        num_heads=4,
        num_gnn_layers=2,
        num_stages=3,
        dropout=0.1,
        use_foreground=True,  # Ablation flag
        use_body=True,  # Ablation flag
        use_appearance=True,  # Ablation flag
        use_motion=True,  # Ablation flag
        use_saliency=True,  # Ablation flag
        use_directed_graph=True,  # Ablation flag: directed vs undirected
    ):
        super().__init__()
        self.num_stages = num_stages
        
        # Three independent subgraphs for three stages
        self.forward_subgraph = IntraStageSubgraph(
            in_dim, node_dim, hidden_dim, num_heads, num_gnn_layers, dropout,
            use_foreground, use_body, use_appearance, use_motion, use_saliency, use_directed_graph
        )
        self.twist_subgraph = IntraStageSubgraph(
            in_dim, node_dim, hidden_dim, num_heads, num_gnn_layers, dropout,
            use_foreground, use_body, use_appearance, use_motion, use_saliency, use_directed_graph
        )
        self.entry_subgraph = IntraStageSubgraph(
            in_dim, node_dim, hidden_dim, num_heads, num_gnn_layers, dropout,
            use_foreground, use_body, use_appearance, use_motion, use_saliency, use_directed_graph
        )
    
    def forward(self, stage_features):

        B, num_stages, in_dim = stage_features.shape
        
        # Process Forward stage (no previous stage)
        forward_nodes, forward_graph, forward_saliency = self.forward_subgraph(
            stage_features[:, 0], prev_stage_feature=None
        )
        
        # Process Twist stage (with Forward as previous)
        twist_nodes, twist_graph, twist_saliency = self.twist_subgraph(
            stage_features[:, 1], prev_stage_feature=stage_features[:, 0]
        )
        
        # Process Entry stage (with Twist as previous)
        entry_nodes, entry_graph, entry_saliency = self.entry_subgraph(
            stage_features[:, 2], prev_stage_feature=stage_features[:, 1]
        )
        
        # Aggregate results
        all_node_dicts = [forward_nodes, twist_nodes, entry_nodes]
        
        stage_graphs = torch.stack([forward_graph, twist_graph, entry_graph], dim=1)
        stage_saliencies = torch.stack([forward_saliency, twist_saliency, entry_saliency], dim=1)
        
        return all_node_dicts, stage_graphs, stage_saliencies

