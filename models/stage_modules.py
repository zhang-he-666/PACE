import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch


class StageSegmenter(nn.Module):


    def __init__(self, stage_names=None, stage_clip_length=32):
        super().__init__()
        self.stage_names = stage_names or ("Forward", "Twist", "Entry")
        self.stage_clip_length = stage_clip_length

    def forward(self, video, boundaries):

        # Ensure video has correct shape
        if video.dim() != 5:
            raise ValueError(
                f"StageSegmenter expects video with 5 dimensions [B, C, T, H, W], "
                f"but received {video.dim()} dimensions with shape {video.shape}"
            )
        
        if boundaries.dim() == 1:
            boundaries = boundaries.unsqueeze(0)
        
        # Ensure boundaries have correct shape [B, num_stages-1]
        if boundaries.dim() != 2:
            raise ValueError(
                f"StageSegmenter expects boundaries with 2 dimensions [B, num_stages-1], "
                f"but received {boundaries.dim()} dimensions with shape {boundaries.shape}"
            )

        B, C, T, H, W = video.shape
        device = video.device
        num_stages = len(self.stage_names)
        # Ensure boundaries are on the same device and dtype as video indices
        boundaries = boundaries.to(device=device, dtype=torch.long)

        if boundaries.shape[1] != num_stages - 1:
            raise ValueError(
                f"StageSegmenter expects {num_stages-1} boundaries, "
                f"but received {boundaries.shape[1]}. "
                f"Boundaries shape: {boundaries.shape}, Video shape: {video.shape}"
            )
        
        # Ensure batch sizes match
        if boundaries.shape[0] != B:
            raise ValueError(
                f"Batch size mismatch: video has {B} batches, but boundaries has {boundaries.shape[0]} batches. "
                f"Video shape: {video.shape}, Boundaries shape: {boundaries.shape}"
            )

        boundaries = boundaries.long().clamp(0, T - 1)
        start_pad = torch.zeros(B, 1, device=device, dtype=torch.long)
        end_pad = torch.full((B, 1), T - 1, device=device, dtype=torch.long)

        starts = torch.cat([start_pad, boundaries], dim=1)
        ends = torch.cat([boundaries, end_pad], dim=1)

        # Ensure each stage covers at least 1 frame to avoid degenerate linspace
        ends = torch.maximum(ends, starts + 1).clamp(max=T - 1)

        # Vectorized temporal sampling for all stages/batches
        linspace = torch.linspace(
            0.0,
            1.0,
            steps=self.stage_clip_length,
            device=device,
            dtype=torch.float32,
        ).view(1, 1, -1)
        indices = starts.unsqueeze(-1) * (1 - linspace) + ends.unsqueeze(-1) * linspace
        indices = indices.round().long().clamp(0, T - 1)  # [B, num_stages, L]

        video_expanded = video.unsqueeze(1).expand(-1, num_stages, -1, -1, -1, -1)
        gather_index = (
            indices.unsqueeze(2).unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, C, -1, H, W
            )
        )
        stage_tensor = torch.gather(video_expanded, 3, gather_index)  # [B, S, C, L, H, W]

        stage_clips = OrderedDict(
            {
                name: stage_tensor[:, idx]
                for idx, name in enumerate(self.stage_names)
            }
        )
        return stage_clips

    def _sample_clip(self, video, start_idx, end_idx):
        # Ensure video has correct shape [B, C, T, H, W]
        if video.dim() != 5:
            raise ValueError(
                f"_sample_clip expects video with 5 dimensions [B, C, T, H, W], "
                f"but received {video.dim()} dimensions with shape {video.shape}"
            )
        
        _, _, total_frames, _, _ = video.shape
        if end_idx <= start_idx:
            end_idx = min(start_idx + 1, total_frames - 1)

        indices = torch.linspace(
            float(start_idx),
            float(end_idx),
            steps=self.stage_clip_length,
            device=video.device,
        )
        indices = indices.round().long().clamp(0, total_frames - 1)

        try:
            return video.index_select(2, indices)
        except RuntimeError as e:
            raise RuntimeError(
                f"Error in _sample_clip: video shape {video.shape}, "
                f"indices shape {indices.shape}, start_idx={start_idx}, end_idx={end_idx}, "
                f"total_frames={total_frames}, stage_clip_length={self.stage_clip_length}. "
                f"Original error: {e}"
            )


class StageGATModule(nn.Module):
    
    def __init__(
        self,
        in_dim=1024,
        hidden_dim=512,
        projection_dim=64,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        
        self.input_norm = nn.LayerNorm(in_dim)
        
       
        self.gat1 = GATv2Conv(
            in_channels=in_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True,
            add_self_loops=True,  # Allow self-loops for stage self-attention
        )
        

        self.gat2 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True,
            add_self_loops=True,
        )
        
 
        self.context_norm = nn.LayerNorm(hidden_dim)
 
        self.graph_proj = nn.Linear(hidden_dim, projection_dim)
        self.graph_norm = nn.LayerNorm(projection_dim)
        
   
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        

        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(projection_dim),
            nn.Linear(projection_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim),
        )
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, stage_feature, edge_index):
        """
        Args:
            stage_feature: Tensor [B, in_dim] - single stage feature
            edge_index: Tensor [2, num_edges] - graph edges connecting stages
        Returns:
            stage_graph: [B, projection_dim]
            stage_score: [B, 1]
            reconstructed: [B, in_dim]
        """

        normalized = self.input_norm(stage_feature)
        

        h1 = self.gat1(normalized, edge_index)
        h1 = F.elu(h1)
        h1 = self.dropout(h1)
        

        h2 = self.gat2(h1, edge_index)
        h2 = F.elu(h2)
        h2 = self.dropout(h2)
        

        context = self.context_norm(h2)
        
  
        stage_graph = self.graph_proj(context)
        stage_graph = self.graph_norm(stage_graph)
        
 
        stage_score = self.score_head(context)
        

        reconstructed = self.reconstruction_head(stage_graph)
        
        return stage_graph, stage_score, reconstructed


class StageCausalReasoner(nn.Module):


    def __init__(
        self,
        in_dim=1024,
        hidden_dim=512,
        projection_dim=64,
        num_heads=4,
        num_stages=3,
        dropout=0.1,
        use_5node_subgraph=True,  # Ablation flag
        use_inter_stage_intervention=True,  # Ablation flag
        use_foreground=True,  # Ablation flag: w/o F_k
        use_body=True,  # Ablation flag: w/o B_k
        use_appearance=True,  # Ablation flag: w/o A_k
        use_motion=True,  # Ablation flag: w/o M_k
        use_saliency=True,  # Ablation flag: w/o S_k
        use_directed_graph=True,  # Ablation flag: directed vs undirected
    ):
        super().__init__()
        self.num_stages = num_stages
        self.num_heads = num_heads
        self.use_5node_subgraph = use_5node_subgraph
        self.use_inter_stage_intervention = use_inter_stage_intervention
        
        if use_5node_subgraph:
        
            from models.intra_stage_subgraph import MultiStageSubgraphReasoner
            from models.gated_inter_stage_intervention import (
                GatedInterStageIntervention,
                AggregateStageFeatures
            )
            
            self.subgraph_reasoner = MultiStageSubgraphReasoner(
                in_dim=in_dim,
                node_dim=hidden_dim,  # Use hidden_dim as node_dim
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_gnn_layers=2,
                num_stages=num_stages,
                dropout=dropout,
                use_foreground=use_foreground,
                use_body=use_body,
                use_appearance=use_appearance,
                use_motion=use_motion,
                use_saliency=use_saliency,
                use_directed_graph=use_directed_graph
            )
            
            # Gated inter-stage intervention
            if use_inter_stage_intervention:
                self.inter_stage_intervention = GatedInterStageIntervention(
                    feature_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    use_gradient_detachment=True
                )
            
            # Aggregate 5 nodes to final stage representation
            self.stage_aggregator = AggregateStageFeatures(
                node_dim=hidden_dim,
                output_dim=projection_dim
            )
            
        else:
            self.forward_gat = StageGATModule(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                projection_dim=projection_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            self.twist_gat = StageGATModule(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                projection_dim=projection_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            self.entry_gat = StageGATModule(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                projection_dim=projection_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
        
        # Reconstruction head (compatible with both modes)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim)
        )

    def forward(self, stage_features):

        B, S, in_dim = stage_features.shape
        
        extra_outputs = {}
        
        if self.use_5node_subgraph:
            all_node_dicts, stage_graphs_raw, stage_saliencies = self.subgraph_reasoner(
                stage_features
            )
            # all_node_dicts: list of 3 dicts, each with 5 nodes [B, hidden_dim]
            # stage_graphs_raw: [B, 3, hidden_dim]
            # stage_saliencies: [B, 3, 1]
            
            # Store original nodes before intervention for causal loss computation
            original_node_dicts = [
                {k: v.clone() for k, v in node_dict.items()} 
                for node_dict in all_node_dicts
            ]
            
            # Step 2: Apply gated inter-stage causal intervention
            # Always use prevent_backward=True to ensure strict causality in both training and inference
            if self.use_inter_stage_intervention:
                intervened_node_dicts, intervention_gates = self.inter_stage_intervention(
                    all_node_dicts, stage_saliencies, prevent_backward=True  # Always True for strict causality
                )
                extra_outputs['intervention_gates'] = intervention_gates
                extra_outputs['original_node_dicts'] = original_node_dicts  # Store for causal loss
            else:
                intervened_node_dicts = all_node_dicts
                extra_outputs['original_node_dicts'] = original_node_dicts
            
            # Step 3: Aggregate 5 nodes to final stage representation
            stage_graphs = []
            for node_dict in intervened_node_dicts:
                aggregated = self.stage_aggregator(node_dict)  # [B, projection_dim]
                stage_graphs.append(aggregated)
            stage_graph = torch.stack(stage_graphs, dim=1)  # [B, 3, projection_dim]
            
            # Step 4: Stage scores (saliencies)
            stage_scores = stage_saliencies  # [B, 3, 1]
            
            # Step 5: Reconstruction
            reconstructed = self.reconstruction_head(stage_graph)  # [B, 3, in_dim]
            
            # Store node features for visualization/loss
            extra_outputs['all_node_dicts'] = intervened_node_dicts
            extra_outputs['original_node_dicts'] = original_node_dicts  # Store original for causal loss
            extra_outputs['stage_saliencies'] = stage_saliencies
            
        else:
            stage_features_flat = stage_features.view(B * S, in_dim)
            
            stage_graphs = []
            stage_scores = []
            reconstructions = []
            
            for stage_idx in range(S):
                stage_feat = stage_features_flat[stage_idx::S]
                
                stage_edges = []
                for i in range(B):
                    for j in range(B):
                        stage_edges.append([i, j])
                stage_edge_idx = torch.tensor(
                    stage_edges, dtype=torch.long, device=stage_features.device
                ).t().contiguous()
                
                if stage_idx == 0:
                    gat_module = self.forward_gat
                elif stage_idx == 1:
                    gat_module = self.twist_gat
                else:
                    gat_module = self.entry_gat
                
                stage_graph_i, stage_score_i, reconstructed_i = gat_module(
                    stage_feat, stage_edge_idx
                )
                
                stage_graphs.append(stage_graph_i)
                stage_scores.append(stage_score_i)
                reconstructions.append(reconstructed_i)
            
            stage_graph = torch.stack(stage_graphs, dim=1)
            stage_scores = torch.stack(stage_scores, dim=1)
            reconstructed = torch.stack(reconstructions, dim=1)
        
        # Dummy attention matrix for compatibility
        attention = torch.eye(S, device=stage_features.device).unsqueeze(0).repeat(B, 1, 1)
        
        return stage_graph, attention, stage_scores, reconstructed, extra_outputs

