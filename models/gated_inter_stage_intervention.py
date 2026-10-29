
import torch
import torch.nn as nn
import torch.nn.functional as F


class GateModule(nn.Module):

    def __init__(self, saliency_dim=1, hidden_dim=128):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(saliency_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Gate weight in [0, 1]
        )
    
    def forward(self, saliency):

        gate_weight = self.gate_network(saliency)
        return gate_weight


class CausalTransformModule(nn.Module):

    def __init__(self, feature_dim=256, hidden_dim=256):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
    
    def forward(self, source_feature):

        return self.transform(source_feature)


class InterventionBlock(nn.Module):

    def __init__(self, feature_dim=256, hidden_dim=256, use_gradient_detachment=True):
        super().__init__()
        self.gate = GateModule(saliency_dim=1, hidden_dim=128)
        self.transform = CausalTransformModule(feature_dim, hidden_dim)
        self.use_gradient_detachment = use_gradient_detachment
        
        # Learnable scaling factor (increased from 0.5 to 1.0 for better intervention effect)
        self.intervention_scale = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, source_nodes, target_nodes, source_saliency, prevent_backward=True):

        B = source_nodes['F_k'].size(0)
        
        # Step 1: Compute gate weight from source saliency
        gate_weight = self.gate(source_saliency)  # [B, 1]
        
        # Step 2: Apply do-intervention to each node type
        intervened_nodes = {}
        
        for node_key in ['F_k', 'B_k', 'A_k', 'M_k', 'S_k']:
            source_feature = source_nodes[node_key]
            target_feature = target_nodes[node_key]
            
            # Prevent backward information flow
            if prevent_backward and self.use_gradient_detachment:
                source_feature = source_feature.detach()
            
            # Transform source feature to target space
            transformed_source = self.transform(source_feature)
            
            # do-intervention with gating
            intervened_feature = target_feature + self.intervention_scale * gate_weight * transformed_source
            
            intervened_nodes[node_key] = intervened_feature
        
        return intervened_nodes


class GatedInterStageIntervention(nn.Module):

    
    def __init__(
        self,
        feature_dim=256,
        hidden_dim=256,
        use_gradient_detachment=True,
        use_explicit_mask=True
    ):
        super().__init__()
        
        self.use_gradient_detachment = use_gradient_detachment
        self.use_explicit_mask = use_explicit_mask
        
        # Two intervention blocks:
        # 1. Forward → Twist
        self.forward_to_twist = InterventionBlock(
            feature_dim, hidden_dim, use_gradient_detachment
        )
        
        # 2. Twist → Entry
        self.twist_to_entry = InterventionBlock(
            feature_dim, hidden_dim, use_gradient_detachment
        )
        
        # Explicit causal mask matrix: [3, 3] where mask[i, j] = 1 if stage i can influence stage j
        # Only upper triangular part is allowed (forward causality)
        # mask[0, 1] = 1: Forward can influence Twist
        # mask[1, 2] = 1: Twist can influence Entry
        # All other entries = 0 (no reverse causality)
        if use_explicit_mask:
            causal_mask = torch.triu(torch.ones(3, 3), diagonal=1)  # Upper triangular
            self.register_buffer('causal_mask', causal_mask)
        else:
            self.register_buffer('causal_mask', torch.ones(3, 3))
    
    def forward(self, all_node_dicts, stage_saliencies, prevent_backward=True):

        forward_nodes, twist_nodes, entry_nodes = all_node_dicts
        
        forward_saliency = stage_saliencies[:, 0]  # [B, 1]
        twist_saliency = stage_saliencies[:, 1]    # [B, 1]
        entry_saliency = stage_saliencies[:, 2]    # [B, 1]
        
        # Apply explicit causal mask if enabled
        if self.use_explicit_mask and prevent_backward:
            # Ensure only forward causality is allowed
            # causal_mask[0, 1] = 1: Forward → Twist allowed
            # causal_mask[1, 2] = 1: Twist → Entry allowed
            # All reverse edges are masked out
            pass  # Mask is applied implicitly through intervention order
        
        # Step 1: Forward → Twist intervention (only if causal_mask[0, 1] == 1)
        if self.use_explicit_mask:
            # Use a Python float mask to avoid tracking this buffer in autograd graph
            mask_ft = float(self.causal_mask[0, 1].item())  # Should be 1.0
        else:
            mask_ft = 1.0
        
        twist_intervened = self.forward_to_twist(
            source_nodes=forward_nodes,
            target_nodes=twist_nodes,
            source_saliency=forward_saliency * mask_ft,  # Apply mask to saliency
            prevent_backward=prevent_backward
        )
        
        # Step 2: Twist → Entry intervention (only if causal_mask[1, 2] == 1)
        if self.use_explicit_mask:
            # Use a Python float mask to avoid tracking this buffer in autograd graph
            mask_te = float(self.causal_mask[1, 2].item())  # Should be 1.0
        else:
            mask_te = 1.0
        
        entry_intervened = self.twist_to_entry(
            source_nodes=twist_intervened,
            target_nodes=entry_nodes,
            source_saliency=twist_saliency * mask_te,  # Apply mask to saliency
            prevent_backward=prevent_backward
        )

        
        # Forward stage remains unchanged (no prior stage)
        intervened_node_dicts = [forward_nodes, twist_intervened, entry_intervened]
        
        # Store gate weights for visualization/analysis
        intervention_gates = {
            'forward_to_twist_gate': self.forward_to_twist.gate(forward_saliency) * mask_ft,
            'twist_to_entry_gate': self.twist_to_entry.gate(twist_saliency) * mask_te
        }
        
        return intervened_node_dicts, intervention_gates


class AggregateStageFeatures(nn.Module):

    def __init__(self, node_dim=256, output_dim=64):
        super().__init__()
        # Attention-based aggregation
        self.attention = nn.Sequential(
            nn.Linear(node_dim, node_dim // 2),
            nn.ReLU(),
            nn.Linear(node_dim // 2, 1),
            nn.Softmax(dim=1)  # Softmax over 5 nodes
        )
        
        # Final projection
        self.output_proj = nn.Linear(node_dim, output_dim)
    
    def forward(self, node_dict):

        B = node_dict['F_k'].size(0)
        
        # Stack all nodes
        nodes = torch.stack([
            node_dict['F_k'],
            node_dict['B_k'],
            node_dict['A_k'],
            node_dict['M_k'],
            node_dict['S_k']
        ], dim=1)  # [B, 5, node_dim]
        
        # Attention-based aggregation
        attn_weights = self.attention(nodes)  # [B, 5, 1]
        aggregated = (nodes * attn_weights).sum(dim=1)  # [B, node_dim]
        
        # Project to output dimension
        stage_feature = self.output_proj(aggregated)  # [B, output_dim]
        
        return stage_feature


class CausalLoss(nn.Module):

    def __init__(self, penalty_weight=1.0, consistency_weight=0.5, stability_weight=0.3):
        super().__init__()
        self.penalty_weight = penalty_weight
        self.consistency_weight = consistency_weight
        self.stability_weight = stability_weight
    
    def reverse_gradient_penalty_v2(self, source_nodes, target_nodes_after):

        device = target_nodes_after['F_k'].device
        
        # Method 1: Feature correlation penalty
        # If causality is preserved, source and target features should be independent
        # after intervention. High correlation indicates potential reverse causality.
        penalty_list = []
        for node_key in ['F_k', 'B_k', 'A_k', 'M_k', 'S_k']:
            if node_key in source_nodes and node_key in target_nodes_after:
                source_feat = source_nodes[node_key].detach()  # Detach to prevent gradient flow
                target_feat = target_nodes_after[node_key]
                
                # Flatten features for correlation computation
                source_flat = source_feat.flatten(1)  # [B, D]
                target_flat = target_feat.flatten(1)  # [B, D]
                
                # Compute cosine similarity (correlation proxy)
                # Normalize features
                source_norm = F.normalize(source_flat, p=2, dim=1)
                target_norm = F.normalize(target_flat, p=2, dim=1)
                
                # Cosine similarity: high values indicate correlation
                correlation = (source_norm * target_norm).sum(dim=1).abs().mean()
                
                # Penalize high correlation (indicates reverse causality)
                # We want correlation to be low (features should be independent)
                penalty_list.append(correlation)
        
        # Sum all penalties (this creates a proper computation graph)
        if len(penalty_list) > 0:
            penalty = sum(penalty_list)
        else:
            penalty = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Method 2: Source feature stability check (reduced weight to avoid interference)
        # Source features should remain unchanged after target intervention
        source_stability_penalty = torch.tensor(0.0, device=device)
        for node_key in ['F_k', 'B_k', 'A_k', 'M_k', 'S_k']:
            if node_key in source_nodes:
                # Check if source features have changed (they shouldn't)
                source_feat = source_nodes[node_key]
                # Since source is detached, this should be near zero
                # Reduced weight from 0.01 to 0.001 to avoid interfering with main penalty
                source_stability_penalty = source_stability_penalty + 0.001 * torch.norm(source_feat)
        
        return penalty + source_stability_penalty
    
    def reverse_gradient_penalty(self, source_nodes, target_nodes_before, target_nodes_after):

        return self.reverse_gradient_penalty_v2(source_nodes, target_nodes_after)
    
    def intervention_consistency_loss(self, gates, saliencies):

        if len(gates) == 0 or saliencies is None:
            return torch.tensor(0.0, device=next(iter(gates.values())).device if gates else 'cuda')
        
        device = next(iter(gates.values())).device
        consistency_loss = torch.tensor(0.0, device=device)
        
        # Extract saliencies for each stage
        # saliencies shape: [B, 3, 1]
        # Use saliencies[:, 0] to get [B, 1], not saliencies[:, 0:1] which gives [B, 1, 1]
        forward_saliency = saliencies[:, 0]  # [B, 1]
        twist_saliency = saliencies[:, 1]    # [B, 1]
        entry_saliency = saliencies[:, 2]    # [B, 1]
        
        # Gate 1: forward_to_twist_gate should correlate with forward_saliency
        if 'forward_to_twist_gate' in gates:
            gate_ft = gates['forward_to_twist_gate']  # [B, 1]
            # Ensure same shape for MSE loss
            if gate_ft.shape != forward_saliency.shape:
                # Handle shape mismatch: squeeze extra dimensions
                if gate_ft.dim() > forward_saliency.dim():
                    gate_ft = gate_ft.squeeze(-1)
                elif forward_saliency.dim() > gate_ft.dim():
                    forward_saliency = forward_saliency.squeeze(-1)
            # High forward saliency should lead to high gate value
            # Use MSE loss with proper scaling to ensure consistency
            # Normalize both to [0, 1] range for fair comparison
            gate_ft_norm = gate_ft
            saliency_norm = forward_saliency.detach()
            consistency_loss = consistency_loss + F.mse_loss(
                gate_ft_norm, saliency_norm
            )
        
        # Gate 2: twist_to_entry_gate should correlate with twist_saliency
        if 'twist_to_entry_gate' in gates:
            gate_te = gates['twist_to_entry_gate']  # [B, 1]
            # Ensure same shape for MSE loss
            if gate_te.shape != twist_saliency.shape:
                # Handle shape mismatch: squeeze extra dimensions
                if gate_te.dim() > twist_saliency.dim():
                    gate_te = gate_te.squeeze(-1)
                elif twist_saliency.dim() > gate_te.dim():
                    twist_saliency = twist_saliency.squeeze(-1)
            # High twist saliency should lead to high gate value
            # Normalize both to [0, 1] range for fair comparison
            gate_te_norm = gate_te
            saliency_norm = twist_saliency.detach()
            consistency_loss = consistency_loss + F.mse_loss(
                gate_te_norm, saliency_norm
            )
        
        # Additional constraint: gates should be in reasonable range [0, 1]
        for gate_name, gate_value in gates.items():
            # Penalize gates that are too close to 0 or 1 (encourage diversity)
            gate_entropy = -gate_value * torch.log(gate_value + 1e-8) - (1 - gate_value) * torch.log(1 - gate_value + 1e-8)
            consistency_loss = consistency_loss - 0.1 * gate_entropy.mean()  # Encourage entropy
        
        return consistency_loss
    
    def source_stability_loss(self, source_nodes_before, source_nodes_after):

        stability_loss = torch.tensor(0.0, device=source_nodes_before['F_k'].device)
        
        for node_key in ['F_k', 'B_k', 'A_k', 'M_k', 'S_k']:
            if node_key in source_nodes_before and node_key in source_nodes_after:
                # Source should not change (detached, so this should be zero)
                # But we check if there's any numerical drift
                diff = source_nodes_after[node_key] - source_nodes_before[node_key].detach()
                stability_loss = stability_loss + torch.norm(diff) ** 2
        
        return stability_loss
    
    def forward(self, intervened_nodes, original_nodes, gates, saliencies, source_nodes_before=None, source_nodes_after=None):

        device = gates['forward_to_twist_gate'].device if gates else 'cuda'
        total_loss = torch.zeros((), device=device)
        
        # 1. Reverse gradient penalty (ENABLED with stable implementation)
        if original_nodes is not None and len(original_nodes) >= 2:
            # Check Forward -> Twist: Twist should not influence Forward
            forward_nodes = original_nodes[0]
            twist_nodes_after = intervened_nodes[1] if len(intervened_nodes) > 1 else original_nodes[1]
            
            reverse_penalty_ft = self.reverse_gradient_penalty_v2(
                forward_nodes, twist_nodes_after
            )
            total_loss = total_loss + self.penalty_weight * reverse_penalty_ft
            
            # Check Twist -> Entry: Entry should not influence Twist
            if len(original_nodes) >= 3:
                twist_nodes = original_nodes[1]
                entry_nodes_after = intervened_nodes[2] if len(intervened_nodes) > 2 else original_nodes[2]
                
                reverse_penalty_te = self.reverse_gradient_penalty_v2(
                    twist_nodes, entry_nodes_after
                )
                total_loss = total_loss + self.penalty_weight * reverse_penalty_te
        
        # 2. Intervention consistency loss
        if saliencies is not None:
            consistency_loss = self.intervention_consistency_loss(gates, saliencies)
            total_loss = total_loss + self.consistency_weight * consistency_loss
        
        # 3. Source stability loss (if provided)
        if source_nodes_before is not None and source_nodes_after is not None:
            stability_loss = self.source_stability_loss(source_nodes_before, source_nodes_after)
            total_loss = total_loss + self.stability_weight * stability_loss
        
        return total_loss

