
import torch


def create_stage_labels_from_boundaries(boundaries, temporal_dim, device='cuda'):

    B = boundaries.shape[0]
    T = temporal_dim
    
    forward_label = torch.zeros(B, T, 1, device=device)
    twist_label = torch.zeros(B, T, 1, device=device)
    entry_label = torch.zeros(B, T, 1, device=device)
    
    for b in range(B):
        fwd_end = int(boundaries[b, 0].item())
        twist_end = int(boundaries[b, 1].item())
        
        # Clamp to valid range
        fwd_end = max(0, min(fwd_end, T-1))
        twist_end = max(fwd_end, min(twist_end, T-1))
        
        # Create hard labels
        forward_label[b, :fwd_end, 0] = 1.0
        twist_label[b, fwd_end:twist_end, 0] = 1.0
        entry_label[b, twist_end:, 0] = 1.0
    
    return forward_label, twist_label, entry_label
