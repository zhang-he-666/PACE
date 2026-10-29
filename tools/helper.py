import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import numpy as np
from utils.misc import segment_iou, cal_tiou, seg_pool_1d, seg_pool_3d
from einops import rearrange
import segmentation_models_pytorch as smp
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

settigns_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_state_dict(model):

    if isinstance(model, DDP):
        return model.module.state_dict()
    return model.state_dict()


def get_model(model):

    if isinstance(model, DDP):
        return model.module
    return model


def scale_stage_boundaries(boundaries, video_length, tas_resolution):

    # Ensure boundaries is 2D [B, 2]
    if boundaries.dim() == 1:
        boundaries = boundaries.unsqueeze(0)
    elif boundaries.dim() > 2:
        raise ValueError(f"boundaries must be 1D or 2D, got {boundaries.dim()}D with shape {boundaries.shape}")
    
    if tas_resolution is None or tas_resolution == 0:
        result = boundaries.long().clamp(0, video_length - 1)
    else:
        scale = float(video_length) / float(tas_resolution)
        scaled = (boundaries.float() * scale).round().long()
        result = torch.clamp(scaled, 0, video_length - 1)
    
    # Ensure result is [B, 2]
    if result.dim() == 1:
        result = result.unsqueeze(0)
    if result.shape[1] != 2:
        raise ValueError(f"scale_stage_boundaries returned wrong shape: {result.shape}, expected [B, 2]")
    
    return result


def create_stage_padding_mask(stage_features, stage_boundaries=None):

    B, S, _ = stage_features.shape
    device = stage_features.device
    
    # For now, assume all stages are valid (no padding)
    # In future, can be extended to handle variable number of stages per sample
    padding_mask = torch.zeros(B, S, dtype=torch.bool, device=device)
    
    return padding_mask


def create_temporal_padding_mask_from_stage_probs(forward_prob, twist_prob, entry_prob, threshold=0.1):

    # Sum all stage probabilities
    total_prob = forward_prob + twist_prob + entry_prob  # [B, T, 1]
    
    # Mark as padding if total probability is too low
    padding_mask = (total_prob.squeeze(-1) < threshold)  # [B, T]
    
    return padding_mask


def extract_stage_features(base_model, stage_clips, stage_names):

    backbone = get_model(base_model).rgb_backbone
    stage_features = []
    for name in stage_names:
        clip = stage_clips[name]
        _, feature = backbone(clip)

        if feature.dim() > 2:
            feature = feature.flatten(2).mean(dim=-1)
        
        # Ensure feature is 2D [B, feature_dim]
        if feature.dim() == 1:
            feature = feature.unsqueeze(0)
        
        stage_features.append(feature)
    # Stack along the stage dimension -> [B, num_stages, in_dim]
    return torch.stack(stage_features, dim=1)


def create_stage_based_mask(feature, stage_boundaries, video_length, tas_resolution):

    B, T, C = feature.shape
    device = feature.device
    
    # Scale boundaries to feature temporal dimension
    if tas_resolution is None or tas_resolution == 0:
        scale = 1.0
    else:
        scale = float(T) / float(tas_resolution)
    

    num_stages = stage_boundaries.shape[1] + 1
    stage_masks = torch.zeros(B, T, num_stages, device=device)
    
    for b in range(B):

        boundaries = stage_boundaries[b].float() * scale
        boundaries = boundaries.round().long().clamp(0, T - 1)
        

        starts = [0] + boundaries.tolist()
        ends = boundaries.tolist() + [T - 1]
        

        for stage_idx in range(num_stages):
            start = starts[stage_idx]
            end = ends[stage_idx]
            if end < start:
                end = start
            stage_masks[b, start:end+1, stage_idx] = 1.0
    

    stage_masks_expanded = stage_masks.unsqueeze(-1).expand(B, T, num_stages, C)

    stage_mask = stage_masks_expanded.mean(dim=2)  # [B, T, C]
    
    return stage_mask


def compute_stage_reconstruction_loss(
    recon_1, recon_2, feat_1, feat_2, args, epoch=None
):

    target_1 = feat_1.detach()
    target_2 = feat_2.detach()
    stage_recon_loss_local = 0.5 * (
        F.smooth_l1_loss(recon_1, target_1) + F.smooth_l1_loss(recon_2, target_2)
    )

    stage_loss_weight = getattr(args, "stage_recon_loss_weight", 1.0)
    stage_priority_epochs = getattr(args, "stage_priority_epochs", 0)
    stage_priority_boost = getattr(args, "stage_priority_boost", 1.0)
    if epoch is not None and epoch < stage_priority_epochs:
        stage_loss_weight *= stage_priority_boost
    return stage_recon_loss_local * stage_loss_weight


def compute_global_causal_loss(global_1, global_2, graph_1, graph_2, args):

    global_consistency_loss = 0.5 * (
        F.mse_loss(global_1, graph_1.detach()) + F.mse_loss(global_2, graph_2.detach())
    )
    global_loss_weight = getattr(args, "global_causal_loss_weight", 1.0)
    return global_consistency_loss * global_loss_weight

end = time.time()


def network_forward_train(
    base_model,
    psnet_model,
    decoder,
    regressor_delta,
    dim_reducer1,
    dim_reducer2,
    pred_scores,
    video_1,
    label_1_score,
    video_2,
    label_2_score,
    video_1_mask,
    video_2_mask,
    mse,
    optimizer,
    opti_flag,
    epoch,
    idx,
    total_batches,
    args,
    label_1_tas,
    label_2_tas,
    bce,
    pred_tious_5,
    pred_tious_75,
    dim_reducer3,
    video_encoder,
    segmenter,
    segment_metrics,
    rank,
    len_dataloader,
    focal_loss,
    scheduler,
    n_iter_per_epoch,
    # causal_intervention,  # REMOVED: Redundant with StageCausalReasoner
    difficulty,
    use_difficulty,
    temporal_causal_attn,
    awl,
    stage_segmenter,
    stage_causal_reasoner,
    boundary_regression_head=None,
    boundary_regression_loss=None,
    label_1_boundaries=None,
    label_2_boundaries=None,
):
    """Forward pass for training with backpropagation"""
    # CRITICAL: Verify temporal_causal_attn is callable at the start of function
    if not callable(temporal_causal_attn):
        raise ValueError(
            f"CRITICAL BUG: temporal_causal_attn is not callable in network_forward_train! "
            f"temporal_causal_attn type: {type(temporal_causal_attn)}, value: {temporal_causal_attn}"
        )
    
    t_loss = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Added 6th element for causal_reg loss
    
    ############# Segmentation #############
    use_mask_module = getattr(args, 'use_mask_module', True)  # Default: True for backward compatibility
    mask_loss_weight = getattr(args, 'mask_loss_weight', 1.0)  # Default: 1.0 for backward compatibility
    
    if use_mask_module:
        total_video = torch.cat((video_1, video_2), 0)
        start_idx = list(range(0, 90, 10))
        video_pack = torch.cat([total_video[:, :, i : i + 16] for i in start_idx])
        mask_feamap, mask_feature, mask_pred = segmenter(video_pack)

        Nt, C, T, H, W = mask_feamap.size()
        mask_feature = mask_feature.reshape(
            len(start_idx), len(total_video), -1
        ).transpose(0, 1)

        total_mask = torch.cat((video_1_mask, video_2_mask))
        mask_target = torch.cat([total_mask[:, :, i : i + 16] for i in start_idx])
        mask_pred_ = [rearrange(pred, "b c t h w -> (b t) c h w") for pred in mask_pred]
        mask_target = rearrange(mask_target, "b c t h w -> (b t) c h w")
        mask_target = mask_target.round().long()

        loss_mask = None
        if mask_loss_weight > 0:  # Only compute loss if weight > 0
            for i in range(5):
                focal_loss_val = focal_loss(mask_pred_[i], mask_target)
                if loss_mask is None:
                    loss_mask = focal_loss_val
                else:
                    loss_mask = loss_mask + focal_loss_val
            loss_mask = loss_mask * mask_loss_weight  # Apply weight
        else:
            loss_mask = torch.tensor(0.0, device=video_1.device, requires_grad=False)

        mask_pred = rearrange(mask_pred[-1], "b c t h w -> (b t) c h w")
        tp, fp, fn, tn = smp.metrics.get_stats(
            mask_pred, mask_target, mode="binary", threshold=0.5
        )
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")

        segment_metrics["iou_scores"].append(iou_score.item())
        segment_metrics["f1_scores"].append(f1_score.item())
        segment_metrics["f2_scores"].append(f2_score.item())
        segment_metrics["accuracy"].append(accuracy.item())
        segment_metrics["recall"].append(recall.item())
    else:
        # Mask module disabled: create dummy mask_feature (all ones) and zero loss
        B = video_1.shape[0] + video_2.shape[0]
        mask_feature = torch.ones(B, 9, 1024, device=video_1.device)  # Dummy mask: all ones (no masking)
        loss_mask = torch.tensor(0.0, device=video_1.device, requires_grad=False)
        
        # Add dummy metrics
        segment_metrics["iou_scores"].append(0.0)
        segment_metrics["f1_scores"].append(0.0)
        segment_metrics["f2_scores"].append(0.0)
        segment_metrics["accuracy"].append(0.0)
        segment_metrics["recall"].append(0.0)

    ############# Video feature (RGB only) #############
    com_feature_12, com_feamap_12 = base_model(video_1, video_2)
    orig_video_1_fea = com_feature_12[:, :, : com_feature_12.shape[2] // 2]
    orig_video_2_fea = com_feature_12[:, :, com_feature_12.shape[2] // 2 :]
    com_feature_12_u = torch.cat((orig_video_1_fea, orig_video_2_fea), 0)

    ############# Mask and I3D Feature Fusion #############
    use_mask_module = getattr(args, 'use_mask_module', True)  # Default: True for backward compatibility
    
    if use_mask_module:
        # Enhanced feature fusion with adaptive weighting
        # Use learnable combination instead of simple multiplication
        mask_weight = torch.sigmoid(mask_feature)  # [4, 9, 1024]
        
        # Enhanced fusion: combine original features with masked features adaptively
        # This allows the model to learn the optimal balance between raw and masked features
        fusion_alpha = getattr(args, 'feature_fusion_alpha', 0.7)  # Default: 70% masked, 30% raw
        u_fea_96 = fusion_alpha * (com_feature_12_u * mask_weight) + (1 - fusion_alpha) * com_feature_12_u
    else:
        # Mask module disabled: use raw features directly without mask weighting
        u_fea_96 = com_feature_12_u
    
    # REMOVED: FeatureGATCausal (causal_intervention) - redundant with StageCausalReasoner
    # The StageCausalReasoner and GatedInterStageIntervention already provide
    # comprehensive causal modeling, making this module redundant.
    # Directly use the fused features without additional GAT intervention.
    
    # Get individual video features
    video_1_fea = u_fea_96[: u_fea_96.shape[0] // 2]
    video_2_fea = u_fea_96[u_fea_96.shape[0] // 2 :]

    ############# Predict transit #############
    y1 = psnet_model(video_1_fea)
    y2 = psnet_model(video_2_fea)
    transits_pred = torch.cat((y1, y2))
    label_12_tas = torch.cat((label_1_tas, label_2_tas), 0)

    label_12_pad = torch.zeros(transits_pred.size())
    for bs in range(transits_pred.shape[0]):
        label_12_pad[bs, int(label_12_tas[bs, 0]), 0] = 1
        label_12_pad[bs, int(label_12_tas[bs, -1]), -1] = 1
    loss_tas = bce(transits_pred, label_12_pad.cuda())

    num = round(transits_pred.shape[1] / transits_pred.shape[-1])
    transits_st_ed = torch.zeros(label_12_tas.size())
    for bs in range(transits_pred.shape[0]):
        for i in range(transits_pred.shape[-1]):
            transits_st_ed[bs, i] = (
                transits_pred[bs, i * num : (i + 1) * num, i].argmax(0).cpu().item()
                + i * num
            )
    # Split predictions for video_1 and video_2
    # transits_st_ed shape: [B*2, 2] (concatenated video_1 and video_2)
    batch_size_total = transits_st_ed.shape[0]
    batch_size_per_video = batch_size_total // 2
    
    label_1_tas_pred = transits_st_ed[:batch_size_per_video]  # [B, 2]
    label_2_tas_pred = transits_st_ed[batch_size_per_video:]  # [B, 2]
    tas_resolution = transits_pred.shape[1]

    # Ensure boundaries have correct shape: [B, 2]
    # StageSegmenter expects [B, num_stages-1] = [B, 2] for 3 stages
    if label_1_tas_pred.dim() == 1:
        label_1_tas_pred = label_1_tas_pred.unsqueeze(0)
    if label_2_tas_pred.dim() == 1:
        label_2_tas_pred = label_2_tas_pred.unsqueeze(0)
    
    # Ensure boundaries have exactly 2 columns (Forward/Twist boundary, Twist/Entry boundary)
    if label_1_tas_pred.shape[1] != 2:
        # If shape is wrong, try to fix it
        if label_1_tas_pred.numel() == 2:
            label_1_tas_pred = label_1_tas_pred.view(-1, 2)
        else:
            raise ValueError(f"label_1_tas_pred has wrong shape: {label_1_tas_pred.shape}, expected [B, 2]. "
                           f"transits_st_ed shape: {transits_st_ed.shape}, batch_size_total: {batch_size_total}")
    if label_2_tas_pred.shape[1] != 2:
        if label_2_tas_pred.numel() == 2:
            label_2_tas_pred = label_2_tas_pred.view(-1, 2)
        else:
            raise ValueError(f"label_2_tas_pred has wrong shape: {label_2_tas_pred.shape}, expected [B, 2]. "
                           f"transits_st_ed shape: {transits_st_ed.shape}, batch_size_total: {batch_size_total}")
    
    # Ensure batch sizes match between video and boundaries
    video_1_batch_size = video_1.shape[0]
    video_2_batch_size = video_2.shape[0]
    if label_1_tas_pred.shape[0] != video_1_batch_size:
        raise ValueError(
            f"Batch size mismatch for video_1: video has {video_1_batch_size} batches, "
            f"but boundaries has {label_1_tas_pred.shape[0]} batches. "
            f"video_1 shape: {video_1.shape}, label_1_tas_pred shape: {label_1_tas_pred.shape}"
        )
    if label_2_tas_pred.shape[0] != video_2_batch_size:
        raise ValueError(
            f"Batch size mismatch for video_2: video has {video_2_batch_size} batches, "
            f"but boundaries has {label_2_tas_pred.shape[0]} batches. "
            f"video_2 shape: {video_2.shape}, label_2_tas_pred shape: {label_2_tas_pred.shape}"
        )

    video_1_stage_bounds = scale_stage_boundaries(
        label_1_tas_pred, video_1.shape[2], tas_resolution
    )
    video_2_stage_bounds = scale_stage_boundaries(
        label_2_tas_pred, video_2.shape[2], tas_resolution
    )
    
    # Ensure boundaries have correct shape after scaling
    # scale_stage_boundaries should return [B, 2] shape
    if video_1_stage_bounds.dim() != 2 or video_1_stage_bounds.shape[1] != 2:
        raise ValueError(
            f"video_1_stage_bounds has wrong shape: {video_1_stage_bounds.shape}, expected [B, 2]. "
            f"label_1_tas_pred shape: {label_1_tas_pred.shape}"
        )
    if video_2_stage_bounds.dim() != 2 or video_2_stage_bounds.shape[1] != 2:
        raise ValueError(
            f"video_2_stage_bounds has wrong shape: {video_2_stage_bounds.shape}, expected [B, 2]. "
            f"label_2_tas_pred shape: {label_2_tas_pred.shape}"
        )
    
    # Ensure batch sizes match
    if video_1_stage_bounds.shape[0] != video_1.shape[0]:
        raise ValueError(
            f"Batch size mismatch: video_1 has {video_1.shape[0]} batches, "
            f"but video_1_stage_bounds has {video_1_stage_bounds.shape[0]} batches. "
            f"video_1 shape: {video_1.shape}, video_1_stage_bounds shape: {video_1_stage_bounds.shape}"
        )
    if video_2_stage_bounds.shape[0] != video_2.shape[0]:
        raise ValueError(
            f"Batch size mismatch: video_2 has {video_2.shape[0]} batches, "
            f"but video_2_stage_bounds has {video_2_stage_bounds.shape[0]} batches. "
            f"video_2 shape: {video_2.shape}, video_2_stage_bounds shape: {video_2_stage_bounds.shape}"
        )
    
    # Ensure video has correct shape: [B, C, T, H, W]
    # StageSegmenter expects 5D tensor
    if video_1.dim() != 5:
        raise ValueError(f"video_1 has wrong number of dimensions: {video_1.dim()}, expected 5 [B, C, T, H, W]")
    if video_2.dim() != 5:
        raise ValueError(f"video_2 has wrong number of dimensions: {video_2.dim()}, expected 5 [B, C, T, H, W]")

    # CRITICAL FIX: Ensure stage_segmenter is actually a StageSegmenter, not AutomaticWeightedLoss
    # This is a critical check to catch variable swap bugs
    from models.stage_modules import StageSegmenter
    from models.AutomaticWeightedLoss import AutomaticWeightedLoss
    
    # Check if stage_segmenter is directly an AutomaticWeightedLoss (should never happen)
    if isinstance(stage_segmenter, AutomaticWeightedLoss):
        # FIX: Try to recover by creating a new StageSegmenter
        # This should not happen, but we try to recover gracefully
        print(f"[WARNING] stage_segmenter is directly assigned to AutomaticWeightedLoss! "
              f"Attempting to recover by creating a new StageSegmenter.")
        stage_segmenter = StageSegmenter(
            stage_names=("Forward", "Twist", "Entry"),
            stage_clip_length=getattr(args, "stage_clip_length", 32),
        ).cuda()
    
    # Check if stage_segmenter is wrapped in DDP but contains AutomaticWeightedLoss (variable swap)
    if hasattr(stage_segmenter, 'module'):
        if isinstance(stage_segmenter.module, AutomaticWeightedLoss):
            # FIX: If stage_segmenter was incorrectly wrapped with awl's DDP wrapper,
            # we need to get the actual stage_segmenter from somewhere else
            # Since StageSegmenter has no trainable parameters, it should NOT be DDP wrapped
            # This is a critical bug that needs to be fixed at the source
            # FIX: Recover by creating a new StageSegmenter
            print(f"[WARNING] stage_segmenter.module is AutomaticWeightedLoss! "
                  f"This indicates stage_segmenter was incorrectly wrapped with awl's DDP wrapper. "
                  f"Attempting to recover by creating a new StageSegmenter.")
            stage_segmenter = StageSegmenter(
                stage_names=("Forward", "Twist", "Entry"),
                stage_clip_length=getattr(args, "stage_clip_length", 32),
            ).cuda()
        elif not isinstance(stage_segmenter.module, StageSegmenter):
            # If DDP wrapped but contains something else (not StageSegmenter, not AutomaticWeightedLoss)
            raise ValueError(
                f"stage_segmenter has wrong type: {type(stage_segmenter)} "
                f"(module: {type(stage_segmenter.module)}), expected StageSegmenter"
            )
        else:
            # If DDP wrapped and contains StageSegmenter, unwrap it
            # StageSegmenter should NOT be DDP wrapped (it has no trainable parameters)
            stage_segmenter = stage_segmenter.module
    elif not isinstance(stage_segmenter, StageSegmenter):
        # If not DDP wrapped and not StageSegmenter, try to recover
        if isinstance(stage_segmenter, AutomaticWeightedLoss):
            print(f"[WARNING] stage_segmenter is AutomaticWeightedLoss (not DDP wrapped)! "
                  f"Attempting to recover by creating a new StageSegmenter.")
            stage_segmenter = StageSegmenter(
                stage_names=("Forward", "Twist", "Entry"),
                stage_clip_length=getattr(args, "stage_clip_length", 32),
            ).cuda()
        else:
            raise ValueError(
                f"stage_segmenter has wrong type: {type(stage_segmenter)}, expected StageSegmenter"
            )

    video_1_stage_clips = stage_segmenter(video_1, video_1_stage_bounds)
    video_2_stage_clips = stage_segmenter(video_2, video_2_stage_bounds)

    video_1_stage_features = extract_stage_features(
        base_model, video_1_stage_clips, stage_segmenter.stage_names
    )
    video_2_stage_features = extract_stage_features(
        base_model, video_2_stage_clips, stage_segmenter.stage_names
    )
    
    # Note: stage_features are already stage-separated features
    # Each stage feature is extracted from its corresponding stage clip
    # This naturally creates "pure" stage features as described in the framework

    # CRITICAL FIX: Verify stage_causal_reasoner is not accidentally swapped with stage_segmenter
    from models.stage_modules import StageCausalReasoner
    if hasattr(stage_causal_reasoner, 'module'):
        # If DDP wrapped, check the module type
        if isinstance(stage_causal_reasoner.module, StageSegmenter):
            # FIX: Try to recover by checking if stage_segmenter is actually StageCausalReasoner
            if isinstance(stage_segmenter, StageCausalReasoner):
                print(f"[WARNING] stage_causal_reasoner.module is StageSegmenter, but stage_segmenter is StageCausalReasoner! "
                      f"Swapping them back.")
                stage_causal_reasoner = stage_segmenter
                # Also need to fix stage_segmenter
                from models.stage_modules import StageSegmenter
                stage_segmenter = StageSegmenter(
                    stage_names=("Forward", "Twist", "Entry"),
                    stage_clip_length=getattr(args, "stage_clip_length", 32),
                ).cuda()
            else:
                raise ValueError(
                    f"CRITICAL BUG: stage_causal_reasoner.module is StageSegmenter! "
                    f"This indicates stage_causal_reasoner was incorrectly swapped with stage_segmenter. "
                    f"stage_causal_reasoner type: {type(stage_causal_reasoner)}, "
                    f"stage_causal_reasoner.module type: {type(stage_causal_reasoner.module)}, "
                    f"stage_segmenter type: {type(stage_segmenter)}"
                )
    elif isinstance(stage_causal_reasoner, StageSegmenter):
        # FIX: Try to recover by checking if stage_segmenter is actually StageCausalReasoner
        if isinstance(stage_segmenter, StageCausalReasoner):
            print(f"[WARNING] stage_causal_reasoner is StageSegmenter, but stage_segmenter is StageCausalReasoner! "
                  f"Swapping them back.")
            stage_causal_reasoner = stage_segmenter
            # Also need to fix stage_segmenter
            from models.stage_modules import StageSegmenter
            stage_segmenter = StageSegmenter(
                stage_names=("Forward", "Twist", "Entry"),
                stage_clip_length=getattr(args, "stage_clip_length", 32),
            ).cuda()
        else:
            # CRITICAL FIX: If both are StageSegmenter, stage_causal_reasoner was lost
            # Recreate it from args
            print(f"[WARNING] stage_causal_reasoner is StageSegmenter but stage_segmenter is also StageSegmenter! "
                  f"This indicates stage_causal_reasoner was lost. Recreating StageCausalReasoner from args.")
            from models.stage_modules import StageCausalReasoner
            use_5node_subgraph = getattr(args, "use_5node_subgraph", True)
            use_inter_stage_intervention = getattr(args, "use_inter_stage_intervention", True)
            stage_feature_dim = getattr(args, "stage_feature_dim", 1024)
            stage_causal_reasoner = StageCausalReasoner(
                in_dim=stage_feature_dim,
                hidden_dim=getattr(args, "stage_reasoner_hidden_dim", 512),
                projection_dim=getattr(args, "stage_graph_dim", 64),
                num_heads=getattr(args, "stage_reasoner_heads", 4),
                num_stages=3,
                use_5node_subgraph=use_5node_subgraph,
                use_inter_stage_intervention=use_inter_stage_intervention,
                use_foreground=getattr(args, "use_foreground_node", True),
                use_body=getattr(args, "use_body_node", True),
                use_appearance=getattr(args, "use_appearance_node", True),
                use_motion=getattr(args, "use_motion_node", True),
                use_saliency=getattr(args, "use_saliency_node", True),
                use_directed_graph=getattr(args, "use_directed_graph", True),
            ).cuda()
            print(f"[WARNING] Recreated StageCausalReasoner. This may cause issues if it had trained weights.")

    (
        video_1_stage_graph,
        _,
        video_1_stage_scores,
        video_1_stage_recon,
        video_1_extra_outputs,
    ) = stage_causal_reasoner(video_1_stage_features)
    (
        video_2_stage_graph,
        _,
        video_2_stage_scores,
        video_2_stage_recon,
        video_2_extra_outputs,
    ) = stage_causal_reasoner(video_2_stage_features)

    padding_mask_1 = create_stage_padding_mask(video_1_stage_graph)
    padding_mask_2 = create_stage_padding_mask(video_2_stage_graph)
    
    video_1_stage_global, _ = temporal_causal_attn(video_1_stage_graph, padding_mask=padding_mask_1)
    video_2_stage_global, _ = temporal_causal_attn(video_2_stage_graph, padding_mask=padding_mask_2)

    ############# Stage-level auxiliary losses #############
    loss_stage = compute_stage_reconstruction_loss(
        video_1_stage_recon,
        video_2_stage_recon,
        video_1_stage_features,
        video_2_stage_features,
        args,
        epoch=epoch,
    )
    
    loss_stage_classification = None
    if hasattr(args, 'use_stage_classification_loss') and args.use_stage_classification_loss:
        from models.stage_classification_loss import StageClassificationLoss
        
        # Initialize loss module if not already done
        if not hasattr(network_forward_train, '_stage_classification_loss_module'):
            network_forward_train._stage_classification_loss_module = StageClassificationLoss(
                use_focal=getattr(args, 'stage_classification_use_focal', True),
                alpha=getattr(args, 'stage_classification_alpha', 0.25),
                gamma=getattr(args, 'stage_classification_gamma', 2.0),
                stage_weights=getattr(args, 'stage_classification_weights', None)
            )
        
        stage_classification_loss_module = network_forward_train._stage_classification_loss_module
        
        pred_saliencies_1 = video_1_stage_scores
        pred_saliencies_2 = video_2_stage_scores
        
        if label_1_tas is not None and label_2_tas is not None:
            from models.stage_segmenter import create_stage_labels_from_boundaries
            
            B = pred_saliencies_1.shape[0]
            device = pred_saliencies_1.device
            temporal_dim = tas_resolution if tas_resolution is not None else video_1.shape[2]
            
            forward_label_1, twist_label_1, entry_label_1 = create_stage_labels_from_boundaries(
                label_1_tas_pred, temporal_dim, device
            )
            forward_label_2, twist_label_2, entry_label_2 = create_stage_labels_from_boundaries(
                label_2_tas_pred, temporal_dim, device
            )
            
            target_saliencies_1 = torch.stack([
                forward_label_1.mean(dim=1),
                twist_label_1.mean(dim=1),
                entry_label_1.mean(dim=1)
            ], dim=1)
            
            target_saliencies_2 = torch.stack([
                forward_label_2.mean(dim=1),
                twist_label_2.mean(dim=1),
                entry_label_2.mean(dim=1)
            ], dim=1)
            
            loss_stage_class_1, loss_dict_1 = stage_classification_loss_module(
                pred_saliencies_1, target_saliencies_1
            )
            loss_stage_class_2, loss_dict_2 = stage_classification_loss_module(
                pred_saliencies_2, target_saliencies_2
            )
            
            loss_stage_classification = loss_stage_class_1 + loss_stage_class_2
            stage_class_weight = getattr(args, 'stage_classification_loss_weight', 0.5)
            loss_stage_classification = stage_class_weight * loss_stage_classification

    loss_global = None

    ############# Static feature #############
    v11, v12, v13 = video_encoder(video_1, label_1_tas_pred)
    v21, v22, v23 = video_encoder(video_2, label_2_tas_pred)

    ############# Interpolate #############
    N, T, C = video_1_fea.size()
    video_1_fea = video_1_fea.transpose(1, 2)
    video_1_fea_re_list = []
    for bs in range(N):
        v1i0 = (
            int(label_1_tas_pred[bs][0].item() + label_2_tas_pred[bs][0].item())
            // 20
        )
        v1i1 = (
            int(label_1_tas_pred[bs][1].item() + label_2_tas_pred[bs][1].item())
            // 20
        )
        video_1_fea_re_list.append(
            seg_pool_1d(video_1_fea[bs].unsqueeze(0), v1i0, v1i1, 4)
        )
    video_1_fea_re = torch.cat(video_1_fea_re_list, 0).transpose(1, 2)

    video_2_fea = video_2_fea.transpose(1, 2)
    video_2_fea_re_list = []
    for bs in range(N):
        v1i0 = (
            int(label_1_tas_pred[bs][0].item() + label_2_tas_pred[bs][0].item())
            // 20
        )
        v1i1 = (
            int(label_1_tas_pred[bs][1].item() + label_2_tas_pred[bs][1].item())
            // 20
        )
        video_2_fea_re_list.append(
            seg_pool_1d(video_2_fea[bs].unsqueeze(0), v1i0, v1i1, 4)
        )
    video_2_fea_re = torch.cat(video_2_fea_re_list, 0).transpose(1, 2)

    ############# Lower dimension #############
    video_1_segs_1 = dim_reducer3(video_1_fea_re[:, 0:4, :])
    video_1_segs_2 = dim_reducer1(video_1_fea_re[:, 4:8, :])
    video_1_segs_3 = dim_reducer2(video_1_fea_re[:, 8:12, :])
    video_2_segs_1 = dim_reducer3(video_2_fea_re[:, 0:4, :])
    video_2_segs_2 = dim_reducer1(video_2_fea_re[:, 4:8, :])
    video_2_segs_3 = dim_reducer2(video_2_fea_re[:, 8:12, :])

    video_1_segs_1 = (
        video_1_segs_1
        + video_1_stage_graph[:, 0].unsqueeze(1)
        + video_1_stage_global[:, 0].unsqueeze(1)
    )
    video_1_segs_2 = (
        video_1_segs_2
        + video_1_stage_graph[:, 1].unsqueeze(1)
        + video_1_stage_global[:, 1].unsqueeze(1)
    )
    video_1_segs_3 = (
        video_1_segs_3
        + video_1_stage_graph[:, 2].unsqueeze(1)
        + video_1_stage_global[:, 2].unsqueeze(1)
    )

    video_2_segs_1 = (
        video_2_segs_1
        + video_2_stage_graph[:, 0].unsqueeze(1)
        + video_2_stage_global[:, 0].unsqueeze(1)
    )
    video_2_segs_2 = (
        video_2_segs_2
        + video_2_stage_graph[:, 1].unsqueeze(1)
        + video_2_stage_global[:, 1].unsqueeze(1)
    )
    video_2_segs_3 = (
        video_2_segs_3
        + video_2_stage_graph[:, 2].unsqueeze(1)
        + video_2_stage_global[:, 2].unsqueeze(1)
    )
    
    loss_boundary = None
    if boundary_regression_head is not None and boundary_regression_loss is not None:
        video_1_fused_features = video_1_stage_graph.view(video_1_stage_graph.shape[0], -1)
        video_2_fused_features = video_2_stage_graph.view(video_2_stage_graph.shape[0], -1)
        
        pred_boundaries_1, pred_completeness_1, pred_category_1 = boundary_regression_head(
            video_1_fused_features, temporal_length=video_1.shape[2]
        )
        pred_boundaries_2, pred_completeness_2, pred_category_2 = boundary_regression_head(
            video_2_fused_features, temporal_length=video_2.shape[2]
        )
        
        if label_1_boundaries is not None and label_2_boundaries is not None:
            temporal_length_1 = video_1.shape[2]
            temporal_length_2 = video_2.shape[2]
            
            target_boundaries_1 = label_1_boundaries.float() / temporal_length_1
            target_boundaries_2 = label_2_boundaries.float() / temporal_length_2
            
            target_completeness_1 = label_1_score if label_1_score is not None else torch.ones_like(pred_completeness_1)
            target_completeness_2 = label_2_score if label_2_score is not None else torch.ones_like(pred_completeness_2)
            
            loss_boundary_1, loss_dict_1 = boundary_regression_loss(
                pred_boundaries_1, pred_completeness_1, pred_category_1,
                target_boundaries_1, target_completeness_1, None
            )
            
            loss_boundary_2, loss_dict_2 = boundary_regression_loss(
                pred_boundaries_2, pred_completeness_2, pred_category_2,
                target_boundaries_2, target_completeness_2, None
            )
            
            loss_boundary = loss_boundary_1 + loss_boundary_2

    ############# Fusion dynamic and static feature #############
    v_12_list = []
    v_21_list = []
    v_12 = decoder[0](v11, v21)
    v_21 = decoder[0](v21, v11)
    v_12_list.append(v_12)
    v_21_list.append(v_21)

    v_12 = decoder[0](v12, v22)
    v_21 = decoder[0](v22, v12)
    v_12_list.append(v_12)
    v_21_list.append(v_21)

    v_12 = decoder[2](v13, v23)
    v_21 = decoder[2](v23, v13)
    v_12_list.append(v_12)
    v_21_list.append(v_21)

    v_12_map = torch.cat(v_12_list, 1)
    v_21_map = torch.cat(v_21_list, 1)
    v_12_21 = torch.cat((v_12_map, v_21_map), 0)

    ############# Cross attention #############
    decoder_video_12_map_list = []
    decoder_video_21_map_list = []

    decoder_video_12_map = decoder[1](video_1_segs_1, video_2_segs_1)
    decoder_video_21_map = decoder[1](video_2_segs_1, video_1_segs_1)
    decoder_video_12_map_list.append(decoder_video_12_map)
    decoder_video_21_map_list.append(decoder_video_21_map)

    decoder_video_12_map = decoder[1](video_1_segs_2, video_2_segs_2)
    decoder_video_21_map = decoder[1](video_2_segs_2, video_1_segs_2)
    decoder_video_12_map_list.append(decoder_video_12_map)
    decoder_video_21_map_list.append(decoder_video_21_map)
    
    decoder_video_12_map = decoder[3](video_1_segs_3, video_2_segs_3)
    decoder_video_21_map = decoder[3](video_2_segs_3, video_1_segs_3)
    decoder_video_12_map_list.append(decoder_video_12_map)
    decoder_video_21_map_list.append(decoder_video_21_map)

    decoder_video_12_map = torch.cat(decoder_video_12_map_list, 1)
    decoder_video_21_map = torch.cat(decoder_video_21_map_list, 1)

    ############# Fine-grained Contrastive Regression #############
    decoder_12_21 = torch.cat((decoder_video_12_map, decoder_video_21_map), 0)

    delta1 = regressor_delta[0](decoder_12_21)
    delta2 = regressor_delta[1](v_12_21)

    delta1_1 = delta1[:, :12].mean(1)
    delta1_2 = delta1[:, 12:24].mean(1)
    delta1_3 = delta1[:, 24:].mean(1)

    delta2_1 = delta2[:, :4].mean(1)
    delta2_2 = delta2[:, 4:8].mean(1)
    delta2_3 = delta2[:, 8:].mean(1)
    
    # OPTIMIZED: Use saliency-weighted fusion instead of fixed weights
    video_1_saliency = video_1_stage_scores.squeeze(-1)  # [B, 3]
    video_2_saliency = video_2_stage_scores.squeeze(-1)  # [B, 3]
    
    # Normalize saliency as weights
    video_1_weights = F.softmax(video_1_saliency, dim=1)
    video_2_weights = F.softmax(video_2_saliency, dim=1)
    
    # Weighted fusion for delta1 and delta2
    delta1 = (delta1_1 * video_1_weights[:, 0:1] + 
              delta1_2 * video_1_weights[:, 1:2] + 
              delta1_3 * video_1_weights[:, 2:3]).squeeze(-1)
    
    delta2 = (delta2_1 * video_2_weights[:, 0:1] + 
              delta2_2 * video_2_weights[:, 1:2] + 
              delta2_3 * video_2_weights[:, 2:3]).squeeze(-1)
    
    # Decoder fusion (configurable)
    decoder_fusion_alpha = getattr(args, 'decoder_fusion_alpha', 0.5)
    delta = decoder_fusion_alpha * delta1 + (1 - decoder_fusion_alpha) * delta2
    delta = delta.unsqueeze(-1)
    
    # OPTIMIZED: Stage score fusion using saliency-weighted scores
    video_1_weighted_score = (video_1_stage_scores.squeeze(-1) * video_1_weights).sum(dim=1)
    video_2_weighted_score = (video_2_stage_scores.squeeze(-1) * video_2_weights).sum(dim=1)
    stage_score_delta = video_1_weighted_score - video_2_weighted_score
    
    # Configurable stage score fusion coefficient
    stage_fusion_alpha = getattr(args, 'stage_score_fusion_alpha', 0.3)
    delta_combined = delta[: delta.shape[0] // 2] + stage_fusion_alpha * stage_score_delta.unsqueeze(-1)
    delta = torch.cat((delta_combined, delta[delta.shape[0] // 2 :]), dim=0)

    # OPTIMIZED: Enhanced loss function with optional correlation loss
    pred_diff = delta[: delta.shape[0] // 2].squeeze()
    true_diff = (label_1_score - label_2_score).squeeze()
    
    # MSE loss
    mse_loss = F.mse_loss(pred_diff, true_diff)
    
    use_correlation_loss = getattr(args, 'use_correlation_loss', False)
    if use_correlation_loss:
        pred_centered = pred_diff - pred_diff.mean()
        true_centered = true_diff - true_diff.mean()
        
        numerator = (pred_centered * true_centered).sum()
        denominator = torch.sqrt((pred_centered ** 2).sum()) * torch.sqrt((true_centered ** 2).sum()) + 1e-8
        correlation = numerator / denominator
        correlation_loss = 1 - correlation  # Convert to loss (maximize correlation)
        
        correlation_weight = getattr(args, 'correlation_loss_weight', 0.1)
        loss_aqa = mse_loss + correlation_weight * correlation_loss
    else:
        loss_aqa = mse_loss
    
    loss_causal_reg = None
    if 'intervention_gates' in video_1_extra_outputs and 'intervention_gates' in video_2_extra_outputs:
        from models.gated_inter_stage_intervention import CausalLoss
        
        if not hasattr(network_forward_train, '_causal_loss_module'):
            network_forward_train._causal_loss_module = CausalLoss(
                penalty_weight=getattr(args, 'causal_penalty_weight', 1.0),
                consistency_weight=getattr(args, 'causal_consistency_weight', 0.5),
                stability_weight=getattr(args, 'causal_stability_weight', 0.3)
            )
        
        causal_loss_module = network_forward_train._causal_loss_module
        
        gates_1 = video_1_extra_outputs['intervention_gates']
        gates_2 = video_2_extra_outputs['intervention_gates']
        saliencies_1 = video_1_extra_outputs.get('stage_saliencies', None)
        saliencies_2 = video_2_extra_outputs.get('stage_saliencies', None)
        
        all_node_dicts_1 = video_1_extra_outputs.get('all_node_dicts', None)
        all_node_dicts_2 = video_2_extra_outputs.get('all_node_dicts', None)
        original_node_dicts_1 = video_1_extra_outputs.get('original_node_dicts', None)
        original_node_dicts_2 = video_2_extra_outputs.get('original_node_dicts', None)
        
        if all_node_dicts_1 is not None and saliencies_1 is not None:
            original_nodes_1 = original_node_dicts_1 if original_node_dicts_1 is not None else all_node_dicts_1
            loss_causal_reg_1 = causal_loss_module(
                intervened_nodes=all_node_dicts_1,
                original_nodes=original_nodes_1,
                gates=gates_1,
                saliencies=saliencies_1
            )
            if loss_causal_reg is None:
                loss_causal_reg = loss_causal_reg_1
            else:
                loss_causal_reg = loss_causal_reg + loss_causal_reg_1
        
        if all_node_dicts_2 is not None and saliencies_2 is not None:
            original_nodes_2 = original_node_dicts_2 if original_node_dicts_2 is not None else all_node_dicts_2
            loss_causal_reg_2 = causal_loss_module(
                intervened_nodes=all_node_dicts_2,
                original_nodes=original_nodes_2,
                gates=gates_2,
                saliencies=saliencies_2
            )
            if loss_causal_reg is None:
                loss_causal_reg = loss_causal_reg_2
            else:
                loss_causal_reg = loss_causal_reg + loss_causal_reg_2

        if loss_causal_reg is None:
    
            forward_to_twist_gate_1 = gates_1['forward_to_twist_gate']
            twist_to_entry_gate_1 = gates_1['twist_to_entry_gate']
            forward_to_twist_gate_2 = gates_2['forward_to_twist_gate']
            twist_to_entry_gate_2 = gates_2['twist_to_entry_gate']
            
            gate_reg = 0.5 * (
                torch.mean((forward_to_twist_gate_1 - 0.5) ** 2) +
                torch.mean((twist_to_entry_gate_1 - 0.5) ** 2) +
                torch.mean((forward_to_twist_gate_2 - 0.5) ** 2) +
                torch.mean((twist_to_entry_gate_2 - 0.5) ** 2)
            )
            loss_causal_reg = gate_reg
        elif hasattr(loss_causal_reg, 'item') and loss_causal_reg.item() == 0.0:
            pass
        
        causal_reg_weight = getattr(args, 'causal_reg_weight', 0.1)
        loss_causal_reg = causal_reg_weight * loss_causal_reg

    main_loss_weight = getattr(args, "main_task_weight", 1.0)
    aux_loss_weight = getattr(args, "aux_task_weight", 1.0)
    if awl is not None:
        try:
            main_aux_loss = awl(loss_aqa, loss_tas, loss_mask)
        except Exception:
            main_aux_loss = main_loss_weight * loss_aqa + aux_loss_weight * (loss_tas + loss_mask)
    else:
        main_aux_loss = main_loss_weight * loss_aqa + aux_loss_weight * (loss_tas + loss_mask)
    total_loss = main_aux_loss
    
    if loss_stage is not None:
        total_loss = total_loss + loss_stage
    if loss_global is not None:
        total_loss = total_loss + loss_global
    if loss_causal_reg is not None:
        total_loss = total_loss + loss_causal_reg
    if loss_boundary is not None:
        total_loss = total_loss + loss_boundary
    if loss_stage_classification is not None:
        total_loss = total_loss + loss_stage_classification

    # Backward pass and optimization
    if opti_flag:
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

    t_loss[0] = loss_aqa.item()
    t_loss[1] = loss_tas.item()
    t_loss[2] = loss_mask.item() if loss_mask is not None else 0.0
    t_loss[3] = loss_stage.item() if loss_stage is not None else 0.0
    t_loss[4] = loss_global.item() if loss_global is not None else 0.0
    t_loss[5] = loss_causal_reg.item() if loss_causal_reg is not None else 0.0

    for bs in range(N):
        tIoU = segment_iou(
            np.array(label_12_tas.squeeze(-1).cpu())[bs],
            np.array(transits_st_ed.squeeze(-1).cpu())[bs],
            args,
        )
        pred_tious_5.append(1 if tIoU >= 0.5 else 0)
        pred_tious_75.append(1 if tIoU >= 0.75 else 0)

    # Update pred_scores
    score = delta[: delta.shape[0] // 2].detach() + label_2_score
    if use_difficulty:
        score = score * difficulty
    pred_scores.extend(score.cpu().numpy().flatten().tolist())

    return t_loss


def network_forward_test(
    base_model,
    psnet_model,
    decoder,
    regressor_delta,
    video_encoder,
    dim_reducer3,
    segmenter,
    dim_reducer1,
    dim_reducer2,
    pred_scores,
    video_1,
    video_2_list,
    label_2_score_list,
    video_1_mask,
    video_2_mask_list,
    args,
    label_1_tas,
    label_2_tas_list,
    pred_tious_test_5,
    pred_tious_test_75,
    segment_metrics,
    mse,
    bce,
    focal_loss,
    label_1_score,
    # causal_intervention,  # REMOVED: Redundant with StageCausalReasoner
    difficulty,
    use_difficulty,
    temporal_causal_attn,
    awl,
    stage_segmenter,
    stage_causal_reasoner,
    epoch=None,
    boundary_regression_head=None,
    boundary_regression_loss=None,
    label_1_boundaries=None,
    label_2_boundaries=None,
):
    score_accumulator = [] 
    tIoU_results = []
    t_loss = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
    for idx, (video_2, video_2_mask, label_2_score, label_2_tas) in enumerate(zip(
        video_2_list, video_2_mask_list, label_2_score_list, label_2_tas_list
    )):

        ############# Segmentation #############
        use_mask_module = getattr(args, 'use_mask_module', True) 
        mask_loss_weight = getattr(args, 'mask_loss_weight', 1.0) 
        
        if use_mask_module:
            total_video = torch.cat((video_1, video_2), 0)
            start_idx = list(range(0, 90, 10))
            video_pack = torch.cat([total_video[:, :, i : i + 16] for i in start_idx])
            mask_feamap, mask_feature, mask_pred = segmenter(video_pack)

            Nt, C, T, H, W = mask_feamap.size()
            mask_feature = mask_feature.reshape(
                len(start_idx), len(total_video), -1
            ).transpose(0, 1)

            total_mask = torch.cat((video_1_mask, video_2_mask))
            mask_target = torch.cat([total_mask[:, :, i : i + 16] for i in start_idx])
            mask_pred_ = [rearrange(pred, "b c t h w -> (b t) c h w") for pred in mask_pred]
            mask_target = rearrange(mask_target, "b c t h w -> (b t) c h w")
            mask_target = mask_target.round().long()

            loss_mask = None
            if mask_loss_weight > 0:  # Only compute loss if weight > 0
                for i in range(5):
                    focal_loss_val = focal_loss(mask_pred_[i], mask_target)
                    if loss_mask is None:
                        loss_mask = focal_loss_val
                    else:
                        loss_mask = loss_mask + focal_loss_val
                loss_mask = loss_mask * mask_loss_weight  # Apply weight
            else:
                loss_mask = torch.tensor(0.0, device=video_1.device, requires_grad=False)

            mask_pred = rearrange(mask_pred[-1], "b c t h w -> (b t) c h w")
            tp, fp, fn, tn = smp.metrics.get_stats(
                mask_pred, mask_target, mode="binary", threshold=0.5
            )
            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
            accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
            recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")

            segment_metrics["iou_scores"].append(iou_score.item())
            segment_metrics["f1_scores"].append(f1_score.item())
            segment_metrics["f2_scores"].append(f2_score.item())
            segment_metrics["accuracy"].append(accuracy.item())
            segment_metrics["recall"].append(recall.item())
        else:
            # Mask module disabled: create dummy mask_feature (all ones) and zero loss
            B = video_1.shape[0] + video_2.shape[0]
            mask_feature = torch.ones(B, 9, 1024, device=video_1.device)  # Dummy mask: all ones (no masking)
            loss_mask = torch.tensor(0.0, device=video_1.device, requires_grad=False)
            
            # Add dummy metrics
            segment_metrics["iou_scores"].append(0.0)
            segment_metrics["f1_scores"].append(0.0)
            segment_metrics["f2_scores"].append(0.0)
            segment_metrics["accuracy"].append(0.0)
            segment_metrics["recall"].append(0.0)

        ############# Video feature (RGB only) #############
        com_feature_12, com_feamap_12 = base_model(video_1, video_2)
        orig_video_1_fea = com_feature_12[:, :, : com_feature_12.shape[2] // 2]
        orig_video_2_fea = com_feature_12[:, :, com_feature_12.shape[2] // 2 :]
        com_feature_12_u = torch.cat((orig_video_1_fea, orig_video_2_fea), 0)

        ############# Mask and I3D Feature Fusion #############
        use_mask_module = getattr(args, 'use_mask_module', True)  # Default: True for backward compatibility
        
        if use_mask_module:
            mask_weight = torch.sigmoid(mask_feature)  # [4, 9, 1024]
            fusion_alpha = getattr(args, 'feature_fusion_alpha', 0.7)  # Default: 70% masked, 30% raw
            u_fea_96 = fusion_alpha * (com_feature_12_u * mask_weight) + (1 - fusion_alpha) * com_feature_12_u
        else
            u_fea_96 = com_feature_12_u
        

        video_1_fea = u_fea_96[: u_fea_96.shape[0] // 2]
        video_2_fea = u_fea_96[u_fea_96.shape[0] // 2 :]

        ############# Predict transit #############
        y1 = psnet_model(video_1_fea)
        y2 = psnet_model(video_2_fea)
        transits_pred = torch.cat((y1, y2))
        label_12_tas = torch.cat((label_1_tas, label_2_tas), 0)

        label_12_pad = torch.zeros(transits_pred.size())
        for bs in range(transits_pred.shape[0]):
            label_12_pad[bs, int(label_12_tas[bs, 0]), 0] = 1
            label_12_pad[bs, int(label_12_tas[bs, -1]), -1] = 1
        loss_tas = bce(transits_pred, label_12_pad.cuda())

    num = round(transits_pred.shape[1] / transits_pred.shape[-1])
    transits_st_ed = torch.zeros(label_12_tas.size(), device=transits_pred.device, dtype=torch.long)
    for bs in range(transits_pred.shape[0]):
        for i in range(transits_pred.shape[-1]):
            transits_st_ed[bs, i] = (
                transits_pred[bs, i * num : (i + 1) * num, i].argmax(0).cpu().item()
                + i * num
            )

    batch_size_total = transits_st_ed.shape[0]
    batch_size_per_video = batch_size_total // 2
    
    label_1_tas_pred = transits_st_ed[:batch_size_per_video] 
    label_2_tas_pred = transits_st_ed[batch_size_per_video:] 
    tas_resolution = transits_pred.shape[1]


    if label_1_tas_pred.dim() == 1:
        label_1_tas_pred = label_1_tas_pred.unsqueeze(0)
    if label_2_tas_pred.dim() == 1:
        label_2_tas_pred = label_2_tas_pred.unsqueeze(0)
    
    # Ensure boundaries have exactly 2 columns (Forward/Twist boundary, Twist/Entry boundary)
    if label_1_tas_pred.shape[1] != 2:
        # If shape is wrong, try to fix it
        if label_1_tas_pred.numel() == 2:
            label_1_tas_pred = label_1_tas_pred.view(-1, 2)
        else:
            raise ValueError(f"label_1_tas_pred has wrong shape: {label_1_tas_pred.shape}, expected [B, 2]. "
                           f"transits_st_ed shape: {transits_st_ed.shape}, batch_size_total: {batch_size_total}")
    if label_2_tas_pred.shape[1] != 2:
        if label_2_tas_pred.numel() == 2:
            label_2_tas_pred = label_2_tas_pred.view(-1, 2)
        else:
            raise ValueError(f"label_2_tas_pred has wrong shape: {label_2_tas_pred.shape}, expected [B, 2]. "
                           f"transits_st_ed shape: {transits_st_ed.shape}, batch_size_total: {batch_size_total}")
    
    # Ensure batch sizes match between video and boundaries
    video_1_batch_size = video_1.shape[0]
    video_2_batch_size = video_2.shape[0]
    if label_1_tas_pred.shape[0] != video_1_batch_size:
        raise ValueError(
            f"Batch size mismatch for video_1: video has {video_1_batch_size} batches, "
            f"but boundaries has {label_1_tas_pred.shape[0]} batches. "
            f"video_1 shape: {video_1.shape}, label_1_tas_pred shape: {label_1_tas_pred.shape}"
        )
    if label_2_tas_pred.shape[0] != video_2_batch_size:
        raise ValueError(
            f"Batch size mismatch for video_2: video has {video_2_batch_size} batches, "
            f"but boundaries has {label_2_tas_pred.shape[0]} batches. "
            f"video_2 shape: {video_2.shape}, label_2_tas_pred shape: {label_2_tas_pred.shape}"
        )

    video_1_stage_bounds = scale_stage_boundaries(
        label_1_tas_pred, video_1.shape[2], tas_resolution
    )
    video_2_stage_bounds = scale_stage_boundaries(
        label_2_tas_pred, video_2.shape[2], tas_resolution
    )
    
    # Ensure boundaries have correct shape after scaling
    # scale_stage_boundaries should return [B, 2] shape
    if video_1_stage_bounds.dim() != 2 or video_1_stage_bounds.shape[1] != 2:
        raise ValueError(
            f"video_1_stage_bounds has wrong shape: {video_1_stage_bounds.shape}, expected [B, 2]. "
            f"label_1_tas_pred shape: {label_1_tas_pred.shape}"
        )
    if video_2_stage_bounds.dim() != 2 or video_2_stage_bounds.shape[1] != 2:
        raise ValueError(
            f"video_2_stage_bounds has wrong shape: {video_2_stage_bounds.shape}, expected [B, 2]. "
            f"label_2_tas_pred shape: {label_2_tas_pred.shape}"
        )
    
    # Ensure batch sizes match
    if video_1_stage_bounds.shape[0] != video_1.shape[0]:
        raise ValueError(
            f"Batch size mismatch: video_1 has {video_1.shape[0]} batches, "
            f"but video_1_stage_bounds has {video_1_stage_bounds.shape[0]} batches. "
            f"video_1 shape: {video_1.shape}, video_1_stage_bounds shape: {video_1_stage_bounds.shape}"
        )
    if video_2_stage_bounds.shape[0] != video_2.shape[0]:
        raise ValueError(
            f"Batch size mismatch: video_2 has {video_2.shape[0]} batches, "
            f"but video_2_stage_bounds has {video_2_stage_bounds.shape[0]} batches. "
            f"video_2 shape: {video_2.shape}, video_2_stage_bounds shape: {video_2_stage_bounds.shape}"
        )
    
    # Ensure video has correct shape: [B, C, T, H, W]
    # StageSegmenter expects 5D tensor
    if video_1.dim() != 5:
        raise ValueError(f"video_1 has wrong number of dimensions: {video_1.dim()}, expected 5 [B, C, T, H, W]")
    if video_2.dim() != 5:
        raise ValueError(f"video_2 has wrong number of dimensions: {video_2.dim()}, expected 5 [B, C, T, H, W]")

    from models.stage_modules import StageSegmenter
    from models.AutomaticWeightedLoss import AutomaticWeightedLoss
    
 
    if isinstance(stage_segmenter, AutomaticWeightedLoss):

        print(f"[WARNING] stage_segmenter is directly assigned to AutomaticWeightedLoss! "
              f"Attempting to recover by creating a new StageSegmenter.")
        stage_segmenter = StageSegmenter(
            stage_names=("Forward", "Twist", "Entry"),
            stage_clip_length=getattr(args, "stage_clip_length", 32),
        ).cuda()
    

    if hasattr(stage_segmenter, 'module'):
        if isinstance(stage_segmenter.module, AutomaticWeightedLoss):
    
            print(f"[WARNING] stage_segmenter.module is AutomaticWeightedLoss! "
                  f"This indicates stage_segmenter was incorrectly wrapped with awl's DDP wrapper. "
                  f"Attempting to recover by creating a new StageSegmenter.")
            stage_segmenter = StageSegmenter(
                stage_names=("Forward", "Twist", "Entry"),
                stage_clip_length=getattr(args, "stage_clip_length", 32),
            ).cuda()
        elif not isinstance(stage_segmenter.module, StageSegmenter):
            raise ValueError(
                f"stage_segmenter has wrong type: {type(stage_segmenter)} "
                f"(module: {type(stage_segmenter.module)}), expected StageSegmenter"
            )
        else:

            stage_segmenter = stage_segmenter.module
    elif not isinstance(stage_segmenter, StageSegmenter):
        # If not DDP wrapped and not StageSegmenter, try to recover
        if isinstance(stage_segmenter, AutomaticWeightedLoss):
            print(f"[WARNING] stage_segmenter is AutomaticWeightedLoss (not DDP wrapped)! "
                  f"Attempting to recover by creating a new StageSegmenter.")
            stage_segmenter = StageSegmenter(
                stage_names=("Forward", "Twist", "Entry"),
                stage_clip_length=getattr(args, "stage_clip_length", 32),
            ).cuda()
        else:
            raise ValueError(
                f"stage_segmenter has wrong type: {type(stage_segmenter)}, expected StageSegmenter"
            )

    video_1_stage_clips = stage_segmenter(video_1, video_1_stage_bounds)
    video_2_stage_clips = stage_segmenter(video_2, video_2_stage_bounds)

    video_1_stage_features = extract_stage_features(
        base_model, video_1_stage_clips, stage_segmenter.stage_names
    )
    video_2_stage_features = extract_stage_features(
        base_model, video_2_stage_clips, stage_segmenter.stage_names
    )

    from models.stage_modules import StageCausalReasoner

        if isinstance(stage_causal_reasoner.module, StageSegmenter):
 
            if isinstance(stage_segmenter, StageCausalReasoner):
                print(f"[WARNING] stage_causal_reasoner.module is StageSegmenter, but stage_segmenter is StageCausalReasoner! "
                      f"Swapping them back.")
                stage_causal_reasoner = stage_segmenter
                # Also need to fix stage_segmenter
                from models.stage_modules import StageSegmenter
                stage_segmenter = StageSegmenter(
                    stage_names=("Forward", "Twist", "Entry"),
                    stage_clip_length=getattr(args, "stage_clip_length", 32),
                ).cuda()
            else:
                raise ValueError(
                    f"CRITICAL BUG: stage_causal_reasoner.module is StageSegmenter! "
                    f"This indicates stage_causal_reasoner was incorrectly swapped with stage_segmenter. "
                    f"stage_causal_reasoner type: {type(stage_causal_reasoner)}, "
                    f"stage_causal_reasoner.module type: {type(stage_causal_reasoner.module)}, "
                    f"stage_segmenter type: {type(stage_segmenter)}"
                )
    elif isinstance(stage_causal_reasoner, StageSegmenter):

        if isinstance(stage_segmenter, StageCausalReasoner):
            print(f"[WARNING] stage_causal_reasoner is StageSegmenter, but stage_segmenter is StageCausalReasoner! "
                  f"Swapping them back.")
            stage_causal_reasoner = stage_segmenter
            # Also need to fix stage_segmenter
            from models.stage_modules import StageSegmenter
            stage_segmenter = StageSegmenter(
                stage_names=("Forward", "Twist", "Entry"),
                stage_clip_length=getattr(args, "stage_clip_length", 32),
            ).cuda()
        else:
            # CRITICAL FIX: If both are StageSegmenter, stage_causal_reasoner was lost
            # Recreate it from args
            print(f"[WARNING] stage_causal_reasoner is StageSegmenter but stage_segmenter is also StageSegmenter! "
                  f"This indicates stage_causal_reasoner was lost. Recreating StageCausalReasoner from args.")
            from models.stage_modules import StageCausalReasoner
            use_5node_subgraph = getattr(args, "use_5node_subgraph", True)
            use_inter_stage_intervention = getattr(args, "use_inter_stage_intervention", True)
            stage_feature_dim = getattr(args, "stage_feature_dim", 1024)
            stage_causal_reasoner = StageCausalReasoner(
                in_dim=stage_feature_dim,
                hidden_dim=getattr(args, "stage_reasoner_hidden_dim", 512),
                projection_dim=getattr(args, "stage_graph_dim", 64),
                num_heads=getattr(args, "stage_reasoner_heads", 4),
                num_stages=3,
                use_5node_subgraph=use_5node_subgraph,
                use_inter_stage_intervention=use_inter_stage_intervention,
                use_foreground=getattr(args, "use_foreground_node", True),
                use_body=getattr(args, "use_body_node", True),
                use_appearance=getattr(args, "use_appearance_node", True),
                use_motion=getattr(args, "use_motion_node", True),
                use_saliency=getattr(args, "use_saliency_node", True),
                use_directed_graph=getattr(args, "use_directed_graph", True),
            ).cuda()
            print(f"[WARNING] Recreated StageCausalReasoner. This may cause issues if it had trained weights.")

    (
        video_1_stage_graph,
        _,
        video_1_stage_scores,
        video_1_stage_recon,
        video_1_extra_outputs,
    ) = stage_causal_reasoner(video_1_stage_features)
    (
        video_2_stage_graph,
        _,
        video_2_stage_scores,
        video_2_stage_recon,
        video_2_extra_outputs,
    ) = stage_causal_reasoner(video_2_stage_features)

    padding_mask_1 = create_stage_padding_mask(video_1_stage_graph)
    padding_mask_2 = create_stage_padding_mask(video_2_stage_graph)
    
    video_1_stage_global, _ = temporal_causal_attn(video_1_stage_graph, padding_mask=padding_mask_1)
    video_2_stage_global, _ = temporal_causal_attn(video_2_stage_graph, padding_mask=padding_mask_2)

    ############# Stage-level auxiliary losses #############
    loss_stage = compute_stage_reconstruction_loss(
        video_1_stage_recon,
        video_2_stage_recon,
        video_1_stage_features,
        video_2_stage_features,
        args,
        epoch=epoch,
    )

    loss_global = None

    ############# Static feature #############
    v11, v12, v13 = video_encoder(video_1, label_1_tas_pred)
    v21, v22, v23 = video_encoder(video_2, label_2_tas_pred)

    ############# Interpolate #############
    N, T, C = video_1_fea.size()
    video_1_fea = video_1_fea.transpose(1, 2)
    video_1_fea_re_list = []
    for bs in range(N):
        v1i0 = (
            int(label_1_tas_pred[bs][0].item() + label_2_tas_pred[bs][0].item())
            // 20
        )
        v1i1 = (
            int(label_1_tas_pred[bs][1].item() + label_2_tas_pred[bs][1].item())
            // 20
        )
        video_1_fea_re_list.append(
            seg_pool_1d(video_1_fea[bs].unsqueeze(0), v1i0, v1i1, 4)
        )
    video_1_fea_re = torch.cat(video_1_fea_re_list, 0).transpose(1, 2)

    video_2_fea = video_2_fea.transpose(1, 2)
    video_2_fea_re_list = []
    for bs in range(N):
        v1i0 = (
            int(label_1_tas_pred[bs][0].item() + label_2_tas_pred[bs][0].item())
            // 20
        )
        v1i1 = (
            int(label_1_tas_pred[bs][1].item() + label_2_tas_pred[bs][1].item())
            // 20
        )
        video_2_fea_re_list.append(
            seg_pool_1d(video_2_fea[bs].unsqueeze(0), v1i0, v1i1, 4)
        )
    video_2_fea_re = torch.cat(video_2_fea_re_list, 0).transpose(1, 2)

    ############# Lower dimension #############
    video_1_segs_1 = dim_reducer3(video_1_fea_re[:, 0:4, :])
    video_1_segs_2 = dim_reducer1(video_1_fea_re[:, 4:8, :])
    video_1_segs_3 = dim_reducer2(video_1_fea_re[:, 8:12, :])
    video_2_segs_1 = dim_reducer3(video_2_fea_re[:, 0:4, :])
    video_2_segs_2 = dim_reducer1(video_2_fea_re[:, 4:8, :])
    video_2_segs_3 = dim_reducer2(video_2_fea_re[:, 8:12, :])

    video_1_segs_1 = (
        video_1_segs_1
        + video_1_stage_graph[:, 0].unsqueeze(1)
        + video_1_stage_global[:, 0].unsqueeze(1)
    )
    video_1_segs_2 = (
        video_1_segs_2
        + video_1_stage_graph[:, 1].unsqueeze(1)
        + video_1_stage_global[:, 1].unsqueeze(1)
    )
    video_1_segs_3 = (
        video_1_segs_3
        + video_1_stage_graph[:, 2].unsqueeze(1)
        + video_1_stage_global[:, 2].unsqueeze(1)
    )

    video_2_segs_1 = (
        video_2_segs_1
        + video_2_stage_graph[:, 0].unsqueeze(1)
        + video_2_stage_global[:, 0].unsqueeze(1)
    )
    video_2_segs_2 = (
        video_2_segs_2
        + video_2_stage_graph[:, 1].unsqueeze(1)
        + video_2_stage_global[:, 1].unsqueeze(1)
    )
    video_2_segs_3 = (
        video_2_segs_3
        + video_2_stage_graph[:, 2].unsqueeze(1)
        + video_2_stage_global[:, 2].unsqueeze(1)
    )
    
    loss_boundary = None
    if boundary_regression_head is not None and boundary_regression_loss is not None:
        video_1_fused_features = video_1_stage_graph.view(video_1_stage_graph.shape[0], -1)
        video_2_fused_features = video_2_stage_graph.view(video_2_stage_graph.shape[0], -1)
        
        pred_boundaries_1, pred_completeness_1, pred_category_1 = boundary_regression_head(
            video_1_fused_features, temporal_length=video_1.shape[2]
        )
        pred_boundaries_2, pred_completeness_2, pred_category_2 = boundary_regression_head(
            video_2_fused_features, temporal_length=video_2.shape[2]
        )
        
        if label_1_boundaries is not None and label_2_boundaries is not None:
            temporal_length_1 = video_1.shape[2]
            temporal_length_2 = video_2.shape[2]
            
            target_boundaries_1 = label_1_boundaries.float() / temporal_length_1
            target_boundaries_2 = label_2_boundaries.float() / temporal_length_2
            
            target_completeness_1 = label_1_score if label_1_score is not None else torch.ones_like(pred_completeness_1)
            target_completeness_2 = label_2_score if label_2_score is not None else torch.ones_like(pred_completeness_2)
            
            loss_boundary_1, loss_dict_1 = boundary_regression_loss(
                pred_boundaries_1, pred_completeness_1, pred_category_1,
                target_boundaries_1, target_completeness_1, None
            )
            
            loss_boundary_2, loss_dict_2 = boundary_regression_loss(
                pred_boundaries_2, pred_completeness_2, pred_category_2,
                target_boundaries_2, target_completeness_2, None
            )
            
            loss_boundary = loss_boundary_1 + loss_boundary_2

    ############# Fusion dynamic and static feature #############
    v_12_list = []
    v_21_list = []
    v_12 = decoder[0](v11, v21)
    v_21 = decoder[0](v21, v11)
    v_12_list.append(v_12)
    v_21_list.append(v_21)

    v_12 = decoder[0](v12, v22)
    v_21 = decoder[0](v22, v12)
    v_12_list.append(v_12)
    v_21_list.append(v_21)

    v_12 = decoder[2](v13, v23)
    v_21 = decoder[2](v23, v13)
    v_12_list.append(v_12)
    v_21_list.append(v_21)

    v_12_map = torch.cat(v_12_list, 1)
    v_21_map = torch.cat(v_21_list, 1)
    v_12_21 = torch.cat((v_12_map, v_21_map), 0)

    ############# Cross attention #############
    decoder_video_12_map_list = []
    decoder_video_21_map_list = []

    decoder_video_12_map = decoder[1](video_1_segs_1, video_2_segs_1)
    decoder_video_21_map = decoder[1](video_2_segs_1, video_1_segs_1)
    decoder_video_12_map_list.append(decoder_video_12_map)
    decoder_video_21_map_list.append(decoder_video_21_map)

    decoder_video_12_map = decoder[1](video_1_segs_2, video_2_segs_2)
    decoder_video_21_map = decoder[1](video_2_segs_2, video_1_segs_2)
    decoder_video_12_map_list.append(decoder_video_12_map)
    decoder_video_21_map_list.append(decoder_video_21_map)
    
    decoder_video_12_map = decoder[3](video_1_segs_3, video_2_segs_3)
    decoder_video_21_map = decoder[3](video_2_segs_3, video_1_segs_3)
    decoder_video_12_map_list.append(decoder_video_12_map)
    decoder_video_21_map_list.append(decoder_video_21_map)

    decoder_video_12_map = torch.cat(decoder_video_12_map_list, 1)
    decoder_video_21_map = torch.cat(decoder_video_21_map_list, 1)

    ############# Fine-grained Contrastive Regression #############
    decoder_12_21 = torch.cat((decoder_video_12_map, decoder_video_21_map), 0)

    delta1 = regressor_delta[0](decoder_12_21)
    delta2 = regressor_delta[1](v_12_21)

    delta1_1 = delta1[:, :12].mean(1)
    delta1_2 = delta1[:, 12:24].mean(1)
    delta1_3 = delta1[:, 24:].mean(1)

    delta2_1 = delta2[:, :4].mean(1)
    delta2_2 = delta2[:, 4:8].mean(1)
    delta2_3 = delta2[:, 8:].mean(1)
    

    video_1_saliency = video_1_stage_scores.squeeze(-1)  # [B, 3]
    video_2_saliency = video_2_stage_scores.squeeze(-1)  # [B, 3]

    video_1_weights = F.softmax(video_1_saliency, dim=1)
    video_2_weights = F.softmax(video_2_saliency, dim=1)
    

    delta1 = (delta1_1 * video_1_weights[:, 0:1] + 
              delta1_2 * video_1_weights[:, 1:2] + 
              delta1_3 * video_1_weights[:, 2:3]).squeeze(-1)
    
    delta2 = (delta2_1 * video_2_weights[:, 0:1] + 
              delta2_2 * video_2_weights[:, 1:2] + 
              delta2_3 * video_2_weights[:, 2:3]).squeeze(-1)
    
    # Decoder fusion (configurable)
    decoder_fusion_alpha = getattr(args, 'decoder_fusion_alpha', 0.5)
    delta = decoder_fusion_alpha * delta1 + (1 - decoder_fusion_alpha) * delta2
    delta = delta.unsqueeze(-1)
    

    video_1_weighted_score = (video_1_stage_scores.squeeze(-1) * video_1_weights).sum(dim=1)
    video_2_weighted_score = (video_2_stage_scores.squeeze(-1) * video_2_weights).sum(dim=1)
    stage_score_delta = video_1_weighted_score - video_2_weighted_score
    

    stage_fusion_alpha = getattr(args, 'stage_score_fusion_alpha', 0.3)
    delta_combined = delta[: delta.shape[0] // 2] + stage_fusion_alpha * stage_score_delta.unsqueeze(-1)
    delta = torch.cat((delta_combined, delta[delta.shape[0] // 2 :]), dim=0)


    pred_diff = delta[: delta.shape[0] // 2].squeeze()
    true_diff = (label_1_score - label_2_score).squeeze()
    

    mse_loss = F.mse_loss(pred_diff, true_diff)
    

    use_correlation_loss = getattr(args, 'use_correlation_loss', False)
    if use_correlation_loss:
        pred_centered = pred_diff - pred_diff.mean()
        true_centered = true_diff - true_diff.mean()
        
        numerator = (pred_centered * true_centered).sum()
        denominator = torch.sqrt((pred_centered ** 2).sum()) * torch.sqrt((true_centered ** 2).sum()) + 1e-8
        correlation = numerator / denominator
        correlation_loss = 1 - correlation  # Convert to loss (maximize correlation)
        
        correlation_weight = getattr(args, 'correlation_loss_weight', 0.1)
        loss_aqa = mse_loss + correlation_weight * correlation_loss
    else:
        loss_aqa = mse_loss
    
    # NEW: Causal regularization loss (if 5-node subgraph is enabled)
    loss_causal_reg = None
    if 'intervention_gates' in video_1_extra_outputs and 'intervention_gates' in video_2_extra_outputs:
        forward_to_twist_gate_1 = video_1_extra_outputs['intervention_gates']['forward_to_twist_gate']
        twist_to_entry_gate_1 = video_1_extra_outputs['intervention_gates']['twist_to_entry_gate']
        forward_to_twist_gate_2 = video_2_extra_outputs['intervention_gates']['forward_to_twist_gate']
        twist_to_entry_gate_2 = video_2_extra_outputs['intervention_gates']['twist_to_entry_gate']
        
        gate_reg = 0.5 * (
            torch.mean((forward_to_twist_gate_1 - 0.5) ** 2) +
            torch.mean((twist_to_entry_gate_1 - 0.5) ** 2) +
            torch.mean((forward_to_twist_gate_2 - 0.5) ** 2) +
            torch.mean((twist_to_entry_gate_2 - 0.5) ** 2)
        )
        
        causal_reg_weight = getattr(args, 'causal_reg_weight', 0.1)
        loss_causal_reg = causal_reg_weight * gate_reg

    t_loss[0] += loss_aqa.item() if hasattr(loss_aqa, 'item') else loss_aqa
    t_loss[1] += loss_tas.item() if hasattr(loss_tas, 'item') else loss_tas
    t_loss[2] += loss_mask.item() if loss_mask is not None and hasattr(loss_mask, 'item') else (loss_mask if loss_mask is not None else 0.0)
    t_loss[3] += loss_stage.item() if loss_stage is not None and hasattr(loss_stage, 'item') else (loss_stage if loss_stage is not None else 0.0)
    t_loss[4] += loss_global.item() if loss_global is not None and hasattr(loss_global, 'item') else (loss_global if loss_global is not None else 0.0)
    t_loss[5] += loss_causal_reg.item() if loss_causal_reg is not None and hasattr(loss_causal_reg, 'item') else (loss_causal_reg if loss_causal_reg is not None else 0.0)

    score_accumulator.append(delta[: delta.shape[0] // 2].detach() + label_2_score)

    for bs in range(N):
        tIoU_results.append(
            segment_iou(
                np.array(label_12_tas.squeeze(-1).cpu())[bs],
                np.array(transits_st_ed.squeeze(-1).cpu())[bs],
                args,
            )
        )

    if score_accumulator:
        score = torch.stack(score_accumulator, dim=0).sum(dim=0)
    else:
        score = torch.zeros_like(label_1_score, dtype=torch.float32)
    averaged_scores = score / max(len(score_accumulator), 1)
    scores = averaged_scores.detach().cpu().view(-1).tolist()
    if use_difficulty:
        difficulty_tensor = (
            difficulty
            if torch.is_tensor(difficulty)
            else torch.tensor(difficulty, device=averaged_scores.device)
        )
        scores_tensor = torch.tensor(scores, device=difficulty_tensor.device, dtype=difficulty_tensor.dtype)
        scores = (scores_tensor * difficulty_tensor).tolist()
    pred_scores.extend(scores)

    tIoU_results_mean = [sum(tIoU_results) / len(tIoU_results)]
    tiou_thresholds = np.array([0.5, 0.75])
    tIoU_correct_per_thr = cal_tiou(tIoU_results_mean, tiou_thresholds)
    pred_tious_test_5.extend([tIoU_correct_per_thr[0]])
    pred_tious_test_75.extend([tIoU_correct_per_thr[1]])

    for i in range(len(t_loss)):
        t_loss[i] /= args.voter_number
    return t_loss


def save_checkpoint(
    base_model,
    psnet_model,
    decoder,
    regressor_delta,
    video_encoder,
    dim_reducer3,
    segmenter,
    dim_reducer1,
    dim_reducer2,
    # causal_intervention, 
    temporal_causal_attn,
    awl,
    stage_causal_reasoner,
    optimizer,
    epoch,
    epoch_best_aqa,
    rho_best,
    L2_min,
    RL2_min,
    prefix,
    args,
    boundary_regression_head=None,  # NEW
):

    from tools.builder import CHECKPOINT_VERSION
    
    checkpoint_dict = {
        "checkpoint_version": CHECKPOINT_VERSION,  # Add version info for compatibility checking
        "base_model": get_state_dict(base_model),
        "psnet_model": get_state_dict(psnet_model),
        "decoder1": get_state_dict(decoder[0]),
        "decoder2": get_state_dict(decoder[1]),
        "decoder3": get_state_dict(decoder[2]),
        "decoder4": get_state_dict(decoder[3]),
        "regressor_delta1": get_state_dict(regressor_delta[0]),
        "regressor_delta2": get_state_dict(regressor_delta[1]),
        "regressor_delta3": get_state_dict(regressor_delta[2]),
        "video_encoder": get_state_dict(video_encoder),
        # OPTIMIZED: All 3 dim_reducers share parameters, so we only save once
        # Save under all 3 keys for backward compatibility with old checkpoints
        "dim_reducer3": get_state_dict(dim_reducer1),  # All 3 are the same now
        "dim_reducer1": get_state_dict(dim_reducer1),
        "dim_reducer2": get_state_dict(dim_reducer1),
        # "causal_intervention": get_state_dict(causal_intervention),  # REMOVED
        "temporal_causal_attn": get_state_dict(temporal_causal_attn),
        "awl": get_state_dict(awl),
        "stage_causal_reasoner": get_state_dict(stage_causal_reasoner),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "epoch_best_aqa": epoch_best_aqa,
        "rho_best": rho_best,
        "L2_min": L2_min,
        "RL2_min": RL2_min,
        "segmenter": get_state_dict(segmenter),
    }
    

    if boundary_regression_head is not None:
        checkpoint_dict["boundary_regression_head"] = get_state_dict(boundary_regression_head)
    
    torch.save(
        checkpoint_dict,
        os.path.join(args.experiment_path, prefix + ".pth"),
    )


def save_outputs(pred_scores, true_scores, args, epoch):
    save_path_pred = os.path.join(
        args.experiment_path, f"pred_{settigns_date_time}_{epoch}.npy"
    )
    save_path_true = os.path.join(
        args.experiment_path, f"true_{settigns_date_time}_{epoch}.npy"
    )
    np.save(save_path_pred, pred_scores)
    np.save(save_path_true, true_scores)
