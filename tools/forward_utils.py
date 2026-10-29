import torch
import torch.nn.functional as F
from einops import rearrange
import segmentation_models_pytorch as smp
from models.stage_modules import StageSegmenter, StageCausalReasoner
from models.AutomaticWeightedLoss import AutomaticWeightedLoss


def compute_segmentation_loss_and_metrics(
    segmenter, video_1, video_2, video_1_mask, video_2_mask, focal_loss, segment_metrics
):

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
    for i in range(5):
        focal_loss_val = focal_loss(mask_pred_[i], mask_target)
        if loss_mask is None:
            loss_mask = focal_loss_val
        else:
            loss_mask = loss_mask + focal_loss_val

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

    return loss_mask, mask_feature


def extract_video_features(base_model, video_1, video_2, mask_feature):

    com_feature_12, com_feamap_12 = base_model(video_1, video_2)
    orig_video_1_fea = com_feature_12[:, :, : com_feature_12.shape[2] // 2]
    orig_video_2_fea = com_feature_12[:, :, com_feature_12.shape[2] // 2 :]
    com_feature_12_u = torch.cat((orig_video_1_fea, orig_video_2_fea), 0)

    # Mask and I3D Feature Fusion
    u_fea_96 = com_feature_12_u * torch.sigmoid(mask_feature)
    
    # Get individual video features
    video_1_fea = u_fea_96[: u_fea_96.shape[0] // 2]
    video_2_fea = u_fea_96[u_fea_96.shape[0] // 2 :]
    
    return video_1_fea, video_2_fea


def predict_transitions(psnet_model, video_1_fea, video_2_fea, label_1_tas, label_2_tas, bce):

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
    
    batch_size_total = transits_st_ed.shape[0]
    batch_size_per_video = batch_size_total // 2
    
    label_1_tas_pred = transits_st_ed[:batch_size_per_video]
    label_2_tas_pred = transits_st_ed[batch_size_per_video:]
    tas_resolution = transits_pred.shape[1]
    
    return loss_tas, label_1_tas_pred, label_2_tas_pred, tas_resolution


def validate_and_fix_stage_segmenter(stage_segmenter, args):

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
                  f"Attempting to recover by creating a new StageSegmenter.")
            stage_segmenter = StageSegmenter(
                stage_names=("Forward", "Twist", "Entry"),
                stage_clip_length=getattr(args, "stage_clip_length", 32),
            ).cuda()
        elif isinstance(stage_segmenter.module, StageCausalReasoner):
            print(f"[WARNING] stage_segmenter.module is StageCausalReasoner! "
                  f"This indicates stage_segmenter and stage_causal_reasoner were swapped. "
                  f"Restoring from original stage_segmenter.")
            # This should be handled by restoring from backup, but we create new one as fallback
            stage_segmenter = StageSegmenter(
                stage_names=("Forward", "Twist", "Entry"),
                stage_clip_length=getattr(args, "stage_clip_length", 32),
            ).cuda()
        elif not isinstance(stage_segmenter.module, StageSegmenter):
            raise ValueError(
                f"CRITICAL BUG: stage_segmenter.module is not StageSegmenter! "
                f"stage_segmenter type: {type(stage_segmenter)}, "
                f"stage_segmenter.module type: {type(stage_segmenter.module)}"
            )
        else:
            # Unwrap DDP if needed
            stage_segmenter = stage_segmenter.module
    elif not isinstance(stage_segmenter, StageSegmenter):
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
    
    return stage_segmenter


def validate_and_fix_stage_causal_reasoner(stage_causal_reasoner, stage_segmenter, args):

    if hasattr(stage_causal_reasoner, 'module'):
        if isinstance(stage_causal_reasoner.module, StageSegmenter):
            if isinstance(stage_segmenter, StageCausalReasoner):
                print(f"[WARNING] stage_causal_reasoner.module is StageSegmenter, but stage_segmenter is StageCausalReasoner! "
                      f"Swapping them back.")
                stage_causal_reasoner = stage_segmenter
            else:
                raise ValueError(
                    f"CRITICAL BUG: stage_causal_reasoner.module is StageSegmenter! "
                    f"stage_causal_reasoner type: {type(stage_causal_reasoner)}, "
                    f"stage_causal_reasoner.module type: {type(stage_causal_reasoner.module)}, "
                    f"stage_segmenter type: {type(stage_segmenter)}"
                )
    elif isinstance(stage_causal_reasoner, StageSegmenter):
        if isinstance(stage_segmenter, StageCausalReasoner):
            print(f"[WARNING] stage_causal_reasoner is StageSegmenter, but stage_segmenter is StageCausalReasoner! "
                  f"Swapping them back.")
            stage_causal_reasoner = stage_segmenter
        else:
            # Recreate StageCausalReasoner from args
            print(f"[WARNING] stage_causal_reasoner is StageSegmenter but stage_segmenter is also StageSegmenter! "
                  f"This indicates stage_causal_reasoner was lost. Recreating StageCausalReasoner from args.")
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
    
    return stage_causal_reasoner


def process_single_video_stages(
    video, stage_bounds, stage_segmenter, base_model, args
):

    # Validate inputs
    if video.dim() != 5:
        raise ValueError(f"video has wrong number of dimensions: {video.dim()}, expected 5 [B, C, T, H, W]")
    
    if stage_bounds.dim() != 2 or stage_bounds.shape[1] != 2:
        raise ValueError(
            f"stage_bounds has wrong shape: {stage_bounds.shape}, expected [B, 2]"
        )
    
    if stage_bounds.shape[0] != video.shape[0]:
        raise ValueError(
            f"Batch size mismatch: video has {video.shape[0]} batches, "
            f"but stage_bounds has {stage_bounds.shape[0]} batches"
        )
    
    # Segment video into stages
    stage_clips = stage_segmenter(video, stage_bounds)
    
    # Extract features for each stage
    from tools.helper import extract_stage_features
    stage_features = extract_stage_features(
        base_model, stage_clips, stage_segmenter.stage_names
    )
    
    return stage_features


def validate_boundaries_shape(boundaries, video_shape, name="boundaries"):

    # Ensure boundaries have correct shape: [B, 2]
    if boundaries.dim() == 1:
        boundaries = boundaries.unsqueeze(0)
    
    if boundaries.shape[1] != 2:
        if boundaries.numel() == 2:
            boundaries = boundaries.view(-1, 2)
        else:
            raise ValueError(
                f"{name} has wrong shape: {boundaries.shape}, expected [B, 2]"
            )
    
    # Ensure batch sizes match
    if boundaries.shape[0] != video_shape[0]:
        raise ValueError(
            f"Batch size mismatch: video has {video_shape[0]} batches, "
            f"but {name} has {boundaries.shape[0]} batches"
        )
    
    return boundaries

