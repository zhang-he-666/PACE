import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))

import torch
import torch.optim as optim
from models import I3D_backbone
from models.PS import PSNet
from utils.misc import import_class
from torchvideotransforms import video_transforms, volume_transforms
from models import decoder_fuser
from models import MLP_score
from models import MLP_score, MLP_twist, I3D_VOS
from models import (
    MaskEncoder,
    AttentionFusion,
    FeatureFusionModule,
    VideoEncoder,
    C_channel,
)
# Historical causal intervention modules have been fully deprecated.
# Keep imports minimal to avoid pulling non-existent files.
from models.causal_attention import CausalAttention
# FeatureGATCausal is deprecated - use StageCausalReasoner instead
# from models.causal_GAT import FeatureGATCausal
from models.stage_modules import StageSegmenter, StageCausalReasoner
from models.PS import PSNet, Pred_twistoffset
from models.AutomaticWeightedLoss import AutomaticWeightedLoss
from datasets.FineDiving_Pair import DebugDataset
import torch
import torchvision.transforms.functional as F
import random
import PIL
from timm.scheduler.cosine_lr import CosineLRScheduler


def get_video_trans():
    train_trans = DualCompose(
        [
            DualRandomHorizontalFlip(),
            DualResize((200, 112)),
            DualRandomCrop(112),
            DualClipToTensor(),
            DualNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_trans = DualCompose(
        [
            DualResize((200, 112)),
            DualCenterCrop(112),
            DualClipToTensor(),
            DualNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_trans, test_trans


def get_video_trans1():
    train_trans = video_transforms.Compose(
        [
            video_transforms.RandomHorizontalFlip(),
            video_transforms.Resize((200, 112)),
            video_transforms.RandomCrop(112),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    test_trans = video_transforms.Compose(
        [
            video_transforms.Resize((200, 112)),
            video_transforms.CenterCrop(112),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    return train_trans, test_trans


def dataset_builder(args):
    train_trans, test_trans = get_video_trans()
    Dataset = import_class("datasets." + args.benchmark)
    train_dataset = Dataset(args, transform=train_trans, subset="train")
    test_dataset = Dataset(args, transform=test_trans, subset="test")

    return train_dataset, test_dataset


def model_builder(args):
    # Enable two-stream (RGB + Optical Flow) if configured
    # Disable optical-flow two-stream; use RGB-only backbone
    use_two_stream = False
    flow_ckpt_path = getattr(args, 'flow_ckpt_path', None)
    base_model = I3D_backbone(I3D_class=400, use_two_stream=use_two_stream, flow_ckpt_path=flow_ckpt_path)
    base_model.load_pretrain(args.pretrained_i3d_weight, flow_ckpt_path=flow_ckpt_path)
    PSNet_model = PSNet()
    Decoder_vit1 = decoder_fuser(dim=64, num_heads=8, num_layers=3)
    Decoder_vit2 = decoder_fuser(dim=64, num_heads=8, num_layers=3)
    Decoder_vit3 = decoder_fuser(dim=64, num_heads=8, num_layers=3)
    Decoder_vit4 = decoder_fuser(dim=64, num_heads=8, num_layers=3)
    Regressor_delta1 = MLP_score(in_channel=64, out_channel=1)
    Regressor_delta2 = MLP_score(in_channel=64, out_channel=1)
    Regressor_delta3 = MLP_score(in_channel=64, out_channel=1)

    shared_dim_reducer = C_channel(in_channel=4)
    dim_reducer1 = shared_dim_reducer  # Share parameters
    dim_reducer2 = shared_dim_reducer  # Share parameters
    dim_reducer3 = shared_dim_reducer  # Share parameters
    segmenter = I3D_VOS(num_classes=400)
    Video_Encoder = VideoEncoder()
 
    causal_intervention = None
)
    temporal_causal_attn = CausalAttention(embed_size=64, n_heads=1, num_layers=2)
    main_task_weight = getattr(args, "main_task_weight", 6.0)  # Increased from 5.0 to 6.0 to focus more on AQA task
    awl = AutomaticWeightedLoss(3, main_task_weight=main_task_weight)
    stage_segmenter = StageSegmenter(
        stage_names=("Forward", "Twist", "Entry"),
        stage_clip_length=getattr(args, "stage_clip_length", 32),
    )
    use_5node_subgraph = getattr(args, "use_5node_subgraph", True)
    use_inter_stage_intervention = getattr(args, "use_inter_stage_intervention", True)
    

    stage_feature_dim = getattr(args, "stage_feature_dim", 1024)  
    
    stage_causal_reasoner = StageCausalReasoner(
        in_dim=stage_feature_dim,  # Use the actual feature dimension
        hidden_dim=getattr(args, "stage_reasoner_hidden_dim", 512),
        projection_dim=getattr(args, "stage_graph_dim", 64),
        num_heads=getattr(args, "stage_reasoner_heads", 4),
        num_stages=3,
        use_5node_subgraph=use_5node_subgraph,  # Ablation flag
        use_inter_stage_intervention=use_inter_stage_intervention,  # Ablation flag
        use_foreground=getattr(args, "use_foreground_node", True),  # Ablation flag: w/o F_k
        use_body=getattr(args, "use_body_node", True),  # Ablation flag: w/o B_k
        use_appearance=getattr(args, "use_appearance_node", True),  # Ablation flag: w/o A_k
        use_motion=getattr(args, "use_motion_node", True),  # Ablation flag: w/o M_k
        use_saliency=getattr(args, "use_saliency_node", True),  # Ablation flag: w/o S_k
        use_directed_graph=getattr(args, "use_directed_graph", True),  # Ablation flag: directed vs undirected
    )
    
    use_boundary_regression = getattr(args, "use_boundary_regression", True)
    boundary_regression_head = None
    boundary_regression_loss = None
    if use_boundary_regression:
        from models.boundary_regression import BoundaryRegressionHead, BoundaryRegressionLoss
        
        boundary_regression_head = BoundaryRegressionHead(
            in_dim=getattr(args, "stage_graph_dim", 64) * 3,  # 3 stages concatenated
            hidden_dim=getattr(args, "boundary_regression_hidden_dim", 256),
            num_categories=getattr(args, "num_action_categories", 1),
            use_category=getattr(args, "use_action_category", False)
        )
        
        boundary_regression_loss = BoundaryRegressionLoss(
            l1_weight=getattr(args, "boundary_l1_weight", 1.0),
            giou_weight=getattr(args, "boundary_giou_weight", 1.0),
            completeness_weight=getattr(args, "boundary_completeness_weight", 0.5),
            category_weight=getattr(args, "boundary_category_weight", 0.3),
            use_category=getattr(args, "use_action_category", False)
        )

    return (
        base_model,
        PSNet_model,
        [Decoder_vit1, Decoder_vit2, Decoder_vit3, Decoder_vit4],
        [Regressor_delta1, Regressor_delta2, Regressor_delta3],
        dim_reducer1,
        dim_reducer2,
        Video_Encoder,
        dim_reducer3,
        segmenter,
        None,  # causal_intervention (removed)
        temporal_causal_attn,
        awl,
        stage_segmenter,
        stage_causal_reasoner,
        boundary_regression_head,
        boundary_regression_loss,
    )


def get_optimizer_parameters(
    base_model,
    psnet_model,
    decoder,
    regressor_delta,
    dim_reducer1,
    video_encoder,
    segmenter,
    temporal_causal_attn,
    awl,
    stage_causal_reasoner,
    args,
    boundary_regression_head=None,
):

    param_groups = [
        {
            "params": base_model.parameters(),
            "lr": args.base_lr * args.lr_factor,
        },
        {"params": psnet_model.parameters()},
        {"params": decoder[0].parameters()},
        {"params": decoder[1].parameters()},
        {"params": decoder[2].parameters()},
        {"params": decoder[3].parameters()},
        {"params": regressor_delta[0].parameters()},
        {"params": regressor_delta[1].parameters()},
        {"params": regressor_delta[2].parameters()},
        # OPTIMIZED: dim_reducer1, dim_reducer2, dim_reducer3 now share parameters
        # Only add once to avoid "parameters appear in more than one parameter group" error
        {"params": dim_reducer1.parameters()},  # This includes all 3 since they share parameters
        {"params": segmenter.parameters()},
        {"params": video_encoder.parameters()},
        {"params": temporal_causal_attn.parameters()},
        {"params": awl.parameters()},
        {"params": stage_causal_reasoner.parameters()},
    ]
    
    if boundary_regression_head is not None:
        param_groups.append({"params": boundary_regression_head.parameters()})
    
    return param_groups


def build_opti_sche(
    base_model,
    psnet_model,
    decoder,
    regressor_delta,
    dim_reducer1,
    dim_reducer2,
    dim_reducer3,
    video_encoder,
    segmenter,
    causal_intervention,
    temporal_causal_attn,
    awl,
    stage_causal_reasoner,
    args,
    n_iter_per_epoch,
    boundary_regression_head=None,
):

    param_groups = get_optimizer_parameters(
        base_model, psnet_model, decoder, regressor_delta, dim_reducer1,
        video_encoder, segmenter, temporal_causal_attn, awl, stage_causal_reasoner,
        args, boundary_regression_head
    )
    
    optimizer_name = getattr(args, "optimizer", "Adam").lower()
    
    if optimizer_name == "adam":
        optimizer = optim.Adam(
            param_groups,
            lr=args.base_lr,
            weight_decay=getattr(args, "weight_decay", 0.0001),
            betas=getattr(args, "adam_betas", (0.9, 0.999)),
            eps=getattr(args, "adam_eps", 1e-8),
        )
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(
            param_groups,
            lr=args.base_lr,
            weight_decay=getattr(args, "weight_decay", 0.01),
            betas=getattr(args, "adamw_betas", (0.9, 0.999)),
            eps=getattr(args, "adamw_eps", 1e-8),
        )
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            param_groups,
            lr=args.base_lr,
            momentum=getattr(args, "sgd_momentum", 0.9),
            weight_decay=getattr(args, "weight_decay", 0.0001),
            nesterov=getattr(args, "sgd_nesterov", False),
        )
    else:
        raise NotImplementedError(
            f"Optimizer '{optimizer_name}' not supported. "
            f"Supported optimizers: Adam, AdamW, SGD"
        )

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.max_epoch * n_iter_per_epoch,
        lr_min=args.base_lr * args.lr_factor,
        warmup_lr_init=args.base_lr * args.lr_factor,
        warmup_t=args.warmup_epochs * n_iter_per_epoch,
        cycle_limit=15,
        t_in_epochs=False,
    )
    return optimizer, scheduler


# Checkpoint version management
CHECKPOINT_VERSION = "1.0"  # Current checkpoint version
MIN_SUPPORTED_VERSION = "1.0"  # Minimum supported version


def get_checkpoint_version(state_dict):

    return state_dict.get("checkpoint_version", "0.0")  # Default to 0.0 for old checkpoints


def validate_checkpoint_version(state_dict):

    version = get_checkpoint_version(state_dict)
    if version < MIN_SUPPORTED_VERSION:
        print(f"[WARNING] Checkpoint version {version} is older than minimum supported version {MIN_SUPPORTED_VERSION}")
        print(f"[WARNING] Some features may not be compatible. Proceeding with caution...")
    return version


def count_parameters(model, name="Model"):

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"{name}:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    
    return total_params, trainable_params


def print_model_statistics(
    base_model,
    psnet_model,
    decoder,
    regressor_delta,
    dim_reducer1,
    video_encoder,
    segmenter,
    temporal_causal_attn,
    awl,
    stage_causal_reasoner,
    boundary_regression_head=None,
):

    print("\n" + "="*60)
    print("MODEL PARAMETER STATISTICS")
    print("="*60)
    
    total_all = 0
    trainable_all = 0
    
    total, trainable = count_parameters(base_model, "Base Model (I3D)")
    total_all += total
    trainable_all += trainable
    
    total, trainable = count_parameters(psnet_model, "PSNet")
    total_all += total
    trainable_all += trainable
    
    for i, dec in enumerate(decoder):
        total, trainable = count_parameters(dec, f"Decoder[{i}]")
        total_all += total
        trainable_all += trainable
    
    for i, reg in enumerate(regressor_delta):
        total, trainable = count_parameters(reg, f"Regressor Delta[{i}]")
        total_all += total
        trainable_all += trainable
    
    total, trainable = count_parameters(dim_reducer1, "Dim Reducer (shared)")
    total_all += total
    trainable_all += trainable
    
    total, trainable = count_parameters(video_encoder, "Video Encoder")
    total_all += total
    trainable_all += trainable
    
    total, trainable = count_parameters(segmenter, "Segmenter")
    total_all += total
    trainable_all += trainable
    
    total, trainable = count_parameters(temporal_causal_attn, "Temporal Causal Attention")
    total_all += total
    trainable_all += trainable
    
    total, trainable = count_parameters(awl, "Automatic Weighted Loss")
    total_all += total
    trainable_all += trainable
    
    total, trainable = count_parameters(stage_causal_reasoner, "Stage Causal Reasoner")
    total_all += total
    trainable_all += trainable
    
    if boundary_regression_head is not None:
        total, trainable = count_parameters(boundary_regression_head, "Boundary Regression Head")
        total_all += total
        trainable_all += trainable
    
    print("-"*60)
    print(f"TOTAL: {total_all:,} parameters ({trainable_all:,} trainable)")
    print("="*60 + "\n")
    
    return total_all, trainable_all


def resume_train(
    base_model,
    psnet_model,
    decoder,
    dim_reducer1,
    dim_reducer2,
    optimizer,
    dim_reducer3,
    regressor_delta,
    segmenter,
    video_encoder,
    causal_intervention,  # Can be None now
    temporal_causal_attn,
    awl,
    stage_causal_reasoner,
    args,
):
    ckpt_path = os.path.join(args.experiment_path, "best.pth")
    if not os.path.exists(ckpt_path):
        print("no checkpoint file from path %s..." % ckpt_path)
        return 0, 0, 0, 1000, 1000
    print("Loading weights from %s..." % ckpt_path)

    # load state dict
    state_dict = torch.load(ckpt_path, map_location="cpu")
    
    # Validate checkpoint version
    checkpoint_version = validate_checkpoint_version(state_dict)
    print(f"Checkpoint version: {checkpoint_version}")

    # parameter resume of base model
    base_ckpt = {}
    for k, v in state_dict["base_model"].items():
        # Remove DDP prefix if present
        new_k = k.replace("module.", "")
        # Handle old checkpoint format: fusion_attention -> fusion_module.attention
        # Handle old checkpoint format: fusion_rgb_proj -> fusion_module.rgb_proj
        # Handle old checkpoint format: fusion_flow_proj -> fusion_module.flow_proj
        # Handle old checkpoint format: fusion_fusion_proj -> fusion_module.fusion_proj
        if new_k.startswith("fusion_attention"):
            new_k = new_k.replace("fusion_attention", "fusion_module.attention")
        elif new_k.startswith("fusion_rgb_proj"):
            new_k = new_k.replace("fusion_rgb_proj", "fusion_module.rgb_proj")
        elif new_k.startswith("fusion_flow_proj"):
            new_k = new_k.replace("fusion_flow_proj", "fusion_module.flow_proj")
        elif new_k.startswith("fusion_fusion_proj"):
            new_k = new_k.replace("fusion_fusion_proj", "fusion_module.fusion_proj")
        base_ckpt[new_k] = v
    
    # Load with strict=False to handle missing keys gracefully
    missing_keys, unexpected_keys = base_model.load_state_dict(base_ckpt, strict=False)
    if missing_keys:
        print(f"Warning: Missing keys in base_model checkpoint (showing first 5): {missing_keys[:5]}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in base_model checkpoint (showing first 5): {unexpected_keys[:5]}")

    psnet_ckpt = {
        k.replace("module.", ""): v for k, v in state_dict["psnet_model"].items()
    }
    psnet_model.load_state_dict(psnet_ckpt)

    decoder_ckpt1 = {
        k.replace("module.", ""): v for k, v in state_dict["decoder1"].items()
    }
    decoder[0].load_state_dict(decoder_ckpt1)
    decoder_ckpt1 = {
        k.replace("module.", ""): v for k, v in state_dict["decoder2"].items()
    }
    decoder[1].load_state_dict(decoder_ckpt1)
    decoder_ckpt1 = {
        k.replace("module.", ""): v for k, v in state_dict["decoder3"].items()
    }
    decoder[2].load_state_dict(decoder_ckpt1)
    decoder_ckpt1 = {
        k.replace("module.", ""): v for k, v in state_dict["decoder4"].items()
    }
    decoder[3].load_state_dict(decoder_ckpt1)
    regressor_delta_ckpt = {
        k.replace("module.", ""): v for k, v in state_dict["regressor_delta1"].items()
    }
    regressor_delta[0].load_state_dict(regressor_delta_ckpt)
    regressor_delta_ckpt = {
        k.replace("module.", ""): v for k, v in state_dict["regressor_delta2"].items()
    }
    regressor_delta[1].load_state_dict(regressor_delta_ckpt)
    regressor_delta_ckpt = {
        k.replace("module.", ""): v for k, v in state_dict["regressor_delta3"].items()
    }
    regressor_delta[2].load_state_dict(regressor_delta_ckpt)

    mask_encoder_ckpt = {
        k.replace("module.", ""): v for k, v in state_dict["segmenter"].items()
    }
    segmenter.load_state_dict(mask_encoder_ckpt)
   
    dim_reducer_ckpt = None
    for key in ["dim_reducer3", "dim_reducer1", "dim_reducer2"]:
        if key in state_dict:
            dim_reducer_ckpt = {
                k.replace("module.", ""): v for k, v in state_dict[key].items()
            }
            break
    
    if dim_reducer_ckpt is not None:
        # Load into shared_dim_reducer (which is referenced by all 3)
        dim_reducer1.load_state_dict(dim_reducer_ckpt)
        # dim_reducer2 and dim_reducer3 will automatically have the same weights since they share parameters

    mask_encoder_ckpt = {
        k.replace("module.", ""): v for k, v in state_dict["video_encoder"].items()
    }
    video_encoder.load_state_dict(mask_encoder_ckpt)

    temporal_causal_attn_ckpt = {
        k.replace("module.", ""): v
        for k, v in state_dict["temporal_causal_attn"].items()
    }
    missing, unexpected = temporal_causal_attn.load_state_dict(
        temporal_causal_attn_ckpt, strict=False
    )
    if missing:
        print("[checkpoint] temporal_causal_attn missing keys (new params will be randomly initialized):", missing)
    if unexpected:
        print("[checkpoint] temporal_causal_attn unexpected keys:", unexpected)

    awl_ckpt = {k.replace("module.", ""): v for k, v in state_dict["awl"].items()}
    awl.load_state_dict(awl_ckpt)

    stage_reasoner_ckpt = {
        k.replace("module.", ""): v
        for k, v in state_dict["stage_causal_reasoner"].items()
    }
    missing, unexpected = stage_causal_reasoner.load_state_dict(
        stage_reasoner_ckpt, strict=False
    )
    if missing:
        print("[resume_train] stage_causal_reasoner missing keys:", missing)
    if unexpected:
        print("[resume_train] stage_causal_reasoner unexpected keys:", unexpected)

    # optimizer
    try:
        optimizer.load_state_dict(state_dict["optimizer"])

        base_lr_scaled = args.base_lr * args.lr_factor
        base_lr_default = args.base_lr
        
        for i, param_group in enumerate(optimizer.param_groups):
            if 'lr' in param_group:
                old_lr = param_group['lr']
                # First param_group is base_model (uses lr_factor)
                if i == 0:
                    param_group['lr'] = base_lr_scaled
                    print(f"[resume_train] Updated base_model learning rate: {old_lr:.6f} -> {base_lr_scaled:.6f}")
                else:
                    # Other modules use base_lr (but may have been scaled in checkpoint)
                    # Keep base_lr for consistency, or use scaled if that was the original design
                    param_group['lr'] = base_lr_default
                    if i == 1:  # Only print once for other modules
                        print(f"[resume_train] Updated other modules learning rate: {old_lr:.6f} -> {base_lr_default:.6f}")
    except ValueError as err:
        print(
            "[resume_train] optimizer state mismatch, skipping load due to:", err
        )
        # Even if loading fails, set the learning rate from config
        base_lr_scaled = args.base_lr * args.lr_factor
        base_lr_default = args.base_lr
        for i, param_group in enumerate(optimizer.param_groups):
            if 'lr' in param_group:
                if i == 0:
                    param_group['lr'] = base_lr_scaled
                else:
                    param_group['lr'] = base_lr_default
        print(f"[resume_train] Set learning rates: base_model={base_lr_scaled:.6f}, others={base_lr_default:.6f}")

    # parameter
    start_epoch = state_dict["epoch"] + 1
    epoch_best_aqa = state_dict["epoch_best_aqa"]
    rho_best = state_dict["rho_best"]
    L2_min = state_dict["L2_min"]
    RL2_min = state_dict["RL2_min"]

    return start_epoch, epoch_best_aqa, rho_best, L2_min, RL2_min


def load_model(
    base_model,
    psnet_model,
    decoder,
    dim_reducer1,
    dim_reducer2,
    dim_reducer3,
    regressor_delta,
    segmenter,
    video_encoder,
    causal_intervention,
    temporal_causal_attn,
    awl,
    stage_causal_reasoner,
    args,
):
    ckpt_path = os.path.join(args.experiment_path, "best.pth")
    if not os.path.exists(ckpt_path):
        raise NotImplementedError("no checkpoint file from path %s..." % ckpt_path)
    print("Loading weights from %s..." % ckpt_path)

    # load state dict
    state_dict = torch.load(ckpt_path, map_location="cpu")

    # parameter resume of base model
    base_ckpt = {
        k.replace("module.", ""): v for k, v in state_dict["base_model"].items()
    }
    base_model.load_state_dict(base_ckpt)

    psnet_ckpt = {
        k.replace("module.", ""): v for k, v in state_dict["psnet_model"].items()
    }
    psnet_model.load_state_dict(psnet_ckpt)

    decoder_ckpt1 = {
        k.replace("module.", ""): v for k, v in state_dict["decoder1"].items()
    }
    decoder[0].load_state_dict(decoder_ckpt1)
    decoder_ckpt1 = {
        k.replace("module.", ""): v for k, v in state_dict["decoder2"].items()
    }
    decoder[1].load_state_dict(decoder_ckpt1)
    decoder_ckpt1 = {
        k.replace("module.", ""): v for k, v in state_dict["decoder3"].items()
    }
    decoder[2].load_state_dict(decoder_ckpt1)
    decoder_ckpt1 = {
        k.replace("module.", ""): v for k, v in state_dict["decoder4"].items()
    }
    decoder[3].load_state_dict(decoder_ckpt1)
    regressor_delta_ckpt = {
        k.replace("module.", ""): v for k, v in state_dict["regressor_delta1"].items()
    }
    regressor_delta[0].load_state_dict(regressor_delta_ckpt)
    regressor_delta_ckpt = {
        k.replace("module.", ""): v for k, v in state_dict["regressor_delta2"].items()
    }
    regressor_delta[1].load_state_dict(regressor_delta_ckpt)
    regressor_delta_ckpt = {
        k.replace("module.", ""): v for k, v in state_dict["regressor_delta3"].items()
    }
    regressor_delta[2].load_state_dict(regressor_delta_ckpt)

    mask_encoder_ckpt = {
        k.replace("module.", ""): v for k, v in state_dict["segmenter"].items()
    }
    segmenter.load_state_dict(mask_encoder_ckpt)

    dim_reducer_ckpt = None
    for key in ["dim_reducer3", "dim_reducer1", "dim_reducer2"]:
        if key in state_dict:
            dim_reducer_ckpt = {
                k.replace("module.", ""): v for k, v in state_dict[key].items()
            }
            break
    
    if dim_reducer_ckpt is not None:
        # Load into shared_dim_reducer (which is referenced by all 3)
        dim_reducer1.load_state_dict(dim_reducer_ckpt)
        # dim_reducer2 and dim_reducer3 will automatically have the same weights since they share parameters
    mask_encoder_ckpt = {
        k.replace("module.", ""): v for k, v in state_dict["video_encoder"].items()
    }
    video_encoder.load_state_dict(mask_encoder_ckpt)

    temporal_causal_attn_ckpt = {
        k.replace("module.", ""): v
        for k, v in state_dict["temporal_causal_attn"].items()
    }
    missing, unexpected = temporal_causal_attn.load_state_dict(
        temporal_causal_attn_ckpt, strict=False
    )
    if missing:
        print("[checkpoint] temporal_causal_attn missing keys (new params will be randomly initialized):", missing)
    if unexpected:
        print("[checkpoint] temporal_causal_attn unexpected keys:", unexpected)

    awl_ckpt = {k.replace("module.", ""): v for k, v in state_dict["awl"].items()}
    awl.load_state_dict(awl_ckpt)
    stage_reasoner_ckpt = {
        k.replace("module.", ""): v
        for k, v in state_dict["stage_causal_reasoner"].items()
    }
    missing, unexpected = stage_causal_reasoner.load_state_dict(
        stage_reasoner_ckpt, strict=False
    )
    if missing:
        print("[load_model] stage_causal_reasoner missing keys:", missing)
    if unexpected:
        print("[load_model] stage_causal_reasoner unexpected keys:", unexpected)

    epoch_best_aqa = state_dict["epoch_best_aqa"]
    rho_best = state_dict["rho_best"]
    L2_min = state_dict["L2_min"]
    RL2_min = state_dict["RL2_min"]
    print(
        "ckpts @ %d epoch(rho = %.4f, L2 = %.4f , RL2 = %.4f)"
        % (epoch_best_aqa - 1, rho_best, L2_min, RL2_min)
    )
    return


class DualRandomHorizontalFlip:
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return [img_.transpose(PIL.Image.FLIP_LEFT_RIGHT) for img_ in img], [
                mask_.transpose(PIL.Image.FLIP_LEFT_RIGHT) for mask_ in mask
            ]

        return img, mask


class DualCenterCrop:
    def __init__(self, size):
        self.trans = video_transforms.CenterCrop(size)

    def __call__(self, img, mask):
        cropped_img = self.trans(img)
        cropped_mask = self.trans(mask)

        return cropped_img, cropped_mask


class DualResize:
    def __init__(self, size):
        self.trans = video_transforms.Resize(size)

    def __call__(self, img, mask):
        return self.trans(img), self.trans(mask)


class DualRandomCrop:
    def __init__(self, size):
        self.size = (size, size)

    def __call__(self, img, mask):
        im_w, im_h = img[0].size
        h, w = self.size
        x1 = random.randint(0, im_w - w)
        y1 = random.randint(0, im_h - h)

        return [img_.crop((x1, y1, x1 + w, y1 + h)) for img_ in img], [
            mask_.crop((x1, y1, x1 + w, y1 + h)) for mask_ in mask
        ]


class DualNormalize:
    def __init__(self, mean, std):
        self.trans = video_transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.trans_mask = video_transforms.Normalize(mean=[0.5], std=[0.5])

    def __call__(self, img, mask):
        return self.trans(img), mask  # self.trans_mask(mask)


class DualClipToTensor:
    def __init__(self):
        self.trans = volume_transforms.ClipToTensor()
        self.trans_mask = volume_transforms.ClipToTensor(channel_nb=1)

    def __call__(self, img, mask):
        return self.trans(img), self.trans_mask(mask)


class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask
