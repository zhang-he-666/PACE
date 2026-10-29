import os
import numpy as np
import torch
import torch.nn as nn
from datetime import timedelta

from scipy import stats
from tools import builder, helper
from utils import misc
import time
import json

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import all_reduce, ReduceOp
import wandb
import segmentation_models_pytorch as smp


def setup(rank, world_size, args):
    requested_backend = getattr(args, "dist_backend", None)
    backend_candidates = []
    if requested_backend:
        backend_candidates.append(requested_backend.lower())
        if requested_backend.lower() != "gloo":
            backend_candidates.append("gloo")
    else:
        if torch.cuda.is_available():
            backend_candidates.append("nccl")
        backend_candidates.append("gloo")
    
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    init_error = None
    for backend in backend_candidates:
        try:
            if backend == "nccl" and torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                rank=rank,
                world_size=world_size,
                timeout=timedelta(hours=2),
            )
            args.actual_dist_backend = backend
            if backend != "nccl" and torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            if rank == 0:
                print(f"[Dist] Initialized process group with backend '{backend}'.")
            init_error = None
            break
        except Exception as err:
            init_error = err
            if rank == 0:
                print(f"[WARNING] Failed to initialize backend '{backend}': {err}")
            if dist.is_initialized():
                dist.destroy_process_group()
    if init_error is not None:
        raise init_error
    
    if rank == 0 and args.wandb:
        wandb.init(project="FineParser")


def freeze_params(model):
    if model is None:
        return
    if not hasattr(model, 'parameters'):
        raise ValueError(f"freeze_params: model must have 'parameters' method, but got type {type(model)}")
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_params(model):
    """Unfreeze model parameters"""
    if model is None:
        return
    if not hasattr(model, 'parameters'):
        raise ValueError(f"unfreeze_params: model must have 'parameters' method, but got type {type(model)}")
    for param in model.parameters():
        param.requires_grad = True


def has_trainable_params(model):
    """Check if model has any trainable parameters"""
    return any(p.requires_grad for p in model.parameters())


def apply_two_stage_training(epoch, args, base_model, psnet_model, segmenter, 
                              stage_causal_reasoner, causal_intervention,  # Can be None 
                              temporal_causal_attn, decoder, regressor_delta,
                              dim_reducer1, dim_reducer2, dim_reducer3, 
                              video_encoder, awl, rank, boundary_regression_head=None):

    # CRITICAL: Verify temporal_causal_attn is callable at function start
    if not callable(temporal_causal_attn):
        raise ValueError(
            f"CRITICAL BUG: temporal_causal_attn is not callable in apply_two_stage_training! "
            f"temporal_causal_attn type: {type(temporal_causal_attn)}, value: {temporal_causal_attn}"
        )
    
    stage1_epochs = getattr(args, 'stage1_epochs', 50)  # Default 50
    
    if epoch < stage1_epochs:
        # ========== STAGE 1: Train only segmenter ==========
        if epoch == 0 and rank == 0:
            print(f"\n{'='*60}")
            print(f"TRAINING STAGE 1 (Epochs 0-{stage1_epochs-1})")
            print(f"  - Training: Segmenter (PSNet/Stage Probability Heads)")
            print(f"  - Frozen: Base model, Scoring modules, GAT modules")
            print(f"{'='*60}\n")
        
        # Freeze everything except segmenter
        freeze_params(base_model)
        unfreeze_params(psnet_model)  # This is the segmenter
        freeze_params(stage_causal_reasoner)
        if causal_intervention is not None:
            freeze_params(causal_intervention)
        freeze_params(temporal_causal_attn)
        freeze_params(video_encoder)
        
        for dec in decoder:
            freeze_params(dec)
        for reg in regressor_delta:
            freeze_params(reg)
        
        freeze_params(dim_reducer1)
        freeze_params(dim_reducer2)
        freeze_params(dim_reducer3)
        
        if awl is not None:
            freeze_params(awl)
        
        if boundary_regression_head is not None:
            freeze_params(boundary_regression_head)
        
        return True  # In stage 1
    
    elif epoch == stage1_epochs:
        # ========== TRANSITION TO STAGE 2 ==========
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"SWITCHING TO TRAINING STAGE 2 (Epochs {stage1_epochs}+)")
            print(f"  - Frozen: Segmenter (using pretrained from Stage 1)")
            print(f"  - Training: All other modules")
            print(f"{'='*60}\n")
        
        # Freeze segmenter, unfreeze everything else
        freeze_params(psnet_model)  # Freeze segmenter
        
        if not args.freeze_base:
            unfreeze_params(base_model)
        
        unfreeze_params(stage_causal_reasoner)
        if causal_intervention is not None:
            unfreeze_params(causal_intervention)
        unfreeze_params(temporal_causal_attn)
        unfreeze_params(video_encoder)
        
        for dec in decoder:
            unfreeze_params(dec)
        for reg in regressor_delta:
            unfreeze_params(reg)
        
        unfreeze_params(dim_reducer1)
        unfreeze_params(dim_reducer2)
        unfreeze_params(dim_reducer3)
        
        if awl is not None:
            unfreeze_params(awl)
        
        if boundary_regression_head is not None:
            unfreeze_params(boundary_regression_head)
        
        return False  # In stage 2
    
    else:
        # ========== STAGE 2: Continue ==========
        # CRITICAL: Verify temporal_causal_attn is still callable before returning
        if not callable(temporal_causal_attn):
            raise ValueError(
                f"CRITICAL BUG: temporal_causal_attn is not callable at end of apply_two_stage_training! "
                f"temporal_causal_attn type: {type(temporal_causal_attn)}, value: {temporal_causal_attn}"
            )
        return False  # In stage 2


def train_net(args, rank, world_size):
    setup(rank, world_size, args)
    print("Trainer start ... ")
    # build dataset
    train_dataset, test_dataset = builder.dataset_builder(args)
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.bs_train,
            shuffle=False,
            num_workers=int(args.workers),
            pin_memory=True,
            sampler=train_sampler,
            worker_init_fn=misc.worker_init_fn,
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.bs_test,
            shuffle=False,
            num_workers=int(args.workers),
            pin_memory=True,
            sampler=test_sampler,
        )
    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.bs_train,
            shuffle=True,
            num_workers=int(args.workers),
            pin_memory=True,
            worker_init_fn=misc.worker_init_fn,
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.bs_test,
            shuffle=False,
            num_workers=int(args.workers),
            pin_memory=True,
        )

    # build model
    (
        base_model,
        psnet_model,
        decoder,
        regressor_delta,
        dim_reducer1,
        dim_reducer2,
        video_encoder,
        dim_reducer3,
        segmenter,
        causal_intervention,
        temporal_causal_attn,
        awl,
        stage_segmenter,
        stage_causal_reasoner,
        boundary_regression_head,  # NEW
        boundary_regression_loss,  # NEW
    ) = builder.model_builder(args)
    
    # CRITICAL: Verify stage_segmenter, awl, and stage_causal_reasoner are correctly assigned (not swapped)
    from models.stage_modules import StageSegmenter, StageCausalReasoner
    from models.AutomaticWeightedLoss import AutomaticWeightedLoss
    if not isinstance(stage_segmenter, StageSegmenter):
        raise ValueError(
            f"CRITICAL BUG: stage_segmenter is not a StageSegmenter after model_builder! "
            f"stage_segmenter type: {type(stage_segmenter)}, "
            f"awl type: {type(awl)}, "
            f"stage_causal_reasoner type: {type(stage_causal_reasoner)}. "
            f"This indicates a variable swap bug in model_builder return statement."
        )
    if not isinstance(awl, AutomaticWeightedLoss):
        raise ValueError(
            f"CRITICAL BUG: awl is not an AutomaticWeightedLoss after model_builder! "
            f"awl type: {type(awl)}, "
            f"stage_segmenter type: {type(stage_segmenter)}, "
            f"stage_causal_reasoner type: {type(stage_causal_reasoner)}. "
            f"This indicates a variable swap bug in model_builder return statement."
        )
    if not isinstance(stage_causal_reasoner, StageCausalReasoner):
        raise ValueError(
            f"CRITICAL BUG: stage_causal_reasoner is not a StageCausalReasoner after model_builder! "
            f"stage_causal_reasoner type: {type(stage_causal_reasoner)}, "
            f"stage_segmenter type: {type(stage_segmenter)}, "
            f"awl type: {type(awl)}. "
            f"This indicates a variable swap bug in model_builder return statement."
        )
    # CRITICAL: Verify stage_causal_reasoner is not accidentally swapped with stage_segmenter
    if isinstance(stage_causal_reasoner, StageSegmenter):
        raise ValueError(
            f"CRITICAL BUG: stage_causal_reasoner is StageSegmenter after model_builder! "
            f"This indicates stage_causal_reasoner was incorrectly swapped with stage_segmenter. "
            f"stage_causal_reasoner type: {type(stage_causal_reasoner)}, "
            f"stage_segmenter type: {type(stage_segmenter)}"
        )
    if isinstance(stage_segmenter, StageCausalReasoner):
        raise ValueError(
            f"CRITICAL BUG: stage_segmenter is StageCausalReasoner after model_builder! "
            f"This indicates stage_segmenter was incorrectly swapped with stage_causal_reasoner. "
            f"stage_segmenter type: {type(stage_segmenter)}, "
            f"stage_causal_reasoner type: {type(stage_causal_reasoner)}"
        )

    # CUDA
    global use_gpu
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        base_model = base_model.cuda()
        psnet_model = psnet_model.cuda()
        decoder = [m.cuda() for m in decoder]
        regressor_delta = [m.cuda() for m in regressor_delta]
        dim_reducer1 = dim_reducer1.cuda()
        dim_reducer2 = dim_reducer2.cuda()
        video_encoder = video_encoder.cuda()
        segmenter = segmenter.cuda()
        dim_reducer3 = dim_reducer3.cuda()
        if causal_intervention is not None:
            causal_intervention = causal_intervention.cuda()
        temporal_causal_attn = temporal_causal_attn.cuda()
        awl = awl.cuda()
        stage_segmenter = stage_segmenter.cuda()  # Add CUDA transfer for stage_segmenter
        stage_causal_reasoner = stage_causal_reasoner.cuda()
        
        # CRITICAL: Verify stage_segmenter and stage_causal_reasoner are not swapped after CUDA transfer
        from models.stage_modules import StageCausalReasoner
        if isinstance(stage_causal_reasoner, StageSegmenter):
            raise ValueError(
                f"CRITICAL BUG: stage_causal_reasoner is StageSegmenter after CUDA transfer! "
                f"This indicates stage_causal_reasoner was incorrectly swapped with stage_segmenter. "
                f"stage_causal_reasoner type: {type(stage_causal_reasoner)}, "
                f"stage_segmenter type: {type(stage_segmenter)}"
            )
        if isinstance(stage_segmenter, StageCausalReasoner):
            raise ValueError(
                f"CRITICAL BUG: stage_segmenter is StageCausalReasoner after CUDA transfer! "
                f"This indicates stage_segmenter was incorrectly swapped with stage_causal_reasoner. "
                f"stage_segmenter type: {type(stage_segmenter)}, "
                f"stage_causal_reasoner type: {type(stage_causal_reasoner)}"
            )
        
        if boundary_regression_head is not None:
            boundary_regression_head = boundary_regression_head.cuda()
    
    # CRITICAL: Store original stage_causal_reasoner for recovery (outside DDP block for single GPU)
    stage_causal_reasoner_original = stage_causal_reasoner if use_gpu else None

    n_iter_per_epoch = len(train_dataloader)
    if args.freeze_base:
        freeze_params(base_model)
        freeze_params(segmenter)
    optimizer, scheduler = builder.build_opti_sche(
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
        boundary_regression_head,
    )

    start_epoch = 0
    global epoch_best_tas, pred_tious_best_5, pred_tious_best_75, epoch_best_aqa, rho_best, L2_min, RL2_min
    epoch_best_tas = 0
    pred_tious_best_5 = 0
    pred_tious_best_75 = 0
    epoch_best_aqa = 0
    rho_best = 0
    L2_min = 1000
    RL2_min = 1000
    
 
    prev_epoch_total_loss = None

    consecutive_decrease_count = 0

    global epoch_best_seg, best_iou_score, best_f1_score, best_f2_score, best_accuracy, best_recall
    epoch_best_seg = 0
    best_iou_score = 0
    best_f1_score = 0
    best_f2_score = 0
    best_accuracy = 0
    best_recall = 0

    # resume ckpts
    if args.resume or args.test:
        start_epoch, epoch_best_aqa, rho_best, L2_min, RL2_min = builder.resume_train(
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
            causal_intervention,
            temporal_causal_attn,
            awl,
            stage_causal_reasoner,
            args,
        )
        print(
            "resume ckpts @ %d epoch(rho = %.4f, L2 = %.4f , RL2 = %.4f)"
            % (start_epoch - 1, rho_best, L2_min, RL2_min)
        )

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    
    # DDP
    if world_size > 1:
        # Wrap models with DDP only if they have trainable parameters
        if has_trainable_params(base_model):
            base_model = DDP(base_model, device_ids=[local_rank])
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
        
        if has_trainable_params(psnet_model):
            psnet_model = DDP(psnet_model, device_ids=[local_rank])
            psnet_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(psnet_model)
        
        decoder = [
            DDP(_decoder, device_ids=[local_rank]) if has_trainable_params(_decoder) else _decoder
            for _decoder in decoder
        ]
        decoder = [
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(m) if isinstance(m, DDP) else m
            for m in decoder
        ]
        
        regressor_delta = [
            DDP(_regressor_delta, device_ids=[local_rank]) if has_trainable_params(_regressor_delta) else _regressor_delta
            for _regressor_delta in regressor_delta
        ]
        regressor_delta = [
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(m) if isinstance(m, DDP) else m
            for m in regressor_delta
        ]
        
        if has_trainable_params(dim_reducer1):
            dim_reducer1 = DDP(dim_reducer1, device_ids=[local_rank])
            dim_reducer1 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dim_reducer1)
        
        if has_trainable_params(dim_reducer2):
            dim_reducer2 = DDP(dim_reducer2, device_ids=[local_rank])
            dim_reducer2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dim_reducer2)
        
        if has_trainable_params(dim_reducer3):
            dim_reducer3 = DDP(dim_reducer3, device_ids=[local_rank])
            dim_reducer3 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dim_reducer3)
        
        # Wrap segmenter with DDP only if it has trainable parameters
        if has_trainable_params(segmenter):
            segmenter = DDP(segmenter, device_ids=[local_rank])
            segmenter = torch.nn.SyncBatchNorm.convert_sync_batchnorm(segmenter)
        
        if has_trainable_params(video_encoder):
            video_encoder = DDP(
                video_encoder, device_ids=[local_rank], find_unused_parameters=True
            )
            video_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(video_encoder)
        
        if causal_intervention is not None and has_trainable_params(causal_intervention):
            causal_intervention = DDP(causal_intervention, device_ids=[local_rank])
            causal_intervention = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                causal_intervention
            )
        
        if has_trainable_params(temporal_causal_attn):
            temporal_causal_attn = DDP(temporal_causal_attn, device_ids=[local_rank])
            temporal_causal_attn = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                temporal_causal_attn
            )
        
        # CRITICAL: Store original stage_segmenter BEFORE any DDP wrapping to prevent variable swap
        # This must be done before awl DDP wrapping, as stage_segmenter might be accidentally swapped
        from models.stage_modules import StageSegmenter
        from models.AutomaticWeightedLoss import AutomaticWeightedLoss
        stage_segmenter_original = stage_segmenter  # Store original to verify it doesn't change
        
        if has_trainable_params(awl):
            # Store original awl before DDP wrapping to prevent variable swap
            awl_original = awl
            awl = DDP(awl, device_ids=[local_rank])
            awl = torch.nn.SyncBatchNorm.convert_sync_batchnorm(awl)
            # CRITICAL: Verify stage_segmenter was NOT accidentally assigned to awl
            # StageSegmenter has no trainable parameters, so it should NOT be wrapped with DDP
            # Check if stage_segmenter was accidentally swapped with awl
            if stage_segmenter is not stage_segmenter_original:
                # FIX: Restore stage_segmenter from original
                print(f"[WARNING] stage_segmenter was modified during awl DDP wrapping! Restoring from original.")
                stage_segmenter = stage_segmenter_original
            if isinstance(stage_segmenter, AutomaticWeightedLoss) or \
               (hasattr(stage_segmenter, 'module') and isinstance(stage_segmenter.module, AutomaticWeightedLoss)):
                # FIX: Restore stage_segmenter from original
                print(f"[WARNING] stage_segmenter was incorrectly assigned to awl! Restoring from original.")
                stage_segmenter = stage_segmenter_original
            # Verify awl is still AutomaticWeightedLoss (wrapped in DDP)
            if hasattr(awl, 'module'):
                if not isinstance(awl.module, AutomaticWeightedLoss):
                    raise ValueError(
                        f"CRITICAL BUG: awl.module is not AutomaticWeightedLoss! "
                        f"awl type: {type(awl)}, awl.module type: {type(awl.module)}"
                    )
        if hasattr(stage_segmenter, 'module'):
            # If stage_segmenter is DDP wrapped, it should contain StageSegmenter, not AutomaticWeightedLoss
            if isinstance(stage_segmenter.module, AutomaticWeightedLoss):
                # CRITICAL BUG: stage_segmenter was incorrectly assigned to awl's DDP wrapper
                # FIX: Restore from original stage_segmenter
                print(f"[WARNING] After DDP wrapping, stage_segmenter.module is AutomaticWeightedLoss! "
                      f"Restoring from original stage_segmenter.")
                stage_segmenter = stage_segmenter_original
            else:
                # Before checking if it's StageSegmenter, check if it's StageCausalReasoner (swap bug)
                from models.stage_modules import StageCausalReasoner
                if isinstance(stage_segmenter.module, StageCausalReasoner):
                    # CRITICAL BUG: stage_segmenter.module is StageCausalReasoner!
                    # This means stage_segmenter and stage_causal_reasoner were swapped
                    print(f"[WARNING] stage_segmenter.module is StageCausalReasoner! "
                          f"This indicates stage_segmenter and stage_causal_reasoner were swapped. "
                          f"Restoring from original stage_segmenter.")
                    stage_segmenter = stage_segmenter_original
                elif not isinstance(stage_segmenter.module, StageSegmenter):
                    raise ValueError(
                        f"CRITICAL BUG: stage_segmenter.module is not StageSegmenter! "
                        f"stage_segmenter type: {type(stage_segmenter)}, "
                        f"stage_segmenter.module type: {type(stage_segmenter.module)}"
                    )
                else:
                    # If stage_segmenter is DDP wrapped but contains StageSegmenter, unwrap it
                    # StageSegmenter should NOT be DDP wrapped (it has no trainable parameters)
                    stage_segmenter = stage_segmenter.module
        elif not isinstance(stage_segmenter, StageSegmenter):
            # If not DDP wrapped, it should be StageSegmenter directly
            # FIX: Restore from original stage_segmenter
            if isinstance(stage_segmenter_original, StageSegmenter):
                print(f"[WARNING] stage_segmenter is not StageSegmenter! Restoring from original.")
                stage_segmenter = stage_segmenter_original
            else:
                raise ValueError(
                    f"CRITICAL BUG: stage_segmenter is not StageSegmenter! "
                    f"stage_segmenter type: {type(stage_segmenter)}, "
                    f"stage_segmenter_original type: {type(stage_segmenter_original)}"
                )

        # CRITICAL: Store original stage_causal_reasoner before DDP wrapping to prevent variable swap
        from models.stage_modules import StageCausalReasoner
        # CRITICAL: Verify stage_causal_reasoner is not already swapped with stage_segmenter
        if isinstance(stage_causal_reasoner, StageSegmenter):
            raise ValueError(
                f"CRITICAL BUG: stage_causal_reasoner is StageSegmenter BEFORE DDP wrapping! "
                f"This indicates stage_causal_reasoner was already swapped with stage_segmenter. "
                f"stage_causal_reasoner type: {type(stage_causal_reasoner)}, "
                f"stage_segmenter type: {type(stage_segmenter)}"
            )
        if hasattr(stage_causal_reasoner, 'module') and isinstance(stage_causal_reasoner.module, StageSegmenter):
            raise ValueError(
                f"CRITICAL BUG: stage_causal_reasoner.module is StageSegmenter BEFORE DDP wrapping! "
                f"This indicates stage_causal_reasoner was already swapped with stage_segmenter. "
                f"stage_causal_reasoner type: {type(stage_causal_reasoner)}, "
                f"stage_causal_reasoner.module type: {type(stage_causal_reasoner.module)}"
            )
        # Update the original backup (in case it was set earlier)
        stage_causal_reasoner_original = stage_causal_reasoner
        
        if has_trainable_params(stage_causal_reasoner):

            stage_causal_reasoner = DDP(
                stage_causal_reasoner, 
                device_ids=[rank], 
                find_unused_parameters=True
            )
            stage_causal_reasoner = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                stage_causal_reasoner
            )
            
            # CRITICAL: Verify stage_causal_reasoner was NOT accidentally swapped with stage_segmenter
            if hasattr(stage_causal_reasoner, 'module'):
                if isinstance(stage_causal_reasoner.module, StageSegmenter):
                    # FIX: Restore from original
                    print(f"[WARNING] stage_causal_reasoner.module is StageSegmenter! "
                          f"Restoring from original stage_causal_reasoner.")
                    stage_causal_reasoner = stage_causal_reasoner_original
                elif not isinstance(stage_causal_reasoner.module, StageCausalReasoner):
                    raise ValueError(
                        f"CRITICAL BUG: stage_causal_reasoner.module is not StageCausalReasoner! "
                        f"stage_causal_reasoner type: {type(stage_causal_reasoner)}, "
                        f"stage_causal_reasoner.module type: {type(stage_causal_reasoner.module)}"
                    )
        elif isinstance(stage_causal_reasoner, StageSegmenter):
            # FIX: Restore from original
            print(f"[WARNING] stage_causal_reasoner is StageSegmenter! Restoring from original.")
            stage_causal_reasoner = stage_causal_reasoner_original
        elif not isinstance(stage_causal_reasoner, StageCausalReasoner):
            raise ValueError(
                f"CRITICAL BUG: stage_causal_reasoner is not StageCausalReasoner! "
                f"stage_causal_reasoner type: {type(stage_causal_reasoner)}"
            )
        
        if boundary_regression_head is not None and has_trainable_params(boundary_regression_head):
            boundary_regression_head = DDP(boundary_regression_head, device_ids=[rank])
            boundary_regression_head = torch.nn.SyncBatchNorm.convert_sync_batchnorm(boundary_regression_head)

    # loss
    mse = nn.MSELoss().cuda()
    bce = nn.BCELoss().cuda()
    focal_loss = smp.losses.FocalLoss("binary").cuda()

    if args.test:
        torch.distributed.barrier()
        validate(
            base_model,
            psnet_model,
            decoder,
            regressor_delta,
            video_encoder,
            dim_reducer3,
            segmenter,
            dim_reducer1,
            dim_reducer2,
            causal_intervention,
            temporal_causal_attn,
            awl,
            stage_segmenter,
            stage_causal_reasoner,
            test_dataloader,
            -1,
            optimizer,
            args,
            rank,
            boundary_regression_head,  # NEW
        )
        exit(0)

    # training phase
    for epoch in range(start_epoch, args.max_epoch):
        # ========== TWO-STAGE TRAINING STRATEGY ==========
        if not callable(temporal_causal_attn):
            raise ValueError(
                f"CRITICAL BUG: temporal_causal_attn is not callable BEFORE apply_two_stage_training! "
                f"temporal_causal_attn type: {type(temporal_causal_attn)}, value: {temporal_causal_attn}"
            )
        
        in_stage1 = apply_two_stage_training(
            epoch, args, base_model, psnet_model, segmenter,
            stage_causal_reasoner, causal_intervention, temporal_causal_attn,
            decoder, regressor_delta, dim_reducer1, dim_reducer2, dim_reducer3,
            video_encoder, awl, rank, boundary_regression_head
        )
        
        # CRITICAL: Verify temporal_causal_attn after apply_two_stage_training
        if not callable(temporal_causal_attn):
            raise ValueError(
                f"CRITICAL BUG: temporal_causal_attn is not callable AFTER apply_two_stage_training! "
                f"temporal_causal_attn type: {type(temporal_causal_attn)}, value: {temporal_causal_attn}, "
                f"in_stage1: {in_stage1}"
            )
        
        if world_size > 1:
            torch.distributed.barrier()
            train_sampler.set_epoch(epoch)
        pred_tious_5 = []
        pred_tious_75 = []
        true_scores = []
        pred_scores = []

        segment_metrics = {
            "iou_scores": [],
            "f1_scores": [],
            "f2_scores": [],
            "accuracy": [],
            "recall": [],
        }

        base_model.train()
        psnet_model.train()
        for _decoder in decoder:
            _decoder.train()
        for _regressor_delta in regressor_delta:
            _regressor_delta.train()
        dim_reducer3.train()
        dim_reducer1.train()
        dim_reducer2.train()
        segmenter.train()
        video_encoder.train()
        if causal_intervention is not None:
            causal_intervention.train()
        # CRITICAL: Verify temporal_causal_attn is a callable model, not a bool
        if not callable(temporal_causal_attn):
            raise ValueError(
                f"CRITICAL BUG: temporal_causal_attn is not callable! "
                f"temporal_causal_attn type: {type(temporal_causal_attn)}, "
                f"temporal_causal_attn value: {temporal_causal_attn}"
            )
        temporal_causal_attn.train()
        awl.train()
        # CRITICAL: Verify stage_causal_reasoner before calling train()
        from models.stage_modules import StageCausalReasoner, StageSegmenter
        if isinstance(stage_causal_reasoner, StageSegmenter):
            # If stage_causal_reasoner is StageSegmenter, try to recover
            if hasattr(stage_causal_reasoner, 'module') and isinstance(stage_causal_reasoner.module, StageCausalReasoner):
                # Unwrap DDP if needed
                stage_causal_reasoner = stage_causal_reasoner.module
            elif stage_causal_reasoner_original is not None and isinstance(stage_causal_reasoner_original, StageCausalReasoner):
                print(f"[WARNING] stage_causal_reasoner is StageSegmenter before training! Restoring from original.")
                stage_causal_reasoner = stage_causal_reasoner_original
            else:
                raise ValueError(
                    f"CRITICAL BUG: stage_causal_reasoner is StageSegmenter and cannot be recovered! "
                    f"stage_causal_reasoner type: {type(stage_causal_reasoner)}, "
                    f"stage_segmenter type: {type(stage_segmenter)}"
                )
        stage_causal_reasoner.train()

        if args.fix_bn:
            base_model.apply(misc.fix_bn)

        total_loss = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Added 6th element for causal_reg loss
        for idx, (data, target) in enumerate(train_dataloader):
            # num_iter += 1
            opti_flag = True

            # video_1 is query and video_2 is exemplar
            video_1 = data["video"].float().cuda()
            video_2 = target["video"].float().cuda()
            video_1_mask = data["video_mask"].float().cuda()
            video_2_mask = target["video_mask"].float().cuda()
            
            label_1_tas = data["transits"].float().cuda() + 1
            label_2_tas = target["transits"].float().cuda() + 1
            if args.DD_choosing:
                label_1_score = data["completeness"].float().reshape(-1, 1).cuda()
                label_2_score = target["completeness"].float().reshape(-1, 1).cuda()
            else:
                label_1_score = data["final_score"].float().reshape(-1, 1).cuda()
                label_2_score = target["final_score"].float().reshape(-1, 1).cuda()

            difficulty = data["difficulty"].float().cuda()
            use_difficulty = args.DD_choosing
            # forward
            res = helper.network_forward_train(
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
                idx + 1,
                len(train_dataloader),
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
                len(train_dataloader),
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
                boundary_regression_head,
                boundary_regression_loss,
            )

            for t in range(len(res)):
                total_loss[t] += res[t]

            true_scores.extend(data["final_score"].cpu().numpy())

        t_loss = []
        for l in total_loss:
            loss_tensor = torch.tensor(l, device='cuda')
            dist.all_reduce(loss_tensor, op=ReduceOp.SUM)
            t_loss.append(loss_tensor)


        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)

        min_len_local = min(len(pred_scores), len(true_scores))
        pred_scores = pred_scores[:min_len_local]
        true_scores = true_scores[:min_len_local]

        pred_scores_tensor = torch.tensor(pred_scores, device='cuda')
        true_scores_tensor = torch.tensor(true_scores, device='cuda')


        pred_len_tensor = torch.tensor(len(pred_scores), dtype=torch.long, device='cuda')
        true_len_tensor = torch.tensor(len(true_scores), dtype=torch.long, device='cuda')
        
        gathered_pred_lens = [
            torch.zeros_like(pred_len_tensor) for _ in range(dist.get_world_size())
        ]
        gathered_true_lens = [
            torch.zeros_like(true_len_tensor) for _ in range(dist.get_world_size())
        ]
        
        dist.all_gather(gathered_pred_lens, pred_len_tensor)
        dist.all_gather(gathered_true_lens, true_len_tensor)
        

        min_pred_len = min([l.item() for l in gathered_pred_lens])
        min_true_len = min([l.item() for l in gathered_true_lens])
        min_len_all = min(min_pred_len, min_true_len)
        

        if len(pred_scores) > min_len_all:
            pred_scores_tensor = pred_scores_tensor[:min_len_all]
        if len(true_scores) > min_len_all:
            true_scores_tensor = true_scores_tensor[:min_len_all]

        gathered_pred_scores = [
            torch.zeros_like(pred_scores_tensor) for _ in range(dist.get_world_size())
        ]
        gathered_true_scores = [
            torch.zeros_like(true_scores_tensor) for _ in range(dist.get_world_size())
        ]

        dist.all_gather(gathered_pred_scores, pred_scores_tensor)
        dist.all_gather(gathered_true_scores, true_scores_tensor)

        sum_iou = torch.tensor(sum(segment_metrics["iou_scores"]), device='cuda')
        sum_f1 = torch.tensor(sum(segment_metrics["f1_scores"]), device='cuda')
        sum_f2 = torch.tensor(sum(segment_metrics["f2_scores"]), device='cuda')
        sum_accuracy = torch.tensor(sum(segment_metrics["accuracy"]), device='cuda')
        sum_recall = torch.tensor(sum(segment_metrics["recall"]), device='cuda')

        dist.all_reduce(sum_iou, op=ReduceOp.SUM)
        dist.all_reduce(sum_f1, op=ReduceOp.SUM)
        dist.all_reduce(sum_f2, op=ReduceOp.SUM)
        dist.all_reduce(sum_accuracy, op=ReduceOp.SUM)
        dist.all_reduce(sum_recall, op=ReduceOp.SUM)

        sum_test_iou_5 = torch.tensor(sum(pred_tious_5), device='cuda')
        sum_test_iou_75 = torch.tensor(sum(pred_tious_75), device='cuda')

        dist.all_reduce(sum_test_iou_5, op=ReduceOp.SUM)
        dist.all_reduce(sum_test_iou_75, op=ReduceOp.SUM)

        if rank == 0:
            gathered_pred_scores = torch.cat(gathered_pred_scores).cpu().numpy()
            gathered_true_scores = torch.cat(gathered_true_scores).cpu().numpy()
            rho, p = stats.spearmanr(gathered_pred_scores, gathered_true_scores)
            L2 = (
                np.power(gathered_pred_scores - gathered_true_scores, 2).sum()
                / gathered_true_scores.shape[0]
            )
            RL2 = (
                np.power(
                    (gathered_pred_scores - gathered_true_scores)
                    / (gathered_true_scores.max() - gathered_true_scores.min()),
                    2,
                ).sum()
                / gathered_true_scores.shape[0]
            )

            pred_tious_mean_5 = (
                sum_test_iou_5 / (len(train_dataloader) * args.bs_train)
            ) / dist.get_world_size()
            pred_tious_mean_75 = (
                sum_test_iou_75 / (len(train_dataloader) * args.bs_train)
            ) / dist.get_world_size()

            iou_score = (sum_iou.item() / dist.get_world_size()) / (
                len(train_dataloader) * args.bs_train
            )
            f1_score = (sum_f1.item() / dist.get_world_size()) / (
                len(train_dataloader) * args.bs_train
            )
            f2_score = (sum_f2.item() / dist.get_world_size()) / (
                len(train_dataloader) * args.bs_train
            )
            accuracy = (sum_accuracy.item() / dist.get_world_size()) / (
                len(train_dataloader) * args.bs_train
            )
            recall = (sum_recall.item() / dist.get_world_size()) / (
                len(train_dataloader) * args.bs_train
            )

            print(
                "[Training] EPOCH: %d, tIoU_5: %.4f, tIoU_75: %.4f, Loss_aqa: %.4f, Loss_tas: %.4f, Loss_mask: %.4f, Loss_stage: %.4f, Loss_global: %.4f, Loss_causal_reg: %.4f"
                % (
                    epoch,
                    pred_tious_mean_5,
                    pred_tious_mean_75,
                    t_loss[0],
                    t_loss[1],
                    t_loss[2],
                    t_loss[3],
                    t_loss[4],
                    t_loss[5],
                )
            )
            print(
                "[Training] EPOCH: %d, correlation: %.4f, L2: %.4f, RL2: %.4f, lr1: %.4f, lr2: %.4f"
                % (
                    epoch,
                    rho,
                    L2,
                    RL2,
                    optimizer.param_groups[0]["lr"],
                    optimizer.param_groups[1]["lr"],
                )
            )
            print(
                "[Training] EPOCH: %d, seg_iou: %.4f, seg_f1: %.4f, seg_f2: %.4f, seg_acc: %.4f, seg_rec: %.4f"
                % (epoch, iou_score, f1_score, f2_score, accuracy, recall)
            )

            if args.wandb:
                wandb.log(
                    {
                        "train/tIoU_5": pred_tious_mean_5,
                        "train/tIoU_75": pred_tious_mean_75,
                        "train/Epoch_Loss_aqa": t_loss[0],
                        "train/Epoch_Loss_tas": t_loss[1],
                        "train/Epoch_Loss_mask": t_loss[2],
                        "train/Epoch_Loss_stage": t_loss[3],
                        "train/Epoch_Loss_global": t_loss[4],
                        "train/Epoch_Loss_causal_reg": t_loss[5],
                        "train/correlation": rho,
                        "train/L2": L2,
                        "train/RL2": RL2,
                        "train/seg_iou": iou_score,
                        "train/seg_f1": f1_score,
                        "train/seg_f2": f2_score,
                        "train/seg_acc": accuracy,
                        "train/seg_rec": recall,
                    }
                )
        

        current_total_loss = sum([loss.item() if isinstance(loss, torch.Tensor) else loss for loss in t_loss])
        

        should_test = False
        
        if epoch < 200:

            if (epoch + 1) % 10 == 0:
                should_test = True
                if rank == 0:
                    print(f"[Training] Epoch {epoch+1}: 触发定期测试（每10个epoch）")
        else:
            if (epoch + 1) % 5 == 0:
                should_test = True
                if rank == 0:
                    print(f"[Training] Epoch {epoch+1}: 触发定期测试（每5个epoch）")
        

        prev_epoch_total_loss = current_total_loss
        

        if world_size > 1:
            torch.distributed.barrier()
        

        if should_test:
            validate(
                base_model,
                psnet_model,
                decoder,
                regressor_delta,
                video_encoder,
                dim_reducer3,
                segmenter,
                dim_reducer1,
                dim_reducer2,
                causal_intervention,
                temporal_causal_attn,
                awl,
                stage_segmenter,
                stage_causal_reasoner,
                test_dataloader,
                epoch,
                optimizer,
                args,
                rank,
                boundary_regression_head,  # NEW
            )
            if rank == 0:
                print(
                    "[TEST] EPOCH: %d, best correlation: %.6f, best L2: %.6f, best RL2: %.6f"
                    % (epoch_best_aqa, rho_best, L2_min, RL2_min)
                )
                print(
                    "[TEST] EPOCH: %d, best tIoU_5: %.6f, best tIoU_75: %.6f"
                    % (epoch_best_tas, pred_tious_best_5, pred_tious_best_75)
                )
                print(
                    "[TEST] EPOCH: %d, best seg_iou: %.6f, best seg_f1: %.6f,  best seg_f2: %.6f,  best seg_acc: %.6f,  best seg_rec: %.6f"
                    % (
                        epoch_best_seg,
                        best_iou_score,
                        best_f1_score,
                        best_f2_score,
                        best_accuracy,
                        best_recall,
                    )
                )

        if rank == 0:
            helper.save_checkpoint(
                base_model,
                psnet_model,
                decoder,
                regressor_delta,
                video_encoder,
                dim_reducer3,
                segmenter,
                dim_reducer1,
                dim_reducer2,
                # causal_intervention,  # REMOVED: Redundant with StageCausalReasoner
                temporal_causal_attn,
                awl,
                stage_causal_reasoner,
                optimizer,
                epoch,
                epoch_best_aqa,
                rho_best,
                L2_min,
                RL2_min,
                "last",
                args,
                boundary_regression_head,  # NEW
            )

        # scheduler lr
        if scheduler is not None:
            scheduler.step(epoch)


def validate(
    base_model,
    psnet_model,
    decoder,
    regressor_delta,
    video_encoder,
    dim_reducer3,
    segmenter,
    dim_reducer1,
    dim_reducer2,
    causal_intervention,
    temporal_causal_attn,
    awl,
    stage_segmenter,
    stage_causal_reasoner,
    test_dataloader,
    epoch,
    optimizer,
    args,
    rank,
    boundary_regression_head=None,  # NEW
):

    if rank == 0:
        print("Start validating epoch {}".format(epoch))
    global use_gpu
    global epoch_best_aqa, rho_best, L2_min, RL2_min, epoch_best_tas, pred_tious_best_5, pred_tious_best_75
    global epoch_best_seg, best_iou_score, best_f1_score, best_f2_score, best_accuracy, best_recall

    true_scores = []
    pred_scores = []
    pred_tious_test_5 = []
    pred_tious_test_75 = []

    segment_metrics = {
        "iou_scores": [],
        "f1_scores": [],
        "f2_scores": [],
        "accuracy": [],
        "recall": [],
    }

    base_model.eval()
    psnet_model.eval()
    for _decoder in decoder:
        _decoder.eval()
    for _regressor_delta in regressor_delta:
        _regressor_delta.eval()
    video_encoder.eval()
    dim_reducer3.eval()
    dim_reducer1.eval()
    dim_reducer2.eval()
    segmenter.eval()
    if causal_intervention is not None:
            causal_intervention.eval()
    temporal_causal_attn.eval()
    awl.eval()
    stage_causal_reasoner.eval()

    mse = nn.MSELoss().cuda()
    bce = nn.BCELoss().cuda()
    focal_loss = smp.losses.FocalLoss("binary").cuda()

    batch_num = len(test_dataloader)
    total_loss = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Added 6th element for causal_reg loss
    with torch.no_grad():
        datatime_start = time.time()

        for batch_idx, (data, target) in enumerate(test_dataloader, 0):
            datatime = time.time() - datatime_start
            start = time.time()

            video_1 = data["video"].float().cuda()
            video_1_mask = data["video_mask"].float().cuda()
            video_2_mask_list = [item["video_mask"].float().cuda() for item in target]
            video_2_list = [item["video"].float().cuda() for item in target]
            
            label_1_tas = data["transits"].float().cuda() + 1
            label_2_tas_list = [item["transits"].float().cuda() + 1 for item in target]

            if args.DD_choosing:
                label_1_score = data["completeness"].float().cuda()
                label_2_score_list = [
                    item["completeness"].float().reshape(-1, 1).cuda()
                    for item in target
                ]
            else:
                label_2_score_list = [
                    item["final_score"].float().reshape(-1, 1).cuda() for item in target
                ]
                label_1_score = data["final_score"].float().cuda()

            difficulty = data["difficulty"].float().cuda()
            use_difficulty = args.DD_choosing

            res = helper.network_forward_test(
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
                boundary_regression_head=boundary_regression_head,
            )

            for t in range(len(res)):
                total_loss[t] += res[t]

            batch_time = time.time() - start
            if rank == 0 and batch_idx % args.print_freq == 0:
                print(
                    "[TEST][%d/%d][%d/%d] \t Batch_time %.2f \t Data_time %.2f"
                    % (
                        epoch,
                        args.max_epoch,
                        batch_idx,
                        batch_num,
                        batch_time,
                        datatime,
                    )
                )
            datatime_start = time.time()

            true_scores.extend(data["final_score"].cpu().numpy())

        # evaluation results
        t_loss = []
        for l in total_loss:
            loss_tensor = torch.tensor(l, device='cuda')
            dist.all_reduce(loss_tensor, op=ReduceOp.SUM)
            t_loss.append(loss_tensor)


        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)


        min_len_local = min(len(pred_scores), len(true_scores))
        pred_scores = pred_scores[:min_len_local]
        true_scores = true_scores[:min_len_local]

        pred_scores_tensor = torch.tensor(pred_scores, device='cuda')
        true_scores_tensor = torch.tensor(true_scores, device='cuda')


        pred_len_tensor = torch.tensor(len(pred_scores), dtype=torch.long, device='cuda')
        true_len_tensor = torch.tensor(len(true_scores), dtype=torch.long, device='cuda')
        
        gathered_pred_lens = [
            torch.zeros_like(pred_len_tensor) for _ in range(dist.get_world_size())
        ]
        gathered_true_lens = [
            torch.zeros_like(true_len_tensor) for _ in range(dist.get_world_size())
        ]
        
        dist.all_gather(gathered_pred_lens, pred_len_tensor)
        dist.all_gather(gathered_true_lens, true_len_tensor)
        
 
        min_pred_len = min([l.item() for l in gathered_pred_lens])
        min_true_len = min([l.item() for l in gathered_true_lens])
        min_len_all = min(min_pred_len, min_true_len)

        if len(pred_scores) > min_len_all:
            pred_scores_tensor = pred_scores_tensor[:min_len_all]
        if len(true_scores) > min_len_all:
            true_scores_tensor = true_scores_tensor[:min_len_all]

        gathered_pred_scores = [
            torch.zeros_like(pred_scores_tensor) for _ in range(dist.get_world_size())
        ]
        gathered_true_scores = [
            torch.zeros_like(true_scores_tensor) for _ in range(dist.get_world_size())
        ]

        dist.all_gather(gathered_pred_scores, pred_scores_tensor)
        dist.all_gather(gathered_true_scores, true_scores_tensor)

        sum_iou = torch.tensor(sum(segment_metrics["iou_scores"]), device='cuda')
        sum_f1 = torch.tensor(sum(segment_metrics["f1_scores"]), device='cuda')
        sum_f2 = torch.tensor(sum(segment_metrics["f2_scores"]), device='cuda')
        sum_accuracy = torch.tensor(sum(segment_metrics["accuracy"]), device='cuda')
        sum_recall = torch.tensor(sum(segment_metrics["recall"]), device='cuda')


        dist.all_reduce(sum_iou, op=ReduceOp.SUM)
        dist.all_reduce(sum_f1, op=ReduceOp.SUM)
        dist.all_reduce(sum_f2, op=ReduceOp.SUM)
        dist.all_reduce(sum_accuracy, op=ReduceOp.SUM)
        dist.all_reduce(sum_recall, op=ReduceOp.SUM)

        sum_test_iou_5 = torch.tensor(sum(pred_tious_test_5), device='cuda')
        sum_test_iou_75 = torch.tensor(sum(pred_tious_test_75), device='cuda')

        dist.all_reduce(sum_test_iou_5, op=ReduceOp.SUM)
        dist.all_reduce(sum_test_iou_75, op=ReduceOp.SUM)

        if rank == 0:
            gathered_pred_scores = torch.cat(gathered_pred_scores).cpu().numpy()
            gathered_true_scores = torch.cat(gathered_true_scores).cpu().numpy()
            rho, p = stats.spearmanr(gathered_pred_scores, gathered_true_scores)
            L2 = (
                np.power(gathered_pred_scores - gathered_true_scores, 2).sum()
                / gathered_true_scores.shape[0]
            )
            RL2 = (
                np.power(
                    (gathered_pred_scores - gathered_true_scores)
                    / (gathered_true_scores.max() - gathered_true_scores.min()),
                    2,
                ).sum()
                / gathered_true_scores.shape[0]
            )
            pred_tious_test_mean_5 = (
                sum_test_iou_5 / (len(test_dataloader) * args.bs_test)
            ) / dist.get_world_size()
            pred_tious_test_mean_75 = (
                sum_test_iou_75 / (len(test_dataloader) * args.bs_test)
            ) / dist.get_world_size()

            if pred_tious_test_mean_5 > pred_tious_best_5:
                pred_tious_best_5 = pred_tious_test_mean_5
            if pred_tious_test_mean_75 > pred_tious_best_75:
                pred_tious_best_75 = pred_tious_test_mean_75
                epoch_best_tas = epoch

            current_iou_score = (
                (sum_iou.item() / dist.get_world_size())
                / (len(test_dataloader) * args.bs_test)
                / args.voter_number
            )
            current_f1_score = (
                (sum_f1.item() / dist.get_world_size())
                / (len(test_dataloader) * args.bs_test)
                / args.voter_number
            )
            current_f2_score = (
                (sum_f2.item() / dist.get_world_size())
                / (len(test_dataloader) * args.bs_test)
                / args.voter_number
            )
            current_accuracy = (
                (sum_accuracy.item() / dist.get_world_size())
                / (len(test_dataloader) * args.bs_test)
                / args.voter_number
            )
            current_recall = (
                (sum_recall.item() / dist.get_world_size())
                / (len(test_dataloader) * args.bs_test)
                / args.voter_number
            )

            if current_iou_score > best_iou_score:
                best_iou_score = current_iou_score
                epoch_best_seg = epoch

            if current_f1_score > best_f1_score:
                best_f1_score = current_f1_score

            if current_f2_score > best_f2_score:
                best_f2_score = current_f2_score

            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy

            if current_recall > best_recall:
                best_recall = current_recall

            if L2_min > L2:
                L2_min = L2
            if RL2_min > RL2:
                RL2_min = RL2
            if rho > rho_best:
                rho_best = rho
                epoch_best_aqa = epoch
                print("-----New best found!-----")
                helper.save_checkpoint(
                    base_model,
                    psnet_model,
                    decoder,
                    regressor_delta,
                    video_encoder,
                    dim_reducer3,
                    segmenter,
                    dim_reducer1,
                    dim_reducer2,
                    # causal_intervention,  # REMOVED: Redundant with StageCausalReasoner
                    temporal_causal_attn,
                    awl,
                    stage_causal_reasoner,
                    optimizer,
                    epoch,
                    epoch_best_aqa,
                    rho_best,
                    L2_min,
                    RL2_min,
                    "best",
                    args,
                    boundary_regression_head,  # NEW
                )
            helper.save_outputs(gathered_pred_scores, gathered_true_scores, args, epoch)
            print(
                "[TEST] EPOCH: %d, Loss_aqa: %.6f, Loss_tas: %.6f, Loss_mask: %.6f, "
                "Loss_stage: %.6f, Loss_global: %.6f, Loss_causal_reg: %.6f"
                % (epoch, t_loss[0], t_loss[1], t_loss[2], t_loss[3], t_loss[4], t_loss[5])
            )
            print(
                "[TEST] EPOCH: %d, tIoU_5: %.6f, tIoU_75: %.6f"
                % (epoch, pred_tious_best_5, pred_tious_best_75)
            )
            print(
                "[TEST] EPOCH: %d, correlation: %.6f, L2: %.6f, RL2: %.6f"
                % (epoch, rho, L2, RL2)
            )
            print(
                "[TEST] EPOCH: %d, IOU Score: %.6f, F1 Score: %.6f, F2 Score: %.6f, Accuracy: %.6f, Recall: %.6f"
                % (
                    epoch,
                    current_iou_score,
                    current_f1_score,
                    current_f2_score,
                    current_accuracy,
                    current_recall,
                )
            )

            if args.wandb:
                wandb.log(
                    {
                        "test/Loss_aqa": t_loss[0],
                        "test/Loss_tas": t_loss[1],
                        "test/Loss_mask": t_loss[2],
                        "test/Loss_stage": t_loss[3],
                        "test/Loss_global": t_loss[4],
                        "test/Loss_causal_reg": t_loss[5],
                        "test/tIoU_5": pred_tious_best_5,
                        "test/tIoU_75": pred_tious_best_75,
                        "test/correlation": rho,
                        "test/L2": L2,
                        "test/RL2": RL2,
                        "test/IOU_Score": current_iou_score,
                        "test/F1_Score": current_f1_score,
                        "test/F2_Score": current_f2_score,
                        "test/Accuracy": current_accuracy,
                        "test/Recall": current_recall,
                    }
                )
