import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):

        # Compute BCE loss
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Compute focal weight
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal loss
        focal_loss = alpha_t * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class StageClassificationLoss(nn.Module):

    
    def __init__(
        self,
        use_focal=True,
        alpha=0.25,
        gamma=2.0,
        stage_weights=None
    ):
        super(StageClassificationLoss, self).__init__()
        self.use_focal = use_focal
        self.stage_weights = stage_weights
        
        if use_focal:
            self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        else:
            self.bce_loss = nn.BCELoss(reduction='mean')
    
    def forward(self, pred_saliencies, target_stage_labels):

        B, num_stages, _ = pred_saliencies.shape
        device = pred_saliencies.device
        
        # Ensure predictions are in [0, 1] range (should already be from sigmoid)
        pred_saliencies = torch.clamp(pred_saliencies, 0.0, 1.0)
        
        # Ensure targets are in [0, 1] range
        target_stage_labels = torch.clamp(target_stage_labels, 0.0, 1.0)
        
        # Compute loss for each stage
        stage_losses = []
        for stage_idx in range(num_stages):
            pred_stage = pred_saliencies[:, stage_idx, :]  # [B, 1]
            target_stage = target_stage_labels[:, stage_idx, :]  # [B, 1]
            
            if self.use_focal:
                stage_loss = self.focal_loss(pred_stage, target_stage)
            else:
                stage_loss = self.bce_loss(pred_stage, target_stage)
            
            stage_losses.append(stage_loss)
        
        # Weighted sum of stage losses
        if self.stage_weights is not None:
            total_loss = sum(w * loss for w, loss in zip(self.stage_weights, stage_losses))
        else:
            total_loss = sum(stage_losses) / num_stages
        
        # Create loss dictionary
        stage_names = ['forward', 'twist', 'entry']
        loss_dict = {
            f'loss_stage_{name}': loss.item() 
            for name, loss in zip(stage_names, stage_losses)
        }
        loss_dict['loss_stage_classification'] = total_loss.item()
        
        return total_loss, loss_dict


def create_stage_labels_from_saliencies(saliencies, threshold=0.5):

    binary_labels = (saliencies > threshold).float()
    return binary_labels

