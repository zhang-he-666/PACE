
import torch
import torch.nn as nn
import torch.nn.functional as F


def generalized_iou_loss(pred_boxes, target_boxes, reduction='mean'):
    """
    Compute Generalized IoU (GIoU) loss for temporal action localization.
    
    Args:
        pred_boxes: [B, 2] - predicted [t_start, t_end]
        target_boxes: [B, 2] - target [t_start, t_end]
        reduction: 'mean' or 'sum'
    Returns:
        giou_loss: scalar tensor
    """
    # Ensure boxes are valid (start < end)
    pred_start = torch.clamp(pred_boxes[:, 0], min=0.0)
    pred_end = torch.clamp(pred_boxes[:, 1], min=pred_start + 1e-6)
    target_start = torch.clamp(target_boxes[:, 0], min=0.0)
    target_end = torch.clamp(target_boxes[:, 1], min=target_start + 1e-6)
    
    # Compute intersection
    inter_start = torch.max(pred_start, target_start)
    inter_end = torch.min(pred_end, target_end)
    inter = torch.clamp(inter_end - inter_start, min=0.0)
    
    # Compute union
    pred_length = pred_end - pred_start
    target_length = target_end - target_start
    union = pred_length + target_length - inter
    
    # Compute IoU
    iou = inter / (union + 1e-8)
    
    # Compute enclosing box
    enclose_start = torch.min(pred_start, target_start)
    enclose_end = torch.max(pred_end, target_end)
    enclose_length = enclose_end - enclose_start
    
    # Compute GIoU
    giou = iou - (enclose_length - union) / (enclose_length + 1e-8)
    
    # GIoU loss = 1 - GIoU
    giou_loss = 1.0 - giou
    
    if reduction == 'mean':
        return giou_loss.mean()
    elif reduction == 'sum':
        return giou_loss.sum()
    else:
        return giou_loss


class BoundaryRegressionHead(nn.Module):
    def __init__(
        self,
        in_dim=64,
        hidden_dim=256,
        num_categories=1,  # Set to 1 if no category prediction
        use_category=True
    ):
        super().__init__()
        self.use_category = use_category
        self.num_categories = num_categories
        
        # Shared feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Boundary regression (t_start, t_end)
        self.boundary_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 2),  # [t_start, t_end]
            nn.Sigmoid()  # Normalize to [0, 1]
        )
        
        # Completeness score regression
        self.completeness_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Normalize to [0, 1]
        )
        
        # Category classification (optional)
        if use_category and num_categories > 1:
            self.category_head = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, num_categories),
                nn.Softmax(dim=-1)
            )
    
    def forward(self, fused_features, temporal_length=None):
    
        if fused_features.dim() == 3:
            # [B, T, in_dim] -> [B, in_dim] (temporal pooling)
            fused_features = fused_features.mean(dim=1)
        
        features = self.feature_extractor(fused_features)  # [B, hidden_dim // 2]
        

        boundaries = self.boundary_head(features)  # [B, 2]
        

        completeness = self.completeness_head(features)  # [B, 1]
        
        if self.use_category and self.num_categories > 1:
            category = self.category_head(features)  # [B, num_categories]
        else:
            category = None
        
        return boundaries, completeness, category
    
    def denormalize_boundaries(self, boundaries, temporal_length):

        return boundaries * temporal_length


class BoundaryRegressionLoss(nn.Module):
    
    def __init__(
        self,
        l1_weight=1.0,
        giou_weight=1.0,
        completeness_weight=0.5,
        category_weight=0.3,
        use_category=True
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.giou_weight = giou_weight
        self.completeness_weight = completeness_weight
        self.category_weight = category_weight
        self.use_category = use_category
        
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')
        if use_category:
            self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
    
    def forward(
        self,
        pred_boundaries,
        pred_completeness,
        pred_category,
        target_boundaries,
        target_completeness,
        target_category=None,
        temporal_length=None
    ):

        # L1 loss for boundaries
        l1_loss = self.l1_loss(pred_boundaries, target_boundaries)
        
        # GIoU loss for boundaries
        giou_loss = generalized_iou_loss(pred_boundaries, target_boundaries)
        
        # Completeness loss
        completeness_loss = self.mse_loss(pred_completeness, target_completeness)
        
        # Category loss (if applicable)
        category_loss = torch.tensor(0.0, device=pred_boundaries.device)
        if self.use_category and pred_category is not None and target_category is not None:
            category_loss = self.ce_loss(pred_category, target_category)
        
        # Total loss
        total_loss = (
            self.l1_weight * l1_loss +
            self.giou_weight * giou_loss +
            self.completeness_weight * completeness_loss +
            (self.category_weight * category_loss if self.use_category else 0.0)
        )
        
        loss_dict = {
            'l1_loss': l1_loss.item(),
            'giou_loss': giou_loss.item(),
            'completeness_loss': completeness_loss.item(),
            'category_loss': category_loss.item() if self.use_category else 0.0,
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict

