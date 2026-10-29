import torch.nn as nn
import torch
import torch.nn.functional as F
from .i3d import I3D
import logging


class I3D_backbone(nn.Module):
    def __init__(self, I3D_class, use_two_stream=True, flow_ckpt_path=None):
        super(I3D_backbone, self).__init__()
        # Force RGB-only backbone; keep arguments for backward compatibility
        print("Using I3D backbone (RGB only)")
        
        self.rgb_backbone = I3D(num_classes=I3D_class, modality="rgb", dropout_prob=0.5)

    def load_pretrain(self, I3D_ckpt_path, flow_ckpt_path=None):
        # Load RGB pretrained weights
        state_dict = torch.load("./models/model_rgb.pth")
        ks = ["conv3d_0c_1x1.conv3d.weight", "conv3d_0c_1x1.conv3d.bias"]
        for k in ks:
            del state_dict[k]
        self.rgb_backbone.load_state_dict(state_dict)
        print("loading RGB ckpt done")

    def forward(self, video_1, video_2):
        """
        Args:
            video_1: [B, C, T, H, W] - RGB video 1
            video_2: [B, C, T, H, W] - RGB video 2
        Returns:
            com_feature_12: [B, T, C] - fused features
            com_feamap_12: [B, T, C, T', H', W'] - feature maps
        """
        # Process RGB stream
        total_video = torch.cat((video_1, video_2), 0)
        start_idx = list(range(0, 90, 10))
        video_pack = torch.cat([total_video[:, :, i : i + 16] for i in start_idx])
        total_feamap_rgb, total_feature_rgb = self.rgb_backbone(video_pack)
        Nt, C, T, H, W = total_feamap_rgb.size()

        total_feature_rgb = total_feature_rgb.reshape(
            len(start_idx), len(total_video), -1
        ).transpose(0, 1)  # [B, 9, 1024]
        
        total_feamap_rgb = total_feamap_rgb.reshape(
            len(start_idx), len(total_video), C, T, H, W
        ).transpose(0, 1)  # [B, 9, 1024, 2, 4, 4]
        # RGB-only features and maps
        com_feature_12 = torch.cat(
            (
                total_feature_rgb[: total_feature_rgb.shape[0] // 2],
                total_feature_rgb[total_feature_rgb.shape[0] // 2 :],
            ),
            2,
        )  # [B//2, 9, 2048]
        com_feamap_12 = torch.cat(
            (
                total_feamap_rgb[: total_feamap_rgb.shape[0] // 2],
                total_feamap_rgb[total_feamap_rgb.shape[0] // 2 :],
            ),
            2,
        )  # [B//2, 9, 2048, 2, 4, 4]
        
        return com_feature_12, com_feamap_12
