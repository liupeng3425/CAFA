import torch.nn as nn
import torch
from mmseg.models.builder import BACKBONES
from mmcv.utils import Registry, build_from_cfg

@BACKBONES.register_module()
class DualBackboneInj(nn.Module):

    def __init__(self,
                 backbone_rgb,
                 backbone_depth,
                 **cfg):
        super().__init__()
        self.backbone_rgb = build_from_cfg(backbone_rgb, BACKBONES) 
        self.backbone_depth = build_from_cfg(backbone_depth, BACKBONES)

    def forward(self, rgb, depth, mode=None, **kwargs):
        wo_depth=kwargs.get('wo_depth', False) and depth is None
        if wo_depth:
            feature_rgb_wo_depth = self.backbone_rgb(rgb, None, mode=mode, wo_depth=True)
            return feature_rgb_wo_depth
        else:

            depth = depth.squeeze()
            if len(depth.shape) == 2:
                depth = depth.unsqueeze(dim=0)
            depth = torch.stack([depth]*3, dim=1).float()
            feature_depth = self.backbone_depth(depth)
            feature_rgb = self.backbone_rgb(rgb, feature_depth[-1], mode=mode)
            return feature_rgb

    def init_weights(self):
        self.backbone_depth.init_weights()
        self.backbone_rgb.init_weights()