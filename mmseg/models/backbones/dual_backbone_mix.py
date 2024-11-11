import torch.nn as nn
import torch
from mmseg.models.backbones.mix_transformer_mid_inject_after import AttentionCross
from mmseg.models.builder import BACKBONES
from mmcv.utils import Registry, build_from_cfg

from mmseg.models.decode_heads.segformer_head import MLP

@BACKBONES.register_module()
class DualBackboneMix(nn.Module):

    def __init__(self,
                 backbone_rgb,
                 backbone_depth,
                 used_index,
                 **cfg):
        super().__init__()
        self.backbone_rgb = build_from_cfg(backbone_rgb, BACKBONES) 
        self.backbone_depth = build_from_cfg(backbone_depth, BACKBONES)
        self.mix_residual = cfg['mix_residual']
        self.used_index = used_index
        embed_dims = self.backbone_rgb.embed_dims
        num_heads = self.backbone_rgb.num_heads

        # depth injection
        self.inj_depth_embed = {}
        self.inj_fusion_layer = {}

        for inj_index in self.used_index:
            self.inj_depth_embed[str(inj_index)] = MLP(input_dim=embed_dims[-1], embed_dim=embed_dims[inj_index])
            self.inj_fusion_layer[str(inj_index)] = AttentionCross(embed_dims[inj_index], num_heads[inj_index])
        self.inj_depth_embed = nn.ModuleDict(self.inj_depth_embed)
        self.inj_fusion_layer = nn.ModuleDict(self.inj_fusion_layer)


    def forward(self, rgb, depth, **kwargs):
        feature_rgb = self.backbone_rgb(rgb)
        feature_depth = []
        if depth is not None:
            depth = depth.squeeze()
            if len(depth.shape) == 2:
                depth = depth.unsqueeze(dim=0)
            depth = torch.stack([depth]*3, dim=1).float()
            depth = self.backbone_depth(depth)[-1]
            for inj_index in self.used_index:
                ds4 = self.inj_depth_embed[f'{inj_index}'](depth)
                f_rgb = feature_rgb[inj_index]
                f_rgb = f_rgb.flatten(2).transpose(1, 2).contiguous()
                f_ds4 = self.inj_fusion_layer[f'{inj_index}'](f_rgb, x2=ds4)
                if self.mix_residual:
                    b, c, h, w = feature_rgb[inj_index].shape
                    x = feature_rgb[inj_index] + f_ds4.transpose(1, 2).reshape((b, c, h, w)).contiguous()
                    feature_depth.append(x)
                else:
                    b, c, h, w = feature_rgb[inj_index].shape
                    f_ds4 = f_ds4.transpose(1, 2).reshape((b, c, h, w)).contiguous()
                    feature_depth.append(f_ds4)

        return feature_rgb, feature_depth

    def init_weights(self):
        self.backbone_depth.init_weights()
        self.backbone_rgb.init_weights()