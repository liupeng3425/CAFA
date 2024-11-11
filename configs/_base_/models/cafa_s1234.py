
_base_ = ["dualbb_conv1_mitb5_depth_pretrain_inj.py"]

norm_cfg = dict(type="BN", requires_grad=True)
model = dict(
    backbone=dict(
        type="DualBackboneMix",
        used_index=[0, 1, 2, 3],
        mix_residual=True,
        backbone_depth=dict(
            _delete_=True,
            type="mit_b1",
            style="pytorch",
            pretrained="pretrained/depth_backbone.pth",
        ),
        backbone_rgb=dict(
            _delete_=True,
            type="mit_b5",
            style="pytorch",
            pretrained="pretrained/mit_b5.pth",
        ),
    ),
    decode_head=dict(
        type="DAFormerDepthMixHead",
        decoder_params=dict(
            used_index=[0, 1, 2, 3],
            mix_seg_weight=1.0,
            fusion_cfg=dict(
                _delete_=True,
                type="aspp",
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type="ReLU"),
                norm_cfg=norm_cfg,
            ),
        ),
    ),
)
