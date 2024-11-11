_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/cafa_s1234.py',
    # SYN->Cityscapes Data Loading
    '../_base_/datasets/uda_depth_synthia_to_cityscapes_512x512.py',
    # Basic UDA Self-Training
    '../_base_/uda/dacs_dual.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 3
mix_index = [0,1,2,3]
model = dict(
    backbone=dict(
        used_index=mix_index,
        mix_residual=True),
    decode_head=dict(
        decoder_params=dict(
            used_index=mix_index,
            mix_seg_weight=0.0001)
    )
)
# Modifications to Basic UDA
uda = dict(
    # Increased Alpha
    alpha=0.999,
    # Thing-Class Feature Distance
    imnet_feature_dist_lambda=0, #0.005,
    imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
    imnet_feature_dist_scale_min_ratio=0.75,
    # Pseudo-Label Crop
    pseudo_weight_ignore_top=15,
    pseudo_weight_ignore_bottom=120)
data = dict(
    train=dict(
        # Rare Class Sampling
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5)))
# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone_depth=dict(lr_mult=1.0),
            inj_depth_embed=dict(lr_mult=5.0),
            inj_fusion_layer=dict(lr_mult=5.0),
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=80000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=1)
evaluation = dict(interval=2000, metric='mIoU', save_best='mIoU')
# Meta Information for Result Analysis
name = 'syn2cs_uda_dualbb_cafa_0_0001_s1234'
exp = 'basic'
name_dataset = 'syn2cityscapes'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'dacs_a999_fd_things_rcs0.01_cpl'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'

