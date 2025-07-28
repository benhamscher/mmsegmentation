_base_ = [
    '../../_base_/models/segformer_mit-b0_adv.py', 
    '../../_base_/datasets/cityscapes_1024x1024.py',
    '../../_base_/default_runtime.py', 
    '../../_base_/schedules/schedule_240k.py'
]

crop_size = (1024, 1024)
# data_preprocessor = dict(size=crop_size)
data_preprocessor = dict()

# Set checkpoint to None for training from scratch, or use MIT-B1 pretrained weights
checkpoint = None  # Use None to train from scratch
# checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b1_20220624-02e5a6a1.pth'  # noqa

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        # Change from MIT-B0 to MIT-B1 architecture
        embed_dims=64,  # B1: 64, B0: 32
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint) if checkpoint else None
    ),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],  # B1 channels, B0: [32, 64, 160, 256]
        num_classes=19  # Cityscapes has 19 classes, ADE20K has 150
    ),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768))
)

# Override optimizer configuration (same as your original 240k config)
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

# Override learning rate schedule for 240k iterations
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=240000,  # Changed from 160k to 240k
        by_epoch=False,
    )
]

# Override training configuration for 240k iterations 
train_cfg = dict(
    type='IterBasedTrainLoop', 
    max_iters=240000,  # Changed from 160k to 240k
    val_interval=16000
)

# Override data root to your custom path
data_root = '/home/maag/datasets/Cityscapes/'

# Override dataloaders with your batch size and data root
train_dataloader = dict(
    batch_size=2, 
    num_workers=4,
    dataset=dict(data_root=data_root)
)
val_dataloader = dict(
    batch_size=1, 
    num_workers=4,
    dataset=dict(data_root=data_root)
)
test_dataloader = val_dataloader

# Custom work directory
work_dir = 'work_dirs/segformer/trainings/segformer_mit-b1_8xb2-240k_cityscapes-1024x1024_batch_size_2'