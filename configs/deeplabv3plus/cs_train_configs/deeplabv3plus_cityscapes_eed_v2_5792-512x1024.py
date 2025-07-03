_base_ = [
    '../../_base_/models/deeplabv3plus_r50-d8.py',
    '../../_base_/datasets/cityscapes.py', 
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_240k.py'
]
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=_base_.dataset_type,  # Direct reference without {{}}
        data_root=_base_.data_root,  # Direct reference without {{}}
        data_prefix=dict(
            img_path='EED_v2/5792/leftImg8bit/train', 
            seg_map_path='gtFine/train'),
        pipeline=_base_.train_pipeline))  # Direct reference without {{}}

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=_base_.dataset_type,  # Direct reference without {{}}
        data_root=_base_.data_root,  # Direct reference without {{}}
        data_prefix=dict(
            img_path='EED_v2/5792/leftImg8bit/val', 
            seg_map_path='gtFine/val'),
        pipeline=_base_.test_pipeline))  # Direct reference without {{}}

test_dataloader = val_dataloader