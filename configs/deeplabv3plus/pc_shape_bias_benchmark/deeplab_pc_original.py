_base_ = [
    '../../../_base_/models/deeplabv3plus_r50-d8.py',
    '../../../_base_/datasets/custom_pascal_context.py', '../../../_base_/default_runtime.py',
    '../../../_base_/schedules/schedule_80k.py'
]
crop_size = (480, 480)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=33),
    auxiliary_head=dict(num_classes=33))

visualizer = dict(
    type='TestSegLocalVisualizer', vis_backends=_base_.vis_backends, name='visualizer', alpha=0.5)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=16000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='TestSegVisualizationHook', interval=200))

optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

train_dataloader = dict(
    batch_size=4,
    num_workers=4, 
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=_base_.dataset_type,
        data_root=_base_.data_root,
        data_prefix=dict(
            img_path='JPEGImages_3_split_cropped/train', seg_map_path='SemsegIDImg_3_split_cropped/train'),
        pipeline=_base_.train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=_base_.dataset_type,
        data_root=_base_.data_root,
        data_prefix=dict(
            img_path='JPEGImages_3_split_cropped/val', seg_map_path='SemsegIDImg_3_split_cropped/val'),
        pipeline=_base_.test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=_base_.dataset_type,
        data_root=_base_.data_root,
        data_prefix=dict(
            img_path='JPEGImages_3_split_cropped/test', seg_map_path='SemsegIDImg_3_split_cropped/test'),
        pipeline=_base_.test_pipeline))

work_dir = 'work_dirs/pascal_context/deeplabv3plus/trainings/deeplabv3plus_r50-d8_4xb4-80k_pascal-context_baseline-480x480_batch_4'
