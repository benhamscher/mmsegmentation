_base_ = [
    '../../_base_/models/deeplabv3plus_r50-d8.py',
    '../../_base_/datasets/custom_pascal_context.py', '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k.py'
]
# custom_imports = dict(imports=['custom_PascalContext'], allow_failed_imports=False)
# custom_imports = dict(imports=['CustomPascalContextDataset'], allow_failed_imports=False)
crop_size = (480, 480)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=33),
    auxiliary_head=dict(num_classes=33),
)

optimizer = dict(type='SGD', lr=0.012, momentum=0.9, weight_decay=0.0001) # changed lr from 0.004 to 0.008
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=16000)

train_dataloader = dict(
    batch_size=8,
    num_workers=4, 
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=_base_.dataset_type,
        data_root=_base_.data_root,
        data_prefix=dict(
            img_path='JPEGImages_3_split_stylized_Voronoi8_stylize_prop1.0_alpha1.0_double_resize/train', seg_map_path='SemsegIDImg_3_split_cropped/train'),
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
            img_path='JPEGImages_3_split_stylized_Voronoi8_stylize_prop1.0_alpha1.0_double_resize/val', seg_map_path='SemsegIDImg_3_split_cropped/val'),
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
            img_path='JPEGImages_3_split_stylized_Voronoi8_stylize_prop1.0_alpha1.0_double_resize/test', seg_map_path='SemsegIDImg_3_split_cropped/test'),
        pipeline=_base_.test_pipeline))

work_dir = './work_dirs/pascal_context/deeplabv3plus/trainings/deeplabv3plus_r50-d8_4xb4-160k_pascal-context_stylized_Voronoi8_double_resize-480x480_batch_8'