_base_ = [
    '../../_base_/models/deeplabv3plus_r50-d8.py',
    '../../_base_/datasets/model_validation/cityscapes_val.py', '../../_base_/default_runtime.py',
    '../../_base_/schedules/test_schedule.py'
]
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)

_base_.data_root = '/home/bhamscher/datasets/Voronoi_Cityscapes_128'

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type= _base_.dataset_type,  
        data_root= '/home/bhamscher/datasets/Voronoi_Cityscapes_128', 
        data_prefix=dict(
            img_path='leftImg8bit/val', 
            seg_map_path='gtFine/val'),
        pipeline=_base_.test_pipeline)) 

test_dataloader = val_dataloader