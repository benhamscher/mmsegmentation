_base_ = [
    '../../_base_/models/deeplabv3plus_r50-d8.py',
    '../../_base_/datasets/model_validation/eed_5792_val.py', '../../_base_/default_runtime.py',
    '../../_base_/schedules/test_schedule.py'
]
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
