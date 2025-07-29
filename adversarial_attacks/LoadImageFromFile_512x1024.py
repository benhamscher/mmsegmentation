
# The config file for all models is structured the same way, but it's better to open the .py file and check it
#Pipeline can vary from model to model.

#I replaced the standard method LoadImageFromFile with the method LoadImageFromFile_512x1024,
# which does the same thing but applies F.interpolate(img, size=(512, 1024), mode='bilinear', align_corners=True)
# immediately upon loading to ensure that the model does not see the original image size 
# and does not perform automatic resizing.

# This is the code to load the model and change pipelines. 

# It's quite simple to just replace the loading method тnd remove or reassign the resize method for the new resolution, 
#rather than reassigning the entire pipeline as I did.
"""

from LoadImageFromFile_512x1024 import LoadImageFromFile_512x1024

model = init_model(args.config, args.checkpoint, device=args.device)


model.cfg.test_pipeline = [{'type': 'LoadImageFromFile_512x1024'},
                            {'type': 'Resize', 'scale': (1024, 512), 
                                'keep_ratio': True}, {'type': 'PackSegInputs'}]

model.cfg.test_dataloader['dataset']["pipeline"] = [{'type': 'LoadImageFromFile_512x1024'},
                                                    {'type': 'Resize', 'scale': (1024, 512), 
                                                    'keep_ratio': True}, 
                                                    # {'type': 'LoadAnnotations'}, 
                                                    {'type': 'PackSegInputs'}]

model.cfg.val_dataloader['dataset']["pipeline"] = [{'type': 'LoadImageFromFile_512x1024'},
                                                    {'type': 'Resize', 'scale': (1024, 512), 
                                                    'keep_ratio': True}, 
                                                    # {'type': 'LoadAnnotations'}, 
                                                    {'type': 'PackSegInputs'}]

model.cfg.train_dataloader['dataset']["pipeline"] = [{'type': 'LoadImageFromFile_512x1024'},
                                                    {'type': 'Resize', 'scale': (1024, 512), 
                                                    'keep_ratio': True}, 
                                                    # {'type': 'LoadAnnotations'}, 
                                                    {'type': 'PackSegInputs'}]
model.cfg.tta_pipeline[0]['type'] = 'LoadImageFromFile_512x1024'
"""




import warnings
from typing import Optional

import torch.nn.functional as F
import torch
import mmengine.fileio as fileio
import numpy as np

import mmcv
#from .base import BaseTransform
#from .builder import TRANSFORMS


import random
import mmcv
from mmcv.transforms import BaseTransform, TRANSFORMS

@TRANSFORMS.register_module()
class LoadImageFromFile_512x1024(BaseTransform):
    """Load an image from file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :func:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                img_bytes = file_client.get(filename)
            else:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename}'
        if self.to_float32:
            img = img.astype(np.float32)

        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0) 

        #################################
        img = F.interpolate(img, size=(512, 1024), mode='bilinear', align_corners=True)
        #################################

        img = img.squeeze(0).permute(1, 2, 0).numpy()
        results['img'] = img 
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        #results['ori_shape'] = (512, 1024)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', ")

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str
class LoadImageFromFile_512x1024:
    """Add your transform

    Args:
        p (float): Probability of shifts. Default 0.5.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, results):
        if random.random() > self.p:
            results['dummy'] = True
        return results