# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import os.path as osp

import scipy.io as io
import mmcv
import numpy as np
import cv2
from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # modify if custom classes
        if results.get('label_map', None) is not None:
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadDepth(object):
    """Load depth for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 depth_file_type='.mat',
                 file_client_args=dict(backend='disk'),
                 dataset_name='cityscapes'):
        self.dataset_name = dataset_name
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.depth_file_type = depth_file_type

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded depth.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('depth_prefix', None) is not None:
            filename = osp.join(results['depth_prefix'],
                                results['depth_info']['depth'])
        else:
            filename = results['depth_info']['depth']

        if self.dataset_name=='cityscapes' and self.depth_file_type=='.mat' and filename.endswith('depth_stereoscopic.mat'):
            depth = io.loadmat(filename)["depth_map"]  
            depth = np.clip(depth, 0., 655.35)
            depth[depth<0.1] = 655.35
            depth = 655.36 / (depth + 0.01)
        elif self.dataset_name=='synthia' and self.depth_file_type=='.png' and filename.endswith('.png'):
            depth = cv2.imread(str(filename), flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)/100.
            depth = 655.36 / (depth + 0.01)
        elif self.dataset_name=='gta' and self.depth_file_type=='.png' and filename.endswith('.png'):
            depth = cv2.imread(str(filename), flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)/256.  + 1.
        else:
            raise NotImplementedError(f'{self.dataset_name} depth format {self.depth_file_type} not implemented.')
        results['gt_depth_map'] = depth
        results['seg_fields'].append('gt_depth_map')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(dataset_name='{self.dataset_name}',"
        repr_str += f"depth_file_type='{self.depth_file_type}')"
        return repr_str



@PIPELINES.register_module()
class LoadDepthAsInput(object):
    """Load depth for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 depth_file_type='.mat',
                 file_client_args=dict(backend='disk'),
                 dataset_name='cityscapes'):
        self.dataset_name = dataset_name
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.depth_file_type = depth_file_type

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded depth.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('depth_prefix', None) is not None:
            filename = osp.join(results['depth_prefix'],
                                results['depth_info']['depth'])
        else:
            filename = results['depth_info']['depth']

        if self.dataset_name=='cityscapes' and self.depth_file_type=='.mat' and filename.endswith('depth_stereoscopic.mat'):
            depth = io.loadmat(filename)["depth_map"]  
            depth = np.clip(depth, 0., 655.35)
            depth[depth<0.1] = 655.35
            depth = 655.36 / (depth + 0.01)
        elif self.dataset_name=='synthia' and self.depth_file_type=='.png' and filename.endswith('.png'):
            depth = cv2.imread(str(filename), flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)/100.
            depth = 655.36 / (depth + 0.01)
        elif self.dataset_name=='gta' and self.depth_file_type=='.png' and filename.endswith('.png'):
            depth = cv2.imread(str(filename), flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)/256.  + 1.
        else:
            raise NotImplementedError(f'{self.dataset_name} depth format {self.depth_file_type} not implemented.')
        results['gt_depth_map'] = depth.astype(np.float32)
        # load depth as input
        results['img'] = cv2.merge([depth, depth, depth]).astype(np.float32)
        
        results['seg_fields'].append('gt_depth_map')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(dataset_name='{self.dataset_name}',"
        repr_str += f"depth_file_type='{self.depth_file_type}')"
        return repr_str
