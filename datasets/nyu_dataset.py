# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class NYUDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(NYUDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size
        self.K = np.array([[5.1885790117450188e+02, 0, 3.2558244941119034e+02, 0],
                           [0, 5.1946961112127485e+02, 2.5373616633400465e+02, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.K[0, :] /= self.width
        self.K[1, :] /= self.height
        self.archive = np.load(os.path.join(self.data_path, "nyu_archive.npy"), allow_pickle=True).item()

    def check_depth(self):
        """We don't include depth data, because we can't use the mat function to reproject depth to rgb
        """
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_image_path(self, folder, frame_index):
        image_name = self.archive[folder][frame_index][1][:-3] + 'jpg'
        image_path = os.path.join(
            self.data_path,
            folder,
            image_name)
        return image_path


class NYULabeledDataset(NYUDataset):
    def __init__(self, *args, **kwargs):
        super(NYULabeledDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        image_name = "{:3d}.jpg".format(frame_index)
        image_path = os.path.join(self.data_path, 'image', image_name)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        depth_name = "{:3d}.npy".format(frame_index)
        depth_path = os.path.join(self.data_path, 'depth', depth_name)

        depth_gt = np.load(depth_name, allow_pickle=True)

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
