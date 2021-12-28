import os

import cv2
import torch
import numpy as np
import imageio

def embed_breakpoint(debug_info='', terminate=True):
    print('\nyou are inside a break point')
    if debug_info:
        print('debug info: {0}'.format(debug_info))
    print('')
    embedding = ('import IPython\n'
                 'import matplotlib.pyplot as plt\n'
                 'IPython.embed()\n'
                 )
    if terminate:
        embedding += (
            'assert 0, \'force termination\'\n'
        )

    return embedding


def H_to_correspondence_map(H, out_shape):
    # in_shape = in_shape[:2]
    out_shape = out_shape[:2]
    # in_h, in_w = in_shape
    out_h, out_w = out_shape
    x, y = np.meshgrid(np.linspace(0, out_w - 1, out_w),
                       np.linspace(0, out_h - 1, out_h))
    x, y = x.flatten(), y.flatten()

    xyz = np.stack([x, y, np.ones_like(x)], axis=1).T
    xyz_warped = np.matmul(H, xyz)
    xy_warped, z_warped = np.split(xyz_warped, [2], axis=0)
    xy_warped = xy_warped / z_warped
    xy_warped = xy_warped.reshape(2, *out_shape)
    xy_warped = xy_warped.transpose(1, 2, 0)
    return xy_warped


def H_to_optical_flow(H, out_shape):
    # assert in_shape == out_shape, 'optical flow requires the source and target image shares the same shape.'
    # if in_shape != out_shape:
    #     exec(embed_breakpoint())
    # in_shape = in_shape[:2]
    out_shape = out_shape[:2]
    # in_h, in_w = in_shape
    out_h, out_w = out_shape
    x, y = np.meshgrid(np.linspace(0, out_w - 1, out_w),
                       np.linspace(0, out_h - 1, out_h))
    base_grid = np.stack([x, y], axis=-1)

    corr_map = H_to_correspondence_map(H, out_shape)
    optical_flow = corr_map - base_grid
    return optical_flow


def optical_flow_to_correspondence_map(optical_flow):
    out_h, out_w = optical_flow.shape[:2]
    x, y = np.meshgrid(np.linspace(0, out_w - 1, out_w),
                       np.linspace(0, out_h - 1, out_h))
    base_grid = np.stack([x, y], axis=-1)
    return optical_flow + base_grid


def correspondence_map_to_optical_flow(corr_map):
    out_h, out_w = corr_map.shape[:2]
    x, y = np.meshgrid(np.linspace(0, out_w - 1, out_w),
                       np.linspace(0, out_h - 1, out_h))
    base_grid = np.stack([x, y], axis=-1)
    return corr_map - base_grid


def H_to_mask(H, in_shape, out_shape):
    in_shape = in_shape[:2]
    out_shape = out_shape[:2]
    in_h, in_w = in_shape
    # out_h, out_w = out_shape
    xy_warped = H_to_correspondence_map(H, out_shape)
    mask = (xy_warped[..., 0] > 0) & (xy_warped[..., 0] < in_w - 1) & (xy_warped[..., 1] > 0) & (xy_warped[..., 1] < in_h - 1)
    return mask


def warp_image(img, H, out_shape=None, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT):
    assert len(img.shape) == 3
    assert H.shape == (3, 3)
    if out_shape is None:
        out_shape = img.shape[:2]
    # in_shape = img.shape[:2]
    xy_warped = H_to_correspondence_map(H, out_shape)
    warped = cv2.remap(img, xy_warped[..., 0].astype(np.float32), xy_warped[..., 1].astype(np.float32), interpolation=interpolation, borderMode=border_mode)
    return warped


class HpatchesScene:
    def __init__(self, path, imgs, Hs):
        self.path = path
        self.imgs = imgs
        self.Hs = Hs

    def warp_a_to_b(self, from_index, to_index):
        H = self.homography(from_index, to_index)
        return warp_image(self.imgs[from_index], H, self.imgs[to_index].shape)

    def homography(self, from_index, to_index):
        '''
        return the homography that warps imgs[from_index] to imgs[to_index],
        such that the warp_image(img[from_index], H) is the same as imgs[to_index]
        '''
        if from_index == to_index:
            return np.eye(3)
        elif to_index == 0:
            return self.Hs[from_index - 1]
        elif from_index == 0:
            return np.linalg.inv(self.Hs[to_index - 1])
        else:
            return np.matmul(self.Hs[from_index - 1], np.linalg.inv(self.Hs[to_index - 1]))


def read_scene(path):
    img_path_list = ['1.ppm', '2.ppm', '3.ppm', '4.ppm', '5.ppm', '6.ppm']
    H__path_list = ['H_1_2', 'H_1_3', 'H_1_4', 'H_1_5', 'H_1_6']
    img_list = []
    H_list = []
    for img_path in img_path_list:
        img_list.append(imageio.imread(os.path.join(path, img_path)))
    for H_path in H__path_list:
        H_list.append(np.loadtxt(os.path.join(path, H_path)))
    scene = HpatchesScene(path, img_list, H_list)
    return scene


def dump_ground_truth(scene):
    '''
    skip dumping if already exists
    '''
    for to in [1, 2, 3, 4, 5]:
        corr_map_path = os.path.join(scene.path, f'gt_corr_map_1_{to+1}.npy')
        optical_flow_path = os.path.join(scene.path, f'gt_optical_flow_1_{to+1}.npy')
        mask_path = os.path.join(scene.path, f'gt_mask_1_{to+1}.npy')
        # if os.path.isfile(corr_map_path) and os.path.isfile(optical_flow_path) and os.path.isfile(mask_path):
        #     continue
        H = scene.homography(0, to)
        # exec(embed_breakpoint())
        NEED_BUG = False
        if NEED_BUG:
            h_ref_orig, w_ref_orig = scene.imgs[0].shape[:2]
            h_trg_orig, w_trg_orig = scene.imgs[to].shape[:2]
            S1 = np.array([[w_trg_orig / w_ref_orig, 0, 0],
                           [0, h_trg_orig / h_ref_orig, 0],
                           [0, 0, 1]])
            # H = np.matmul(S1, H)
            corr_map = H_to_correspondence_map(np.matmul(S1, H), scene.imgs[to].shape)
            optical_flow = H_to_optical_flow(np.matmul(S1, H), scene.imgs[to].shape)
        else:
            corr_map = H_to_correspondence_map(H, scene.imgs[to].shape)
            optical_flow = H_to_optical_flow(H, scene.imgs[to].shape)
        mask = H_to_mask(H, scene.imgs[0].shape, scene.imgs[to].shape)
        # if scene.imgs[0].shape != scene.imgs[to].shape:
        # exec(embed_breakpoint())
        np.save(corr_map_path, corr_map)
        np.save(optical_flow_path, optical_flow)
        np.save(mask_path, mask)
