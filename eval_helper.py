import os
import numpy as np
import imageio
import cv2

import math
import numpy as np
import torch
import torch.nn as nn
import imageio
import vispy
import vispy.plot as vp
import vispy.io as io
from vispy import gloo
from vispy import app
from vispy.util.ptime import time
from vispy.gloo.util import _screenshot
from scipy.spatial import Delaunay
from vispy.gloo.wrappers import read_pixels

import hpatches_helper
import matplotlib.pyplot as plt
from scipy.misc import imresize

app.use_app('glfw')

def visualize_corrs(img1, img2, corrs, mask=None, save_to=None):
    if mask is None:
        mask = np.ones(len(corrs)).astype(bool)

    scale1 = 1.0
    scale2 = 1.0
    if img1.shape[1] > img2.shape[1]:
        scale2 = img1.shape[1] / img2.shape[1]
        w = img1.shape[1]
    else:
        scale1 = img2.shape[1] / img1.shape[1]
        w = img2.shape[1]
    # Resize if too big
    max_w = w
    if w > max_w:
        scale1 *= max_w / w
        scale2 *= max_w / w
    img1 = imresize(img1, scale1)
    img2 = imresize(img2, scale2)

    x1, x2 = corrs[:, :2], corrs[:, 2:]
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img = np.zeros((h1 + h2, max(w1, w2), 3), dtype=img1.dtype)
    img[:h1, :w1] = img1
    img[h1:, :w2] = img2
    # Move keypoints to coordinates to image coordinates
    x1 = x1 * scale1
    x2 = x2 * scale2
    # recompute the coordinates for the second image
    x2p = x2 + np.array([[0, h1]])
    fig = plt.figure(frameon=False)
    fig = plt.imshow(img)

    cols = [
        [0.0, 0.67, 0.0],
        [0.9, 0.1, 0.1],
    ]
    lw = 2
    alpha = 0.1

    # Draw outliers
    _x1 = x1[~mask]
    _x2p = x2p[~mask]
    xs = np.stack([_x1[:, 0], _x2p[:, 0]], axis=1).T
    ys = np.stack([_x1[:, 1], _x2p[:, 1]], axis=1).T
    plt.plot(
        xs, ys,
        alpha=alpha,
        linestyle="-",
        linewidth=lw,
        aa=False,
        color=cols[1],
    )
    # colorline(xs, ys, cmap=plt.get_cmap('jet'), linewidth=w)

    # Draw Inliers
    _x1 = x1[mask]
    _x2p = x2p[mask]
    xs = np.stack([_x1[:, 0], _x2p[:, 0]], axis=1).T
    ys = np.stack([_x1[:, 1], _x2p[:, 1]], axis=1).T
    plt.plot(
        xs, ys,
        alpha=alpha,
        linestyle="-",
        linewidth=lw,
        aa=False,
        color=cols[0],
    )
    # exec(debug_utils.embed_breakpoint())
    # for xi, yi in zip(xs.T, ys.T):
    #     path = mpath.Path(np.column_stack([xi, yi]))
    #     verts = path.interpolated(steps=3000).vertices
    #     xi, yi = verts[:, 0], verts[:, 1]
    #     z = np.linspace(0, 1, len(xi))
    #     colorline(xi, yi, z, cmap=plt.get_cmap('jet'), linewidth=lw, alpha=1.0)
    plt.scatter(xs, ys)

    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    ax = plt.gca()
    ax.set_axis_off()
    # return fig
    plt.show()



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

def calculate_epe(input_flow, target_flow, mask):
    """
    End-point-Error computation
    Args:
        input_flow: estimated flow [BxHxW,2]
        target_flow: ground-truth flow [BxHxW,2]
    Output:
        Averaged end-point-error (value)
    """
    epe = np.linalg.norm(input_flow-target_flow, axis=2)
    epe = epe[mask].mean()
    # epe = np.median(epe[mask])

    # epe = input_flow-target_flow
    # exec(embed_breakpoint())
    
    # epe = (input_flow-target_flow)[mask]
    # epe = np.linalg.norm(input_flow-target_flow, axis=2)[mask]
    return epe


def read_scenes(opt):
    if opt.scene == 'all':
        # exec(embed_breakpoint())
        scenes = []
        for cur, dirs, files in os.walk(opt.dset_dir):
            for d in dirs:
                if d.startswith('v_'):
                    path = os.path.join(cur, d)
                    scenes.append(hpatches_helper.read_scene(path))
            break
        return scenes
    else:
        assert opt.scene.startswith('v_'), 'only support viewpoint changing scenes'
        path = os.path.join(opt.dset_dir, opt.scene)
        return [hpatches_helper.read_scene(path)]


# def restore_conf_from_square(conf, flow_shape_old):
#     size = max(conf.shape[:2])
#     start_x = size // 2 - flow_shape_old[1] // 2
#     start_y = size // 2 - flow_shape_old[0] // 2
#     conf = conf[start_y:start_y+flow_shape_old[0], start_x:start_x+flow_shape_old[1]]
#     return conf


# def get_reprojection_error_and_confidence(opt, scene):
#     reprojs = []
#     confs = []
#     for to in [1,2]:
#         gt = os.path.join(scene.path, f'gt_optical_flow_1_{to+1}.npy')
#         pred = os.path.join(scene.path, f'{opt.method}_pred_{opt.mode}_1_{to+1}.npy')
#         gt = np.load(gt) #* mask[..., None]
#         pred = np.load(pred)# * mask[..., None]
#         pred = hpatches_helper.correspondence_map_to_optical_flow(pred)

#         mask = os.path.join(scene.path, f'gt_mask_1_{to+1}.npy')
#         mask = np.load(mask)

#         conf = os.path.join(scene.path, f'cotr_new_conf_corr_map_1_{to+1}.npy')
#         conf = np.load(conf)
#         conf = restore_conf_from_square(conf, mask.shape)

#         reproj = np.linalg.norm(pred - gt, axis=2)

#         reprojs.append(reproj*mask)
#         confs.append(conf*mask)
#         exec(embed_breakpoint())
#     return reprojs, confs


# def eval_scene_sparse(opt, scene):
#     for to in [1,2,3,4,5]:
#         gt = os.path.join(scene.path, f'gt_optical_flow_1_{to+1}.npy')
#         gt = np.load(gt)
#         mask = os.path.join(scene.path, f'gt_mask_1_{to+1}.npy')
#         mask = np.load(mask)
#         pred = os.path.join(scene.path, f'{opt.method}_pred_{opt.mode}_1_{to+1}.npy')


# def pad_to_square_np(img):
#     if len(img.shape) == 2:
#         h, w = img.shape
#     elif len(img.shape) == 3:
#         h, w, c = img.shape
#     elif len(img.shape) == 4:
#         b, h, w, c = img.shape
#     else:
#         raise ValueError
#     size = max(h, w)
#     start_x = size // 2 - w // 2
#     start_y = size // 2 - h // 2
#     if len(img.shape) == 2:
#         canvas = np.zeros([size, size], dtype=img.dtype)
#         canvas[start_y:start_y + h, start_x:start_x + w] = img
#     elif len(img.shape) == 3:
#         canvas = np.zeros([size, size, c], dtype=img.dtype)
#         canvas[start_y:start_y + h, start_x:start_x + w, :] = img
#     elif len(img.shape) == 4:
#         canvas = np.zeros([b, size, size, c], dtype=img.dtype)
#         canvas[:, start_y:start_y + h, start_x:start_x + w, :] = img
#     return canvas


def evaluate_scene(opt, scene, mask_from='gt', percentile=None, threshold=None, component='xy'):
    epes = []
    pck1 = []
    pck3 = []
    pck5 = []
    # confs = []
    for to in [1,2,3,4,5]:
    # for to in [3]:
        
        gt = os.path.join(scene.path, f'gt_optical_flow_1_{to+1}.npy')
        # pred = os.path.join(scene.path, f'{opt.method}_pred_{opt.mode}_1_{to+1}.npy')
        pred = os.path.join(scene.path, f'pred_{opt.mode}_1_{to+1}.npy')
        # pred = os.path.join(scene.path, f'verification_{opt.mode}_1_{to+1}.npy')
        gt = np.load(gt) #* mask[..., None]
        pred = np.load(pred)# * mask[..., None]
        
        
        

        if opt.mode == 'corr_map':
            gt = hpatches_helper.optical_flow_to_correspondence_map(gt)
        if opt.mode=='optical_flow':
            x, y = np.meshgrid(np.linspace(0, gt.shape[1] - 1, gt.shape[1]),
                           np.linspace(0, gt.shape[0] - 1, gt.shape[0]))
            base_grid = np.stack([x, y], axis=-1)
            im1_shape = np.array([scene.imgs[0].shape[1], scene.imgs[0].shape[0]])
            im2_shape = np.array([scene.imgs[to].shape[1], scene.imgs[to].shape[0]])
            # if (im1_shape != im2_shape).any():
            #     exec(embed_breakpoint())
            # gt = (gt + base_grid) / (im1_shape / im2_shape) - base_grid
            # pred = (pred + base_grid) * (im1_shape / im2_shape) - base_grid
            # flow_gt_np = (flow_gt_np + base_grid) * (im1_shape / im2_shape) - base_grid
            # exec(embed_breakpoint())
            pass
        # if opt.mode == 'corr_map':
        #     pred = hpatches_helper.correspondence_map_to_optical_flow(pred)
        if mask_from == 'gt':
            mask = os.path.join(scene.path, f'gt_mask_1_{to+1}.npy')
            mask = np.load(mask)
            # if False:
            #     mask = pad_to_square_np(mask)
            #     raw_pred = os.path.join(scene.path, 'raw_' + f'{opt.method}_pred_{opt.mode}_1_{to+1}.npy')
            #     raw_pred=np.load(raw_pred)
            #     mask_a =np.zeros_like(mask)
            #     mask_a[tuple(raw_pred[...,:2][...,::-1].astype(np.int).T)] = 1
            #     # exec(embed_breakpoint())
            #     mask = mask * mask_a
            #     size = max(*mask.shape)
            #     h, w = pred.shape[:2]
            #     start_x = size // 2 - w // 2
            #     start_y = size // 2 - h // 2
            #     mask = mask[start_y:start_y + h, start_x:start_x + w]
            #     # exec(embed_breakpoint())

        elif mask_from == 'cotr':
            mask_gt = os.path.join(scene.path, f'gt_mask_1_{to+1}.npy')
            mask_gt = np.load(mask_gt)
            conf = os.path.join(scene.path, f'cotr_new_conf_corr_map_1_{to+1}.npy')
            conf = np.load(conf)
            conf = restore_conf_from_square(conf, mask_gt.shape)

            # conf = np.linalg.norm(pred - gt, axis=2)
            # exec(embed_breakpoint())
            conf = conf * mask_gt
            if threshold is None:
                threshold = np.percentile(conf[mask_gt], percentile)
            print(threshold)
            mask = conf < threshold
            mask = mask * mask_gt
            # plt.imshow(mask)
            exec(embed_breakpoint())
            # plt.show()
            # plt.savefig(f'{threshold}_pred.png', bbox_inches='tight', pad_inches=0)
        
        # out_h, out_w = mask.shape
        # x, y = np.meshgrid(np.linspace(0, out_w - 1, out_w), 
        #                    np.linspace(0, out_h - 1, out_h))
        # w_ref_orig = 735
        # h_ref_orig = 1612
        # w_trg_orig = 1210
        # h_trg_orig = 1613
        # im1_shape = np.array([w_ref_orig, h_ref_orig])
        # im2_shape = np.array([w_trg_orig, h_trg_orig])
        # exec(embed_breakpoint())
        
        # if False:
        #     if scene.imgs[0].shape[0] < max(*scene.imgs[0].shape):
        #         gt[..., 1] += (max(*scene.imgs[0].shape) - scene.imgs[0].shape[0]) // 2
        #         pred[..., 1] += (max(*scene.imgs[0].shape) - scene.imgs[0].shape[0]) // 2
        #     if scene.imgs[0].shape[1] < max(*scene.imgs[0].shape):
        #         gt[..., 0] += (max(*scene.imgs[0].shape) - scene.imgs[0].shape[1]) // 2
        #         pred[..., 0] += (max(*scene.imgs[0].shape) - scene.imgs[0].shape[1]) // 2
        #     gt = pad_to_square_np(gt)
        #     mask = pad_to_square_np(mask)
        #     pred = pad_to_square_np(pred)
        
        if component == 'xy':
            print('xy')
            epe = calculate_epe(pred, gt, mask)
        elif component == 'x':
            print('x')
            epe = calculate_epe(pred[...,0:1], gt[...,0:1], mask)
        elif component == 'y':
            print('y')
            epe = calculate_epe(pred[...,1:2], gt[...,1:2], mask)
        offset = np.linalg.norm(pred-gt, axis=2)[mask]
        # epe = epe[mask].mean()
        epes.append(epe)
        pck1.append((offset<1).sum() / len(offset))
        pck3.append((offset<3).sum() / len(offset))
        pck5.append((offset<5).sum() / len(offset))

        # confs.append(conf)
        # save_img = np.linalg.norm(pred - gt, axis=2)*mask / 50
        # save_img = np.clip(save_img, 0, 1)
        # save_img = np.clip((np.repeat(save_img[...,None], 3, axis=2) * 255), 0, 255).astype(np.uint8)
        # imageio.imsave(f'./{opt.method}/{os.path.basename(scene.path)}_epe_1_{to+1}.png', save_img)
        # if to == 5:
        #     exec(embed_breakpoint())
        # if opt.method == 'cotr_ci_1_1.0->0.1_16X':
        #     exec(hpatches_helper.embed_breakpoint())
    # exec(embed_breakpoint())
    # epes = np.concatenate(epes)
    epes = np.array(epes)

    # confs = np.array(confs)
    pck1 = np.array(pck1)
    pck3 = np.array(pck3)
    pck5 = np.array(pck5)
    return epes, pck1, pck3, pck5
    exec(embed_breakpoint())


def evaluate_scene_sparse(opt, scene):
    print(scene.path)
    epes = []
    pck1 = []
    pck3 = []
    pck5 = []
    # confs = []
    for to in [1,2,3,4,5]:
        if opt.mode == 'corr_map':
            gt = os.path.join(scene.path, f'gt_optical_flow_1_{to+1}.npy')
            gt = np.load(gt)
            gt = hpatches_helper.optical_flow_to_correspondence_map(gt)
            mask = os.path.join(scene.path, f'gt_mask_1_{to+1}.npy')
            mask = np.load(mask)
            # offset_x = (src_shape_new[1] - src_shape_old[1]) // 2
            # offset_y = (src_shape_new[0] - src_shape_old[0]) // 2
            if False:
                if scene.imgs[0].shape[0] < max(*scene.imgs[0].shape):
                    gt[..., 1] += (max(*scene.imgs[0].shape) - scene.imgs[0].shape[0]) // 2
                if scene.imgs[0].shape[1] < max(*scene.imgs[0].shape):
                    gt[..., 0] += (max(*scene.imgs[0].shape) - scene.imgs[0].shape[1]) // 2
                gt = pad_to_square_np(gt)
                mask = pad_to_square_np(mask)

            # raw_pred = os.path.join(scene.path, 'raw_' + f'{opt.method}_pred_{opt.mode}_1_{to+1}.npy')
            raw_pred = os.path.join(scene.path, f'cotr_stretch_0.5->0.0625_4X_sparse_1_{to+1}.npy')
            raw_pred = np.load(raw_pred)
            if raw_pred.shape[0] == 0:
                epes.append(np.nan)
                pck1.append(np.nan)
                pck3.append(np.nan)
                pck5.append(np.nan)
                continue
            print(raw_pred.shape)
            # exec(embed_breakpoint())

            valid = mask[tuple(raw_pred[...,:2][...,::-1].astype(np.int).T)]
            offset = raw_pred[..., 2:] - gt[tuple(raw_pred[...,:2][...,::-1].astype(np.int).T)]
            offset = offset[valid]
            offset = np.linalg.norm(offset, axis=-1)
            # print(offset.mean())
            # exec(embed_breakpoint())
            epes.append(offset.mean())
            pck1.append((offset<1).sum() / len(offset))
            pck3.append((offset<3).sum() / len(offset))
            pck5.append((offset<5).sum() / len(offset))
        elif opt.mode == 'optical_flow':
            gt = os.path.join(scene.path, f'gt_optical_flow_1_{to+1}.npy')
            gt = np.load(gt)
            gt = hpatches_helper.optical_flow_to_correspondence_map(gt)
            pred = os.path.join(scene.path, f'{opt.method}_pred_optical_flow_1_{to+1}.npy')
            pred = np.load(pred)
            pred = hpatches_helper.optical_flow_to_correspondence_map(pred)
            mask = os.path.join(scene.path, f'gt_mask_1_{to+1}.npy')
            mask = np.load(mask)
            # offset_x = (src_shape_new[1] - src_shape_old[1]) // 2
            # offset_y = (src_shape_new[0] - src_shape_old[0]) // 2
            if scene.imgs[0].shape[0] < max(*scene.imgs[0].shape):
                gt[..., 1] += (max(*scene.imgs[0].shape) - scene.imgs[0].shape[0]) // 2
                pred[..., 1] += (max(*scene.imgs[0].shape) - scene.imgs[0].shape[0]) // 2
            if scene.imgs[0].shape[1] < max(*scene.imgs[0].shape):
                gt[..., 0] += (max(*scene.imgs[0].shape) - scene.imgs[0].shape[1]) // 2
                pred[..., 0] += (max(*scene.imgs[0].shape) - scene.imgs[0].shape[1]) // 2
            gt = pad_to_square_np(gt)
            pred = pad_to_square_np(pred)
            mask = pad_to_square_np(mask)

            raw_pred = os.path.join(scene.path, 'raw_' + f'zoom_128q_w_hist_1.0->0.1_16X_pred_corr_map_1_{to+1}.npy')
            raw_pred = np.load(raw_pred)

            valid = mask[tuple(raw_pred[...,:2][...,::-1].astype(np.int).T)]
            offset = pred[tuple(raw_pred[...,:2][...,::-1].astype(np.int).T)] - gt[tuple(raw_pred[...,:2][...,::-1].astype(np.int).T)]
            offset = offset[valid]
            offset = np.linalg.norm(offset, axis=-1)
            print('of',offset.mean())
            epes.append(offset.mean())
            pck1.append((offset<1).sum() / len(offset))
            pck5.append((offset<5).sum() / len(offset))
        # epes.append(np.median(offset))
    # exec(embed_breakpoint())
    epes = np.array(epes)
    pck1 = np.array(pck1)
    pck3 = np.array(pck3)
    pck5 = np.array(pck5)
    return epes, pck1, pck3, pck5

# def evaluate_scenes_sparse(opt, scenes):
#     epes = []
#     for s in scenes:
#         epes.append(np.median(evaluate_scene_sparse(opt, s)))
#     epes = np.array(epes)
#     return epes
#         # exec(embed_breakpoint())


def eval_scene_with_fitting(opt, scene):
    print(scene.path)
    epes = []
    # confs = []
    for to in [1,2,3,4,5]:
        if opt.mode == 'corr_map':
            gt = os.path.join(scene.path, f'gt_optical_flow_1_{to+1}.npy')
            gt = np.load(gt)
            gt = hpatches_helper.optical_flow_to_correspondence_map(gt)
            mask = os.path.join(scene.path, f'gt_mask_1_{to+1}.npy')
            mask = np.load(mask)
            # offset_x = (src_shape_new[1] - src_shape_old[1]) // 2
            # offset_y = (src_shape_new[0] - src_shape_old[0]) // 2
            if False:
                if scene.imgs[0].shape[0] < max(*scene.imgs[0].shape):
                    gt[..., 1] += (max(*scene.imgs[0].shape) - scene.imgs[0].shape[0]) // 2
                if scene.imgs[0].shape[1] < max(*scene.imgs[0].shape):
                    gt[..., 0] += (max(*scene.imgs[0].shape) - scene.imgs[0].shape[1]) // 2
                gt = pad_to_square_np(gt)
                mask = pad_to_square_np(mask)

            # raw_pred = os.path.join(scene.path, 'raw_' + f'{opt.method}_pred_{opt.mode}_1_{to+1}.npy')
            raw_pred = os.path.join(scene.path, f'cotr_0.5->0.0625_4X_sparse_1_{to+1}.npy')
            raw_pred = np.load(raw_pred)
            if raw_pred.shape[0] <= 10:
                epes.append(np.nan)
                continue
            print(raw_pred.shape)
            M, _ = cv2.findHomography(raw_pred[:,:2], raw_pred[:,2:], cv2.LMEDS, 25.0)
            pred = hpatches_helper.H_to_correspondence_map(M, gt.shape)
            offset = np.linalg.norm(pred - gt, axis=2)[mask]
            
            # valid = mask[tuple(raw_pred[...,:2][...,::-1].astype(np.int).T)]
            # offset = raw_pred[..., 2:] - gt[tuple(raw_pred[...,:2][...,::-1].astype(np.int).T)]
            # offset = offset[valid]
            # offset = np.linalg.norm(offset, axis=-1)
            # print(offset.mean())
            # exec(embed_breakpoint())
            epes.append(offset.mean())
        elif opt.mode == 'optical_flow':
            gt = os.path.join(scene.path, f'gt_optical_flow_1_{to+1}.npy')
            gt = np.load(gt)
            gt = hpatches_helper.optical_flow_to_correspondence_map(gt)
            pred = os.path.join(scene.path, f'{opt.method}_pred_optical_flow_1_{to+1}.npy')
            pred = np.load(pred)
            pred = hpatches_helper.optical_flow_to_correspondence_map(pred)
            mask = os.path.join(scene.path, f'gt_mask_1_{to+1}.npy')
            mask = np.load(mask)
            # offset_x = (src_shape_new[1] - src_shape_old[1]) // 2
            # offset_y = (src_shape_new[0] - src_shape_old[0]) // 2
            if scene.imgs[0].shape[0] < max(*scene.imgs[0].shape):
                gt[..., 1] += (max(*scene.imgs[0].shape) - scene.imgs[0].shape[0]) // 2
                pred[..., 1] += (max(*scene.imgs[0].shape) - scene.imgs[0].shape[0]) // 2
            if scene.imgs[0].shape[1] < max(*scene.imgs[0].shape):
                gt[..., 0] += (max(*scene.imgs[0].shape) - scene.imgs[0].shape[1]) // 2
                pred[..., 0] += (max(*scene.imgs[0].shape) - scene.imgs[0].shape[1]) // 2
            gt = pad_to_square_np(gt)
            pred = pad_to_square_np(pred)
            mask = pad_to_square_np(mask)

            raw_pred = os.path.join(scene.path, 'raw_' + f'zoom_128q_w_hist_1.0->0.1_16X_pred_corr_map_1_{to+1}.npy')
            raw_pred = np.load(raw_pred)

            valid = mask[tuple(raw_pred[...,:2][...,::-1].astype(np.int).T)]
            offset = pred[tuple(raw_pred[...,:2][...,::-1].astype(np.int).T)] - gt[tuple(raw_pred[...,:2][...,::-1].astype(np.int).T)]
            offset = offset[valid]
            offset = np.linalg.norm(offset, axis=-1)
            print('of',offset.mean())
            epes.append(offset.mean())
        # epes.append(np.median(offset))
    epes = np.array(epes)
    return epes

# def init_weights(m):
#     if type(m) == nn.Linear:
#         torch.nn.init.xavier_normal_(m.weight)

# class NerfPositionalEncoding(nn.Module):
#     def __init__(self, depth=32):
#         '''
#         out_dim = in_dim * depth * 2
#         '''
#         super().__init__()
#         self.bases  = [i+1 for i in range(depth)]
#         print(self.bases)


#     # @torch.no_grad()
#     def forward(self, inputs):
#         out = torch.cat([torch.sin(i * math.pi * inputs) for i in self.bases] + [torch.cos(i * math.pi * inputs) for i in self.bases], axis=-1)
#         assert torch.isnan(out).any() == False
#         return out

# class SimpleMappingNet(nn.Module):
#     def __init__(self):
#         super(SimpleMappingNet, self).__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(128, 128, bias=True),
#             nn.Tanh(),
#             nn.Linear(128, 128, bias=True),
#             nn.Tanh(),
#             nn.Linear(128, 128, bias=True),
#             nn.Tanh(),
#             nn.Linear(128, 2, bias=True),
#             nn.Tanh()
#         )
#         self.mlp.apply(init_weights)
#         self.pos = NerfPositionalEncoding()

#     def forward(self, x):
#         x = self.pos(x)
#         # print(x.shape)
#         return self.mlp(x)

# def overfit(model, points, iterations=10000, threshold=1e-4):
#     def form_batch(points):
#         mask = np.random.choice(points.shape[0], 128)
#         batch = points[mask]
#         x, y = batch[:,:2], batch[:,2:]
#         x= torch.from_numpy(x).float().cuda()
#         y= torch.from_numpy(y).float().cuda()
#         return x, y
#     optim_list = [{"params": filter(
#         lambda p: p.requires_grad, model.parameters()), "lr": 1e-5}]
#     optim = torch.optim.Adam(optim_list, weight_decay=0)
#     for i in range(iterations):
#         x, y = form_batch(points)
#         y_ = model(x)
#         loss = torch.nn.functional.mse_loss(y_, y)
#         loss.backward()
#         optim.step()
#         # print(loss)
#         # exec(embed_breakpoint())
#     print(loss)
#     # exec(embed_breakpoint())

# def triangulate_corr(corr, from_shape, to_shape):
#     corr = corr.copy()
#     from_shape = from_shape[:2]
#     to_shape = to_shape[:2]
#     corr = (corr / np.concatenate([from_shape[::-1], to_shape[::-1]])) * 2 - 1
#     smn = SimpleMappingNet().cuda()
#     # 
#     overfit(smn, corr)
#     x, y = np.meshgrid(np.linspace(-1, 1, to_shape[1]),
#                         np.linspace(-1, 1, to_shape[0]))
#     base_grid = np.stack([x, y], axis=-1)
#     out_grid = []
#     for row in base_grid:
#         out = smn(torch.from_numpy(row).float().cuda())
#         out = out.detach().cpu().numpy()
#         out_grid.append(out)
#     out_grid = np.array(out_grid)
#     out_grid = np.array(out_grid) * 0.5 + 0.5
#     out_grid *= to_shape[::-1]
#     return out_grid
#     exec(embed_breakpoint())

vertex = """
    attribute vec4 color;
    attribute vec2 position;
    varying vec4 v_color;
    void main()
    {
        gl_Position = vec4(position, 0.0, 1.0);
        v_color = color;
    } """

fragment = """
    varying vec4 v_color;
    void main()
    {
        gl_FragColor = v_color;
    } """


class Canvas(app.Canvas):
    def __init__(self, mesh, color, size=(600, 600)):
        # We hide the canvas upon creation.
        app.Canvas.__init__(self, show=False, size=size)
        self._t0 = time()
        # Texture where we render the scene.
        self._rendertex = gloo.Texture2D(shape=self.size[::-1] + (4,), internalformat='rgba32f')
        # FBO.
        self._fbo = gloo.FrameBuffer(self._rendertex,
                                     gloo.RenderBuffer(self.size[::-1]))
        # Regular program that will be rendered to the FBO.
        self.program = gloo.Program(vertex, fragment)
        self.program["position"] = mesh
        self.program['color'] = color
        # self.program["scale"] = 3
        # self.program["center"] = [-0.5, 0]
        # self.program["iter"] = 300
        # self.program['resolution'] = self.size
        # We manually draw the hidden canvas.
        self.update()

    def on_draw(self, event):
        # Render in the FBO.
        with self._fbo:
            gloo.clear('black')
            gloo.set_viewport(0, 0, *self.size)
            self.program.draw()
            # Retrieve the contents of the FBO texture.
            # exec(embed_breakpoint())
            self.im = read_pixels((0, 0, self.size[0], self.size[1]), True, out_type='float')
        self._time = time() - self._t0
        # Immediately exit the application.
        app.quit()


def triangulate_corr(corr, from_shape, to_shape):
    corr = corr.copy()
    # corr = corr[:3]
    to_shape = to_shape[:2]
    from_shape = from_shape[:2]
    corr = corr / np.concatenate([from_shape[::-1], to_shape[::-1]])
    # print()
    tri = Delaunay(corr[:,:2])
    mesh = corr[:,:2][tri.simplices].astype(np.float32) * 2 - 1
    mesh[..., 1] *= -1
    color = corr[:,2:][tri.simplices].astype(np.float32)
    color = np.concatenate([color, np.ones_like(color[...,0:2])], axis=-1)
    c = Canvas(mesh.reshape(-1, 2), color.reshape(-1, 4), size=(from_shape[::-1]))
    app.run()
    render = c.im.copy()
    render = render[..., :2]
    # render = render / 255
    render *= np.array(to_shape[::-1])
    return render
    exec(embed_breakpoint())



def eval_scene_with_triangulation(opt, scene):
    print(scene.path)
    epes = []
    pck1 = []
    pck3 = []
    pck5 = []
    # confs = []
    for to in [1,2,3,4,5]:
        if opt.mode == 'corr_map':
            gt = os.path.join(scene.path, f'gt_optical_flow_1_{to+1}.npy')
            gt = np.load(gt)
            gt = hpatches_helper.optical_flow_to_correspondence_map(gt)
            mask = os.path.join(scene.path, f'gt_mask_1_{to+1}.npy')
            mask = np.load(mask)
            # offset_x = (src_shape_new[1] - src_shape_old[1]) // 2
            # offset_y = (src_shape_new[0] - src_shape_old[0]) // 2
            if False:
                if scene.imgs[0].shape[0] < max(*scene.imgs[0].shape):
                    gt[..., 1] += (max(*scene.imgs[0].shape) - scene.imgs[0].shape[0]) // 2
                if scene.imgs[0].shape[1] < max(*scene.imgs[0].shape):
                    gt[..., 0] += (max(*scene.imgs[0].shape) - scene.imgs[0].shape[1]) // 2
                gt = pad_to_square_np(gt)
                mask = pad_to_square_np(mask)

            # raw_pred = os.path.join(scene.path, 'raw_' + f'{opt.method}_pred_{opt.mode}_1_{to+1}.npy')
            raw_pred = os.path.join(scene.path, f'cotr_stretch_0.5->0.0625_4X_sparse_1_{to+1}.npy')
            raw_pred = np.load(raw_pred)
            if raw_pred.shape[0] <= 10:
                epes.append(np.nan)
                pck1.append(np.nan)
                pck3.append(np.nan)
                pck5.append(np.nan)
                continue
            # print(raw_pred.shape)
            # raw_pred = raw_pred[:4]
            pred = triangulate_corr(raw_pred, scene.imgs[to].shape, scene.imgs[0].shape)
            mask = mask * np.prod((pred != 0.0), axis=2).astype(np.bool)
            # exec(embed_breakpoint())
            # M, _ = cv2.findHomography(raw_pred[:,:2], raw_pred[:,2:], cv2.LMEDS, 25.0)
            # pred = hpatches_helper.H_to_correspondence_map(M, gt.shape)
            offset = np.linalg.norm(pred - gt, axis=2)[mask]
            
            # valid = mask[tuple(raw_pred[...,:2][...,::-1].astype(np.int).T)]
            # offset = raw_pred[..., 2:] - gt[tuple(raw_pred[...,:2][...,::-1].astype(np.int).T)]
            # offset = offset[valid]
            # offset = np.linalg.norm(offset, axis=-1)
            print(offset.mean())
            # exec(embed_breakpoint())
            epes.append(offset.mean())
            pck1.append((offset<1).sum() / len(offset))
            pck3.append((offset<3).sum() / len(offset))
            pck5.append((offset<5).sum() / len(offset))
        elif opt.mode == 'optical_flow':
            assert 0
            gt = os.path.join(scene.path, f'gt_optical_flow_1_{to+1}.npy')
            gt = np.load(gt)
            gt = hpatches_helper.optical_flow_to_correspondence_map(gt)
            pred = os.path.join(scene.path, f'{opt.method}_pred_optical_flow_1_{to+1}.npy')
            pred = np.load(pred)
            pred = hpatches_helper.optical_flow_to_correspondence_map(pred)
            mask = os.path.join(scene.path, f'gt_mask_1_{to+1}.npy')
            mask = np.load(mask)
            # offset_x = (src_shape_new[1] - src_shape_old[1]) // 2
            # offset_y = (src_shape_new[0] - src_shape_old[0]) // 2
            if scene.imgs[0].shape[0] < max(*scene.imgs[0].shape):
                gt[..., 1] += (max(*scene.imgs[0].shape) - scene.imgs[0].shape[0]) // 2
                pred[..., 1] += (max(*scene.imgs[0].shape) - scene.imgs[0].shape[0]) // 2
            if scene.imgs[0].shape[1] < max(*scene.imgs[0].shape):
                gt[..., 0] += (max(*scene.imgs[0].shape) - scene.imgs[0].shape[1]) // 2
                pred[..., 0] += (max(*scene.imgs[0].shape) - scene.imgs[0].shape[1]) // 2
            gt = pad_to_square_np(gt)
            pred = pad_to_square_np(pred)
            mask = pad_to_square_np(mask)

            raw_pred = os.path.join(scene.path, 'raw_' + f'zoom_128q_w_hist_1.0->0.1_16X_pred_corr_map_1_{to+1}.npy')
            raw_pred = np.load(raw_pred)

            valid = mask[tuple(raw_pred[...,:2][...,::-1].astype(np.int).T)]
            offset = pred[tuple(raw_pred[...,:2][...,::-1].astype(np.int).T)] - gt[tuple(raw_pred[...,:2][...,::-1].astype(np.int).T)]
            offset = offset[valid]
            offset = np.linalg.norm(offset, axis=-1)
            print('of',offset.mean())
            epes.append(offset.mean())
            pck1.append((offset<1).sum() / len(offset))
            pck5.append((offset<5).sum() / len(offset))
        # epes.append(np.median(offset))
    epes = np.array(epes)
    pck1 = np.array(pck1)
    pck3 = np.array(pck3)
    pck5 = np.array(pck5)
    return epes, pck1, pck3, pck5
