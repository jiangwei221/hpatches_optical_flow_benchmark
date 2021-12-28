
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

import eval_helper
import hpatches_helper
import easydict
import cv2
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.path as mpath
from scipy.misc import imresize


def pad_to_square_np(img):
    if len(img.shape) == 2:
        h, w = img.shape
    elif len(img.shape) == 3:
        h, w, c = img.shape
    elif len(img.shape) == 4:
        b, h, w, c = img.shape
    else:
        raise ValueError
    size = max(h, w)
    start_x = size // 2 - w // 2
    start_y = size // 2 - h // 2
    if len(img.shape) == 2:
        canvas = np.zeros([size, size], dtype=img.dtype)
        canvas[start_y:start_y + h, start_x:start_x + w] = img
    elif len(img.shape) == 3:
        canvas = np.zeros([size, size, c], dtype=img.dtype)
        canvas[start_y:start_y + h, start_x:start_x + w, :] = img
    elif len(img.shape) == 4:
        canvas = np.zeros([b, size, size, c], dtype=img.dtype)
        canvas[:, start_y:start_y + h, start_x:start_x + w, :] = img
    return canvas



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


def main(opt):
    level_result = None
    scenes = eval_helper.read_scenes(opt)
    beyond_point = []
    cheated_min = {}
    raw_min = {}
    inter_min = {}
    for scene in scenes:
        for to in [1,2,3,4,5]:
            gt = os.path.join(scene.path, f'gt_optical_flow_1_{to+1}.npy')
            gt = np.load(gt)
            gt = hpatches_helper.optical_flow_to_correspondence_map(gt)
            mask = os.path.join(scene.path, f'gt_mask_1_{to+1}.npy')
            mask = np.load(mask)
            # offset_x = (src_shape_new[1] - src_shape_old[1]) // 2
            # offset_y = (src_shape_new[0] - src_shape_old[0]) // 2
            if scene.imgs[0].shape[0] < max(*scene.imgs[0].shape):
                gt[..., 1] += (max(*scene.imgs[0].shape) - scene.imgs[0].shape[0]) // 2
            if scene.imgs[0].shape[1] < max(*scene.imgs[0].shape):
                gt[..., 0] += (max(*scene.imgs[0].shape) - scene.imgs[0].shape[1]) // 2
            gt = pad_to_square_np(gt)
            mask = pad_to_square_np(mask)
            

            raw_pred = os.path.join(scene.path, 'raw_' + f'{opt.method}_pred_{opt.mode}_1_{to+1}.npy')
            raw_pred = np.load(raw_pred)
            # visualize_corrs(pad_to_square_np(scene.imgs[1]), pad_to_square_np(scene.imgs[0]), raw_pred)
            # exec(hpatches_helper.embed_breakpoint())
            inter_pred = os.path.join(scene.path, 'inter_' + f'{opt.method}_pred_{opt.mode}_1_{to+1}.npy')
            inter_pred = np.load(inter_pred)

            valid = mask[tuple(raw_pred[...,:2][...,::-1].astype(np.int).T)]
            offset = inter_pred[..., :2] - gt[tuple(raw_pred[...,:2][...,::-1].astype(np.int).T)][:,None,:]
            offset = offset[valid]
            offset = np.linalg.norm(offset, axis=-1)
            # offset = np.clip(offset / (inter_pred[0,:,4] / 256)[None,:], 0, 256*1.414)
            # exec(hpatches_helper.embed_breakpoint())
            offset2 = raw_pred[..., 2:] - gt[tuple(raw_pred[...,:2][...,::-1].astype(np.int).T)]
            offset2 = offset2[valid]
            offset2 = np.linalg.norm(offset2, axis=-1)
            if to not in cheated_min:
                cheated_min[to] = np.array([offset.min(axis=1).mean()])# offset[:,-1]# offset.min(axis=1)
                raw_min[to] = np.array([offset2.mean()])
                inter_min[to] = offset
            else:
                cheated_min[to] = np.concatenate([cheated_min[to], np.array([offset.min(axis=1).mean()])])
                raw_min[to] = np.concatenate([raw_min[to], np.array([offset2.mean()])])
                inter_min[to] = np.concatenate([inter_min[to], offset])
            beyond_point.append((inter_pred[0,:,4]<256).sum())

            if level_result is None:
                level_result = offset
            else:
                level_result = np.concatenate([level_result, offset])
    # exec(hpatches_helper.embed_breakpoint())
    
    import matplotlib
    from matplotlib import rc

    # plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
    })
    cmap = matplotlib.cm.get_cmap('gist_rainbow')
    # fig, ax = plt.subplots()
    # ax.tick_params(axis="y",direction="in", pad=-22)
    # ax.tick_params(axis="x",direction="in", pad=-15)
    # ax.set_aspect(2)

    # plt.show()
    zoom_list = 1/ np.linspace(1, 0.1, 16)
    for i, e in enumerate(level_result.T):
        # if i not in [10, 15]:
        #     continue
        # n,x,_ = plt.hist(e, bins=100, label=f'{i+1}', range=(0,50), color=cmap(i/16), histtype='step')
        n,x = np.histogram(e, bins=500, range=(0,20))
        bin_centers = 0.5*(x[1:]+x[:-1])
        plt.plot(bin_centers,n, color=cmap(i/16),label=f'{round(zoom_list[i], 2)}X')
    plt.legend(ncol=2)
    plt.ylim([0, 33581])
    plt.xlabel('End Pixel Error') 
    plt.ylabel('Number of Pixels')
    # ax.xaxis.set_label_coords(1.1, -0.050)
    # ax.set_aspect(2)
    # plt.title('Error Histogram at Different Zoom-in Level')
    plt.show()
    assert 0

    plt.plot(np.median(level_result, axis=0))
    plt.xlabel('zoom in level')  
    plt.ylabel('median error')  
    plt.title('Median error w.r.t network input at different zoom in')
    plt.show()

    for k, v in cheated_min.items():
        print(k, v.mean())
    for k, v in raw_min.items():
        print(k, v.mean())
    print("median")
    for k, v in cheated_min.items():
        print(k, np.median(v))
    for k, v in raw_min.items():
        print(k, np.median(v))

    # for i in range(16):
    #     # print(i)
    #     ll = []
    #     for k, v in inter_min.items():
    #         # print(k, np.median(v[:,i]))
    #         ll.append(np.median(v[:,i]))
    #     plt.plot(ll, label=f'{i}',color=cmap(i/16))
    # plt.legend()
    # plt.show()
    exec(hpatches_helper.embed_breakpoint())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='corr_map', choices=['optical_flow', 'corr_map'], help='dumped data type')
    parser.add_argument('--dset_dir', type=str, required=True, help='directory to "hpatches-sequences-release"')
    parser.add_argument('--scene', type=str, default='all', help='which scene to evaluate, default is all scenes')
    parser.add_argument('--method', type=str, required=True, help='method identifier')
    opt = parser.parse_args()
    main(opt)
