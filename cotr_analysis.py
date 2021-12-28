
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


import eval_helper
import hpatches_helper
import easydict

def main(opt):
    scene = hpatches_helper.read_scene(os.path.join(opt.dset_dir, 'v_yuri'))
    volume = []
    for data in ['v_yuri']:
        for to_index in [2]:
            for zoom_start in [1.0]:
                for zoom_end in [0.5, 0.25, 0.125, 0.0625, 0.03125]:
                    surface = []
                    for zoom_levels in [2,4,8,16]:
                        line = []
                        for converge_iters in [1]:
                            fake = easydict.EasyDict()
                            fake.method = f'cotr_ci_{converge_iters}_{zoom_start}->{zoom_end}_{zoom_levels}X'
                            fake.mode = 'corr_map'
                            epe = eval_helper.evaluate_scene(fake, scene)[0]
                            line.append(epe)
                        surface.append(line)
                    volume.append(surface)
    # exec(hpatches_helper.embed_breakpoint())
    volume = np.array(volume)
    # volume[0] = volume[0] * (0.7 * 0.7)
    # volume[1] = volume[1] * (0.6 * 0.7)
    # volume[2] = volume[2] * (0.5 * 0.7)
    # volume[3] = volume[3] * (0.4 * 0.7)
    # volume[4] = volume[4] * (0.3 * 0.7)
    # volume[5] = volume[5] * (0.2 * 0.7)
    # volume[6] = volume[6] * (0.1 * 0.7)
    volume[0] = volume[0] / (691.3899999999999/256)
    volume[1] = volume[1] / (592.6199999999999/256)
    volume[2] = volume[2] / (493.84999999999997/256)
    volume[3] = volume[3] / (395.08/256)
    volume[4] = volume[4] / (296.30999999999995/256)
    volume[5] = volume[5]/ (197.54/256)
    volume[6] = volume[6] / (98.77/256)

    plt.subplot(2, 3, 4)
    plt.plot( volume[:,3,0], label='covergence=1')
    plt.plot(volume[:,3,1], label='covergence=2')
    plt.plot( volume[:,3,2], label='covergence=4')
    plt.plot(volume[:,3,3], label='covergence=8')
    plt.plot( volume[:,3,4], label='covergence=16')
    my_xticks = ['784x784','672x672','560x560','448x448', '336x336', '224x224', '112x112']
    plt.xticks(np.array([0,1,2,3,4,5,6]), my_xticks)
    plt.xlabel('crop size at ending zoom(full=1600x1600)')
    plt.ylabel('error')
    plt.legend()
    plt.title('using 16 zoom in level')
    # plt.show()

    plt.subplot(2, 3, 1)
    plt.plot( volume[0,3,:], label='end=0.7')
    plt.plot( volume[1,3,:], label='end=0.6')
    plt.plot( volume[2,3,:], label='end=0.5')
    plt.plot( volume[3,3,:], label='end=0.4')
    plt.plot( volume[4,3,:], label='end=0.3')
    plt.plot( volume[5,3,:], label='end=0.2')
    plt.plot( volume[6,3,:], label='end=0.1')
    my_xticks = ['1','2','4','8', '16']
    plt.xticks(np.array([0,1,2,3,4]), my_xticks)
    plt.xlabel('different convergence iterations')
    plt.ylabel('error')
    plt.legend()
    plt.title('using 16 zoom in level')
    # plt.show()

    plt.subplot(2, 3, 2)
    plt.plot( volume[6,:,0], label='covergence=1')
    plt.plot( volume[6,:,1], label='covergence=2')
    plt.plot( volume[6,:,2], label='covergence=4')
    plt.plot( volume[6,:,3], label='covergence=8')
    plt.plot( volume[6,:,4], label='covergence=16')
    my_xticks = ['2','4','8','16']
    plt.xticks(np.array([0,1,2,3]), my_xticks)
    plt.xlabel('number of zoom in levels')
    plt.ylabel('error')
    plt.legend()
    plt.title('using end zoom=112x112')
    # plt.show()

    plt.subplot(2, 3, 5)
    plt.plot( volume[6,0,:], label='#levels=2')
    plt.plot( volume[6,1,:], label='#levels=4')
    plt.plot( volume[6,2,:], label='#levels=8')
    plt.plot( volume[6,3,:], label='#levels=16')
    my_xticks = ['1','2','4','8', '16']
    plt.xticks(np.array([0,1,2,3,4]), my_xticks)
    plt.xlabel('different convergence iterations')
    plt.ylabel('error')
    plt.legend()
    plt.title('using end zoom=112x112')
    # plt.show()

    plt.subplot(2, 3, 6)
    plt.plot( volume[0,:,0], label='end=0.7')
    plt.plot( volume[1,:,0], label='end=0.6')
    plt.plot( volume[2,:,0], label='end=0.5')
    plt.plot( volume[3,:,0], label='end=0.4')
    plt.plot( volume[4,:,0], label='end=0.3')
    plt.plot( volume[5,:,0], label='end=0.2')
    plt.plot( volume[6,:,0], label='end=0.1')
    my_xticks = ['2','4','8','16']
    plt.xticks(np.array([0,1,2,3]), my_xticks)
    plt.xlabel('number of zoom in levels')
    plt.ylabel('error')
    plt.legend()
    plt.title('using convergence iteration=1')
    # plt.show()

    plt.subplot(2, 3, 3)
    plt.plot( volume[:,0,0], label='#levels=2')
    plt.plot( volume[:,1,0], label='#levels=4')
    plt.plot( volume[:,2,0], label='#levels=8')
    plt.plot( volume[:,3,0], label='#levels=16')
    my_xticks = ['784x784','672x672','560x560','448x448', '336x336', '224x224', '112x112']
    plt.xticks(np.array([0,1,2,3,4,5,6]), my_xticks)
    plt.xlabel('crop size at ending zoom(full=1600x1600)')
    plt.ylabel('error')
    plt.legend()
    plt.title('using convergence iteration=1')
    plt.show()
    exec(hpatches_helper.embed_breakpoint())

def main2(opt):
    scenes = eval_helper.read_scenes(opt)
    zoom_end = [0.25, 0.125, 0.0625, 0.03125]
    zoom_levels = [2,4,8,16]
    converge_iters = [1]

    volume = []
    for ze in zoom_end:
        surface = []
        for zl in zoom_levels:
            for ci in converge_iters:
                fake = easydict.EasyDict()
                fake.method = f'zoom_10q_ci_{ci}_{0.5}->{ze}_{zl}X'
                fake.mode = 'corr_map'
                epes = eval_helper.evaluate_scenes_sparse(fake, scenes)
                surface.append(epes)
        volume.append(surface)
    volume = np.array(volume)
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')

    # # Make data.
    # X = np.array(zoom_end)
    # Y = np.array(zoom_levels)
    # X, Y = np.meshgrid(X, Y)
    # # R = np.sqrt(X**2 + Y**2)
    # Z = volume[...,0].T

    # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
    #                     linewidth=0, antialiased=False)

    # # Customize the z axis.
    # # ax.set_zlim(-1.01, 1.01)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    # plt.show()

# plt.plot( volume[:,0,:].mean(axis=1), label='#levels=2')
# plt.plot( volume[:,1,:].mean(axis=1), label='#levels=4')
# plt.plot( volume[:,2,:].mean(axis=1), label='#levels=8')
# plt.plot( volume[:,3,:].mean(axis=1), label='#levels=16')
    plt.plot( np.median(volume[:,0,:], axis=1), label='#levels=2')
    plt.plot( np.median(volume[:,1,:], axis=1), label='#levels=4')
    plt.plot( np.median(volume[:,2,:], axis=1), label='#levels=8')
    plt.plot( np.median(volume[:,3,:], axis=1), label='#levels=16')
    my_xticks = ['4x','8x','16x', '32x']
    plt.xticks(np.array([0,1,2,3]), my_xticks)
    plt.xlabel('zoom-in levels')
    plt.ylabel('error')
    plt.legend()
    plt.title('using convergence iteration=1')
    plt.show()

    exec(hpatches_helper.embed_breakpoint())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='corr_map', choices=['optical_flow', 'corr_map'], help='dumped data type')
    parser.add_argument('--dset_dir', type=str, required=True, help='directory to "hpatches-sequences-release"')
    parser.add_argument('--scene', type=str, default='all', help='which scene to evaluate, default is all scenes')
    # parser.add_argument('--method', type=str, required=True, help='method identifier')
    opt = parser.parse_args()
    main(opt)
