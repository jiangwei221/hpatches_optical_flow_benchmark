'''
eval the predicted optical flow or correspondence map on hpatch dataset.
name predicted optical flow as "pred_optical_flow_1_X.npy" in shape [H, W, 2],
name predicted correspondence map as "pred_corr_map_1_X.npy" in shape [H, W, 2],
where 'X' is from [2, 3, 4, 5].
'''
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

import eval_helper
import hpatches_helper

def main(opt):
    scenes = eval_helper.read_scenes(opt)
    # exec(hpatches_helper.embed_breakpoint())
    epes_list = []
    pck1_list = []
    pck3_list = []
    pck5_list = []
    for s in scenes:
        hpatches_helper.dump_ground_truth(s)
        epes, pck1, pck3, pck5 = eval_helper.evaluate_scene(opt, s)
        epes_list.append(epes)
        pck1_list.append(pck1)
        pck3_list.append(pck3)
        pck5_list.append(pck5)
        # exec(hpatches_helper.embed_breakpoint())
    epes_list = np.array(epes_list)
    pck1_list = np.array(pck1_list)
    pck3_list = np.array(pck3_list)
    pck5_list = np.array(pck5_list)

    for s, e in zip(scenes, epes_list):
        print(os.path.basename(s.path), e)
    
    for item in [epes_list, pck1_list, pck3_list, pck5_list]:
        print('*********')
        sum_=0
        for e in item.T:
            mask = e == e
            print(e[mask].mean())
            sum_ += e[mask].mean()

        print('mean:', sum_/5)
    exec(hpatches_helper.embed_breakpoint())


    epes_list2 = []
    for s in scenes:
        # hpatches_helper.dump_ground_truth(s)
        epes = eval_helper.evaluate_scene_sparse(opt, s)
        epes_list2.append(epes)
    epes_list2 = np.array(epes_list2)

    for s, e, e2 in zip(scenes, epes_list, epes_list2):
        mask = e-e2 == e-e2
        print(os.path.basename(s.path), (abs(e-e2))[mask].max())
    exec(hpatches_helper.embed_breakpoint())

    # scenes = eval_helper.read_scenes(opt)
    # reprojs = []
    # confs = []
    # for s in scenes:
    #     hpatches_helper.dump_ground_truth(s)
    #     reproj, conf = eval_helper.get_reprojection_error_and_confidence(opt, s)
    #     reprojs.append(reproj)
    #     confs.append(conf)

    # exec(hpatches_helper.embed_breakpoint())

    slots = 5
    scenes = eval_helper.read_scenes(opt)
    epes_list = {}
    for s in scenes:
        hpatches_helper.dump_ground_truth(s)
        # for threshold in [84.78373751499694, 14.921728542764763, 8.968928168711981, 5.5561766924354234, 1.3104856420723663]:
            # epes = eval_helper.evaluate_scene(opt, s, mask_from='cotr', threshold=threshold)
        for percentile in np.linspace(100, 1, slots):
            epes = eval_helper.evaluate_scene(opt, s, mask_from='gt', percentile=percentile)
            # percentile = threshold
            if percentile not in epes_list:
                epes_list[percentile] = np.array([])
            # exec(hpatches_helper.embed_breakpoint())
            epes_list[percentile] = np.concatenate([epes_list[percentile], epes]) 
    # epes_list = np.array(epes_list)
    # avg_epes = {}
    # epe_mat = []
    # for k, v in epes_list.items():
    #     avg_epes[k] = np.array(v).mean(axis=0)
    #     epe_mat.append(np.array(v).mean(axis=0))

    exec(hpatches_helper.embed_breakpoint())
    for i, percentile in enumerate(np.linspace(100, 1, slots)):
    # for threshold in [84.78373751499694, 14.921728542764763, 8.968928168711981, 5.5561766924354234, 1.3104856420723663]:
        # plt.subplot(3,4,i+1)
        a = epes_list[percentile]
        hist, bins = np.histogram(a, bins=300, range=(0, 10))
        #hist = hist * (100/hist.max() )
        width = (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=width, label=f'percentile={percentile}', alpha=0.2)
        plt.legend()
    plt.title('how reprojection error histogram -- GLUnet')

    # for i in range(2):
    #     plt.plot(np.linspace(100, 1, slots), np.array(epe_mat).T[i], label=f'view_{i+2}')
    # plt.legend()
    # plt.title('how reprojection error change as lifting the threshold')
    # plt.xlabel('cycle consistency error percentile')
    # plt.ylabel('average reprojection error')
    # plt.show()

    exec(hpatches_helper.embed_breakpoint())


    # scenes = eval_helper.read_scenes(opt)
    # epes_list = {}
    # for s in scenes:
    #     for com in ['x']:
    #         hpatches_helper.dump_ground_truth(s)
    #         epes = eval_helper.evaluate_scene(opt, s, component=com)
    #         if com not in epes_list:
    #             epes_list[com] = []
    #         epes_list[com].append(epes)

    # avg_epes = {}
    # epe_mat = []

    # a = np.array([])
    # for i in epes_list['x']:
    #     for ii in i:
    #         a = np.concatenate([a, ii.reshape(-1)])


    # exec(hpatches_helper.embed_breakpoint())
    # for k, v in epes_list.items():
    #     avg_epes[k] = np.array(v).mean(axis=0)
    #     epe_mat.append(np.array(v).mean(axis=0))

    # for i, com in enumerate(['x', 'y', 'xy']):
    #     plt.plot(np.arange(5), np.array(epe_mat)[i], label=com)
    # plt.legend()
    # plt.title(f'X and Y component performance of {opt.method}')
    # plt.xlabel('Different views')
    # plt.ylabel('average reprojection error')
    # plt.show()

    # scenes = eval_helper.read_scenes(opt)
    
    # slots = 5
    # for val, percentile in enumerate(np.linspace(100, 1, slots)):
    #     epes_list = {}
    #     confs_list = {}
    #     for s in scenes:
    #         for com in ['xy']:
    #             hpatches_helper.dump_ground_truth(s)
    #             epes, confs = eval_helper.evaluate_scene(opt, s, component=com, percentile=percentile, mask_from='cotr')
    #             if com not in epes_list:
    #                 epes_list[com] = []
    #                 confs_list[com] = []
    #             epes_list[com].append(epes)
    #             confs_list[com].append(confs)


        # a = np.array([])
        # b = np.array([])
        # exec(hpatches_helper.embed_breakpoint())
        # for i in epes_list['xy']:
        #     for ii in i:
        #         a = np.concatenate([a, ii.reshape(-1)])
        # for i in confs_list['xy']:
        #     for ii in i:
        #         b = np.concatenate([b, ii.reshape(-1)])
        # a = (a-a.mean())/a.std()
        # b = (b-b.mean())/b.std()
        # exec(hpatches_helper.embed_breakpoint())

        # hist, bins = np.histogram(a, bins=300, range=(-25, 25))
        # np.save(f'./hist_{val}.npy',hist)
        # np.save(f'./bins_{val}.npy',bins)
    # slots = 5

# plt.figure(figsize=(20,10))
# for i, percentile in enumerate(np.linspace(100, 1, slots)):
#     # plt.subplot(3,4,i+1)
#     hist = np.load(f'./hist_{i}.npy') 
#     bins = np.load(f'./bins_{i}.npy')
#     # hist = hist * (100/hist.max() )
#     width = (bins[1] - bins[0])
#     center = (bins[:-1] + bins[1:]) / 2
#     plt.bar(center, hist, align='center', width=width, label=f'percentile={percentile}', alpha=0.2)
#     plt.legend()
# plt.title('how reprojection error histogram changes as tighten the cycle mask')
    exec(hpatches_helper.embed_breakpoint())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='corr_map', choices=['optical_flow', 'corr_map'], help='dumped data type')
    parser.add_argument('--dset_dir', type=str, required=True, help='directory to "hpatches-sequences-release"')
    parser.add_argument('--scene', type=str, default='all', help='which scene to evaluate, default is all scenes')
    parser.add_argument('--method', type=str, required=True, help='method identifier')
    opt = parser.parse_args()
    main(opt)
