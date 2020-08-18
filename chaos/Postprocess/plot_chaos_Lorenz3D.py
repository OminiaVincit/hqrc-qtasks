#!/usr/bin/env python
# # -*- coding: utf-8 -*-
#!/usr/bin/env python
import pickle as pickle
import glob, os
import numpy as np
import argparse
import time
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys,inspect
import utils
from utils import *

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--sysname", help="type of chaotic system", type=str, default='Lorenz3D')
    parser.add_argument('--tidx', type=int, default=0)
    parser.add_argument('--used', type=int, default=0)

    args = parser.parse_args()
    sysname, tidx = args.sysname, args.tidx

    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    results_dir = os.path.dirname(current_dir) + "/Results"
    print(results_dir)
    eval_path = os.path.join(results_dir, '{}/Evaluation_Data'.format(sysname))
    print(eval_path)
    fig_path = os.path.join(results_dir, '{}/Eval_Figures'.format(sysname))
    if os.path.isdir(fig_path) == False:
        os.mkdir(fig_path)
    
    maxLyp = 1.0
    dt = 0.01
    if sysname == 'Lorenz3D':
        maxLyp = 0.9056
    elif 'Lorenz96_F10' in sysname:
        maxLyp = 2.27
    elif 'Lorenz96_F8' in sysname:
        maxLyp = 1.68
    elif 'KuramotoSivashinskyGP64' in sysname:
        maxLyp = 20
        dt = 0.25

    # list of models
    models_10000 = [\
        #['hqrc_pinv-RDIM_1-N_used_10000-DL_2000-Nqr_1-J_1.0-fJ_ion_trap-V_15-TAU_0.125-NL_1-IPL_1000-IUL_0-REG_1e-07-AU_0-NICS_2', 'V_15-TAU_0.125'],
        ['hqrc_pinv-RDIM_1-N_used_10000-DL_2000-Nqr_1-J_1.0-fJ_ion_trap-V_15-TAU_0.25-NL_1-IPL_1000-IUL_0-REG_1e-07-AU_0-NICS_2', 'V_15-TAU_0.25'],
        ['hqrc_pinv-RDIM_1-N_used_10000-DL_2000-Nqr_1-J_1.0-fJ_ion_trap-V_15-TAU_0.5-NL_1-IPL_1000-IUL_0-REG_1e-07-AU_0-NICS_2', 'V_15-TAU_0.5'],
        ['hqrc_pinv-RDIM_1-N_used_10000-DL_2000-Nqr_1-J_1.0-fJ_ion_trap-V_15-TAU_1.0-NL_1-IPL_1000-IUL_0-REG_1e-07-AU_0-NICS_2', 'V_15-TAU_1.0'],
        ['hqrc_pinv-RDIM_1-N_used_10000-DL_2000-Nqr_1-J_1.0-fJ_ion_trap-V_15-TAU_2.0-NL_1-IPL_1000-IUL_0-REG_1e-07-AU_0-NICS_2', 'V_15-TAU_2.0'],
        ['hqrc_pinv-RDIM_1-N_used_10000-DL_2000-Nqr_1-J_1.0-fJ_ion_trap-V_15-TAU_4.0-NL_1-IPL_1000-IUL_0-REG_1e-07-AU_0-NICS_2', 'V_15-TAU_4.0'],
        ['hqrc_pinv-RDIM_1-N_used_10000-DL_2000-Nqr_1-J_1.0-fJ_ion_trap-V_15-TAU_8.0-NL_1-IPL_1000-IUL_0-REG_1e-07-AU_0-NICS_2', 'V_15-TAU_8.0'],
        ['hqrc_pinv-RDIM_1-N_used_10000-DL_2000-Nqr_1-J_1.0-fJ_ion_trap-V_15-TAU_16.0-NL_1-IPL_1000-IUL_0-REG_1e-07-AU_0-NICS_2', 'V_15-TAU_16.0'],
        ['hqrc_pinv-RDIM_1-N_used_10000-DL_2000-Nqr_1-J_1.0-fJ_ion_trap-V_15-TAU_32.0-NL_1-IPL_1000-IUL_0-REG_1e-07-AU_0-NICS_2', 'V_15-TAU_32.0'],
    ]

    models = [[os.path.join(eval_path, m[0]), m[1]] for m in models_10000]

    rmse_dict = dict()
    vpt_dict = dict()
    targets = dict()
    outputs = dict()
    sp_outputs = dict()
    sp_targets = dict()

    # PLOTTING
    title = 'models_10000'
    cmap = plt.get_cmap("RdBu")
    ecmap = plt.get_cmap("summer_r")
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=9
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    #fig, axs = plt.subplots(1, 2, figsize=(16, 4))
    #fig.subplots_adjust(hspace=0.4, wspace = 0.4)
    #axs = axs.ravel()

    fig = plt.figure(figsize=(16, 6))
    fig.subplots_adjust(hspace=0.6, wspace = 0.2)
    ax1 = plt.subplot2grid((2,1), (0,0), colspan=1, rowspan=1)
    ax2 = plt.subplot2grid((2,1), (1,0), colspan=1, rowspan=1)
    

    for i in range(len(models)):
        rfolder, label = models[i][0], models[i][1]
        fname = os.path.join(rfolder, 'results.pickle')
        if os.path.isfile(fname):
            with open(fname, 'rb') as rfile:
                try:
                    rs = pickle.load(rfile)
                except:
                    continue
                print(rs.keys())
                qs = QResults()
                qs.rmnse_avg_test = rs['rmnse_avg_TEST']
                qs.rmnse_avg_train = rs['rmnse_avg_TRAIN']
                qs.n_pred_005_avg_test = rs['num_accurate_pred_005_avg_TEST']
                qs.n_pred_005_avg_train = rs['num_accurate_pred_005_avg_TRAIN']
                qs.n_pred_050_avg_test = rs['num_accurate_pred_050_avg_TEST']
                qs.n_pred_050_avg_train = rs['num_accurate_pred_050_avg_TRAIN']
                qs.model_name = rs['model_name']
                #if qs.rmnse_avg_test != np.inf and qs.rmnse_avg_train != np.inf:
                    #print(rs.keys())
                #print(qs.model_name)
                #print('train={}, test={}'.format(qs.rmnse_avg_train, qs.rmnse_avg_test))
                #qs.info()

                pred = rs['predictions_all_TEST'][tidx]
                truth = rs['truths_all_TEST'][tidx]
                
                mxall = rs['pre_Mx_augment_all_TEST'][tidx]
                mzall = rs['pre_Mz_augment_all_TEST'][tidx]

                mxs = np.mean(mxall, axis=1)
                mzs = np.mean(mzall, axis=1)
                #ax1.plot(range(len(mxs)), mxs, label='{}-{}'.format(j,label))
                if i == 0:
                    ax2.plot(range(len(truth)), truth, c='k', label='Target-{}'.format(tidx))
                ax2.plot(range(len(pred)), pred, label='{}-{}'.format(tidx, label))
                ax1.plot(range(len(mzs)), mzs, label='{}-{}'.format(tidx,label))
                #mxs = rs['pre_Mx_augment_TEST'].flatten()
                #mzs = rs['pre_Mz_augment_TEST'].flatten()
        else:
            print('Not found {}'.format(fname))
    ax2.set_title('Time series-{}'.format(title))
    ax1.set_title('Mz-{}'.format(title))

    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(models)//2 + 1, fontsize=8)
    #ax1.legend()
    ax1.set_ylim([-1.0, 1.0])
    ax2.set_ylim([-50, 50])
    
    outbase = 'mxz_tidx_{}_{}_v1'.format(tidx, title)
    outbase = os.path.join(fig_path, outbase)
    for ftype in ['png']:
        plt.savefig('{}_{}_v1.{}'.format(outbase, sysname, ftype), bbox_inches='tight', dpi=600)
    
    plt.show()


