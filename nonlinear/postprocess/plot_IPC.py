import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--prefix', type=str, default='degree_capa')
    parser.add_argument('--posfix', type=str, default='V_5_qrc_IPC_ion_trap_mdelay_100_mdeg_10_mvar_1_thres_0.0_T_4000000')


    args = parser.parse_args()
    print(args)
    folder, prefix, posfix = args.folder, args.prefix, args.posfix
    

    tx = list(np.arange(-7, 7.1, 0.2))
    degcapa, xs = [], []
    for x in tx:
        filename = os.path.join(folder, '{}_logtau_{:.3f}_{}.txt'.format(prefix, x, posfix))
        if os.path.isfile(filename) == False:
            print('Not found {}'.format(filename))
            continue
        deg_arr = np.loadtxt(filename)
        #print(deg_arr.shape)
        degcapa.append(deg_arr[:,1].ravel())
        xs.append(x)

    degcapa = np.array(degcapa).T
    print(degcapa.shape)
    sum_by_cols = np.sum(degcapa, axis=0)

    plt.rc('font', family='serif', size=14)
    plt.rc('mathtext', fontset='cm')
    fig = plt.figure(figsize=(24, 16), dpi=600)
    
    
    d_colors = ['#2166ac',
                '#fee090',
                '#fdbb84',
                '#fc8d59',
                '#e34a33',
                '#b30000',
                '#00706c',
                '#777777']
    N = min(len(d_colors)+1, degcapa.shape[0])

    ax1 = plt.subplot2grid((2,1), (0,0), colspan=1, rowspan=1)
    ax1.set_title('IPC by degree {}/{}'.format(folder, posfix), size=14)

    ax2 = plt.subplot2grid((2,1), (1,0), colspan=1, rowspan=1)
    ax2.set_title('Normalized IPC by degree {}/{}'.format(folder, posfix), size=14)

    ax1.bar(xs, degcapa[0])
    ax2.bar(xs, degcapa[0] / sum_by_cols)
    for i in range(1, N):
        bt = degcapa[:i].reshape(i, -1)
        bt = np.sum(bt, axis=0).ravel()
        ax1.bar(xs, degcapa[i], bottom=bt, width=0.2, label='deg-{}'.format(i), color=d_colors[i-1], edgecolor='k')
        ax2.bar(xs, degcapa[i] / sum_by_cols, bottom=bt/sum_by_cols, width=0.2, label='deg-{}'.format(i), color=d_colors[i-1], edgecolor='k')
    
    ax1.legend()
    ax2.legend()
    ax1.set_xlabel('$log_2\\tau$', size=24)
    ax2.set_xlabel('$log_2\\tau$', size=24)

    #ax1.set_xscale('log', basex=2)


    fig_folder = '{}_fig'.format(folder)
    if os.path.isdir(fig_folder) == False:
        os.mkdir(fig_folder)
    outbase = os.path.join(fig_folder, 'fig_deg_{}'.format(posfix))
    for ftype in ['png']:
        plt.savefig('{}.{}'.format(outbase, ftype), bbox_inches='tight', dpi=600)
    plt.show()



