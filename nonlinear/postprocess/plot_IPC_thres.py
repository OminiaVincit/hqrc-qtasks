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
    parser.add_argument('--thres', type=float, default=0.0)
    parser.add_argument('--max_capa', type=float, default=4.0)
    
    parser.add_argument('--prefix', type=str, default='ipc_capa')
    parser.add_argument('--posfix', type=str, default='V_5_qrc_IPC_ion_trap_mdelay_100_mdeg_10_mvar_1_thres_0.0_T_4000000')


    args = parser.parse_args()
    print(args)
    folder, prefix, posfix, thres = args.folder, args.prefix, args.posfix, args.thres
    
    dfolder = os.path.join(folder, 'ipc')
    tx = list(np.arange(-7, 7.1, 0.2))
    degcapa, xs = [], []
    for x in tx:
        tarr = []
        filename = os.path.join(dfolder, '{}_logtau_{:.3f}_{}.pickle'.format(prefix, x, posfix))
        if os.path.isfile(filename) == False:
            print('Not found {}'.format(filename))
            continue
        with open(filename, "rb") as rfile:
            data = pickle.load(rfile)
        ipc_arr = data['ipc_arr']
        for deg in sorted(ipc_arr.keys()):
            darr = ipc_arr[deg].ravel()
            tarr.append( np.sum(darr[darr >= thres]) )
        #print(deg_arr.shape)
        degcapa.append(np.array(tarr).ravel())
        xs.append(x)

    degcapa = np.array(degcapa).T
    print(degcapa.shape)
    sum_by_cols = np.sum(degcapa, axis=0)

    plt.rc('font', family='serif', size=14)
    plt.rc('mathtext', fontset='cm')
    fig = plt.figure(figsize=(24, 16), dpi=600)
    
    
    d_colors = ['#777777',
                '#2166ac',
                '#fee090',
                '#fdbb84',
                '#fc8d59',
                '#e34a33',
                '#b30000',
                '#00706c'
                ]
    N = min(len(d_colors), degcapa.shape[0])

    ax1 = plt.subplot2grid((2,1), (0,0), colspan=1, rowspan=1)
    ax1.set_title('IPC by degree {}/THRES_{}_{}'.format(folder, thres, posfix), size=14)

    ax2 = plt.subplot2grid((2,1), (1,0), colspan=1, rowspan=1)
    ax2.set_title('Normalized IPC by degree {}/THRES_{}_{}'.format(folder, thres, posfix), size=14)

    ax1.bar(xs, degcapa[0], width=0.2, color=d_colors[0], edgecolor='k', label='deg-0')
    ax2.bar(xs, degcapa[0] / sum_by_cols,  width=0.2, color=d_colors[0], edgecolor='k', label='deg-0')
    for i in range(1, N):
        bt = degcapa[:i].reshape(i, -1)
        bt = np.sum(bt, axis=0).ravel()
        ax1.bar(xs, degcapa[i], bottom=bt, width=0.2, label='deg-{}'.format(i), color=d_colors[i], edgecolor='k')
        ax2.bar(xs, degcapa[i] / sum_by_cols, bottom=bt/sum_by_cols, width=0.2, label='deg-{}'.format(i), color=d_colors[i], edgecolor='k')
    
    ax1.legend()
    ax2.legend()
    ax1.set_xlabel('$log_2\\tau$', size=24)
    ax2.set_xlabel('$log_2\\tau$', size=24)
    
    #ax1.set_ylim([0, 4.0])
    #ax1.set_xscale('log', basex=2)
    ax1.axhline(y=args.max_capa, color='k', linestyle='-')

    fig_folder = os.path.join(folder, 'fig')
    if os.path.isdir(fig_folder) == False:
        os.mkdir(fig_folder)
    outbase = os.path.join(fig_folder, 'fig_thres_{}_{}'.format(thres, posfix))
    for ftype in ['png']:
        plt.savefig('{}.{}'.format(outbase, ftype), bbox_inches='tight', dpi=600)
    plt.show()



