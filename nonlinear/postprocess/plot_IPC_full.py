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
    parser.add_argument('--nspins', type=int, default=5, help='Number of spins')
    parser.add_argument('--virtuals', type=str, default='5', help='Number of virtual nodes')

    parser.add_argument('--prefix', type=str, default='ipc_capa')
    parser.add_argument('--posfix', type=str, default='seed_1_mdeg_7_mvar_7_thres_0.0_delays_0,100,50,50,20,20,10,10_T_2000000')

    parser.add_argument('--thres', type=float, default=0.0)
    parser.add_argument('--width', type=float, default=0.2)
    parser.add_argument('--max_capa', type=float, default=20.0)
    parser.add_argument('--dynamic', type=str, default='full_random')
    
    args = parser.parse_args()
    print(args)
    folder, dynamic, prefix, posfix = args.folder, args.dynamic, args.prefix, args.posfix
    V, spins, thres, width, max_capa = args.virtuals, args.nspins, args.thres, args.width, args.max_capa
    
    
    dfolder = os.path.join(folder, 'ipc')
    posfix  = 'V_{}_qrc_IPC_{}_{}'.format(V, dynamic, posfix)

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
    ax1.set_title('IPC by degree {}/{}'.format(folder, posfix), size=14)

    ax2 = plt.subplot2grid((2,1), (1,0), colspan=1, rowspan=1)
    ax2.set_title('Normalized IPC by degree {}/{}'.format(folder, posfix), size=14)
    alpha = 0.9

    ax1.bar(xs, degcapa[0], width=width, color=d_colors[0], edgecolor='k', label='deg-0', alpha=alpha)
    ax2.bar(xs, degcapa[0] / sum_by_cols,  width=0.2, color=d_colors[0], edgecolor='k', label='deg-0', alpha=alpha)
    for i in range(1, N):
        bt = degcapa[:i].reshape(i, -1)
        bt = np.sum(bt, axis=0).ravel()
        ax1.bar(xs, degcapa[i], bottom=bt, width=width, label='deg-{}'.format(i), color=d_colors[i], edgecolor='k', alpha=alpha)
        ax2.bar(xs, degcapa[i] / sum_by_cols, bottom=bt/sum_by_cols, width=0.2, label='deg-{}'.format(i), alpha=alpha, color=d_colors[i], edgecolor='k')
    
    tmin, tmax = -7, 7
    ax1.set_xlabel('$log2\\tau$', size=32)
    ax1.set_ylabel('IPC', fontsize=32)
    ax1.set_xlim([tmin, tmax])
    ax1.set_xticks(list(range(int(tmin), int(tmax)+1)))
    
    ax2.set_xlabel('$log2\\tau$', size=32)
    ax2.set_ylabel('Normalized IPC', fontsize=32)
    ax2.set_xlim([tmin, tmax])
    ax2.set_xticks(list(range(int(tmin), int(tmax)+1)))
    
    #ax1.set_xscale('log', basex=2)


    ax1.axhline(y=args.max_capa, color='k', linestyle='-')

    ax1.legend()
    ax2.legend()
    
    for ax in [ax1, ax2]:
        #ax.minorticks_on()
        ax.tick_params('both', length=8, width=1, which='major', labelsize=28)
        #ax.tick_params('both', length=4, width=1, which='minor')

    plt.tight_layout()

    fig_folder = os.path.join(folder, 'figs')
    if os.path.isdir(fig_folder) == False:
        os.mkdir(fig_folder)
    outbase = os.path.join(fig_folder, 'fig_thres_{}_{}'.format(thres, posfix))
    print('Save file', outbase)

    for ftype in ['png', 'svg']:
        plt.savefig('{}.{}'.format(outbase, ftype), bbox_inches='tight', dpi=600)
    plt.show()




