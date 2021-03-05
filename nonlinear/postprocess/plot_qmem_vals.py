import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import plotutils as plu
# Plot for quantum memory
MEM_FUNC_DATA='/data/zoro/qrep/quan_capacity'

def plotContour(fig, ax, data, title, fontsize, vmin, vmax, cmap):
    ax.set_title(title, fontsize=fontsize)
    t, s = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
    mp = ax.contourf(s, t, np.transpose(data), 15, cmap=cmap, levels=np.linspace(vmin, vmax, 60), extend="both", zorder=-20)
    #fig.colorbar(mp, ax=ax)
    ax.set_rasterization_zorder(-10)
    #ax.set_xlabel(r"Time", fontsize=fontsize)
    return mp

    
d_colors = ['#777777',
            '#2166ac',
            '#fee090',
            '#fdbb84',
            '#fc8d59',
            '#e34a33',
            '#b30000',
            '#00706c'
            ]

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default=MEM_FUNC_DATA)
    parser.add_argument('--ymin', type=float, default='0.0')
    parser.add_argument('--ymax', type=float, default='1.0')
    parser.add_argument('--tmin', type=float, default=0.0, help='Minimum of tauB')
    parser.add_argument('--tmax', type=float, default=25.0, help='Maximum of tauB')
    parser.add_argument('--ntaus', type=int, default=25, help='Number of tausB')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--bcoef', type=float, default=2.0)
    parser.add_argument('--thres', type=float, default=1e-2)
    parser.add_argument('--V', type=int, default=1)
    parser.add_argument('--width', type=float, default=1.0)
    parser.add_argument('--nspins', type=int, default=5)
    parser.add_argument('--nenvs', type=int, default=1)
    parser.add_argument('--prefix', type=str, default='quanrc_ion_trap_nspins')
    parser.add_argument('--posfix', type=str, default='len_1000_3000_100_trials_5')
    
    args = parser.parse_args()
    print(args)

    folder, prefix, posfix = args.folder, args.prefix, args.posfix
    prefix = '{}_{}_{}'.format(prefix, args.nspins, args.nenvs)
    posfix = 'V_{}_{}'.format(args.V, posfix)
    ymin, ymax = args.ymin, args.ymax
    tmin, tmax, ntaus = args.tmin, args.tmax, args.ntaus
    alpha, bc, width = args.alpha, args.bcoef, args.width
    tauBs = list(np.linspace(tmin, tmax, ntaus + 1))
    tauBs = tauBs[1:]

    if alpha == 0.0:
        # variable = a
        vals = np.arange(0.1, 2.1, 0.1)
        vlabel = '$\\alpha$'
        xticks = np.linspace(0.0, 2.0, 21)
        xticklabels = ['{:.1f}'.format(t) for t in xticks]
        xticks2 = np.arange(0.0, 2.1, 0.2)
    elif bc == 0.0:
        #vals = np.arange(0.2, 5.1, 0.2)
        #yticks = [4, 9, 14, 19]
        #yticklabels = ['{:.1f}'.format((t+1)/5) for t in yticks]
        vals = np.arange(0.02, 2.21, 0.02)
        vlabel = '$J_b/B$'
        xticks = np.linspace(0, 2.2, 23)
        xticklabels = ['{:.1f}'.format(t) for t in xticks]
        xticks2 = np.linspace(0, 2.2, 23)
    else:
        vals = tauBs
        vlabel = '$\\tau B$'
        #xticks = range(0, ntaus, 5)
        xticks = np.linspace(0, 5, 26)
        xticklabels = ['{:.1f}'.format(t) for t in xticks]
        xticks2 = np.arange(0.0, 25.1, 5.0)
    cmap = plt.get_cmap("twilight")
    fig, axs = plt.subplots(3, 1, figsize=(20, 12), gridspec_kw={'height_ratios': [3, 2, 2]})
    ax = axs[0]
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=16

    ntitle = '{}_a_{}_bc_{}_{}_thres_{}'.format(prefix, alpha, bc, posfix, args.thres)
    memarr, ts, mcs = [], [], []
    for val in vals:
        if alpha == 0.0:
            valbase = '{}_a_{:.2f}_bc_{:.1f}_tauB_{:.3f}_{}.txt'.format(prefix, val, bc, tauBs[0], posfix)
            if os.path.isfile(os.path.join(folder, valbase)) == False:
                valbase = '{}_a_{:.1f}_bc_{:.1f}_tauB_{:.3f}_{}.txt'.format(prefix, val, bc, tauBs[0], posfix)
        elif bc == 0.0:
            valbase = '{}_a_{:.1f}_bc_{:.2f}_tauB_{:.3f}_{}.txt'.format(prefix, alpha, val, tauBs[0], posfix)
            if os.path.isfile(os.path.join(folder, valbase)) == False:
                valbase = '{}_a_{:.1f}_bc_{:.1f}_tauB_{:.3f}_{}.txt'.format(prefix, alpha, val, tauBs[0], posfix)
        else:
            valbase = '{}_a_{:.1f}_bc_{:.1f}_tauB_{:.3f}_{}.txt'.format(prefix, alpha, bc, val, posfix)
        memfile = os.path.join(folder, valbase)
        if os.path.isfile(memfile) == False:
            print('Not found {}'.format(memfile))
            continue
        arr = np.loadtxt(memfile)
        print('read {} with shape'.format(memfile), arr.shape)
        loc_arr = arr[:, 1]
        memarr.append(loc_arr)
        # loc_arr = loc_arr - np.min(loc_arr)
        loc_arr[loc_arr < args.thres] = 0.0
        mcs.append(np.sum(loc_arr))
        ts.append(val)
    memarr = np.array(memarr).T
    ymin, ymax = np.min(memarr), np.max(memarr)
    #im = plotContour(fig, ax, memarr.T, '{}'.format(ntitle), 16, ymin, ymax, cmap)
    ax.bar(ts, memarr[0], width=width, color=d_colors[0], edgecolor='k', label='d=0')
    for i in range(1, len(d_colors)):
        bt = memarr[:i].reshape(i, -1)
        bt = np.sum(bt, axis=0).ravel()
        ax.bar(ts, memarr[i], bottom=bt, width=width, label='d={}'.format(i), color=d_colors[i], edgecolor='k', alpha=0.7)
    
    ax.set_xlim(vals[0], vals[-1])
    ax.legend(loc='upper right')
    #ax.set_ylabel('QMC', fontsize=24)
    ax.set_xlabel(vlabel, fontsize=24)
    #ax.set_ylim([0, 5])
    #ax.set_ylim([np.min(avg_tests)/2, 2*np.max(avg_tests)])
    #ax.set_yscale('log')
    #ax.set_ylim([5*10**(-2), 10**0])
    
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels=xticklabels)
    #ax.set_yticklabels(labels='')
    #ax.set_title('{}'.format(ntitle), fontsize=12)
    #ax.grid(True, which="both", ls="-", color='0.65')
    #ax.legend()

    for i in range(1, 3):
        if i == 2:
            # normalize mcs
            mcs = mcs / np.max(mcs)
            lb = 'NQMC'
        else:
            lb = 'QMC'
        ax2 = axs[i]
        ax2.set_ylabel(lb, fontsize=24)
        #ax2.set_xlabel(vlabel, fontsize=24)
        #ax2.barh(ts, mcs, height=1.0, edgecolor='k', alpha=0.8)
        ax2.plot(ts, mcs, alpha=0.8, marker='o', markeredgecolor='k', \
            markersize=10, linewidth=3, markerfacecolor=plu.VERMILLION)
        ax2.set_xlim(vals[0], vals[-1])
        ax2.set_xticks(xticks2)
    

    for bx in axs:
        bx.tick_params(axis='both', which='major', labelsize=16)
        bx.tick_params(axis='both', which='minor', labelsize=12)
        bx.tick_params('both', length=10, width=1.0, which='major')

    fig_folder = os.path.join(folder, 'figs')
    if os.path.isdir(fig_folder) == False:
        os.mkdir(fig_folder)

    outbase = '{}/{}'.format(fig_folder, ntitle)
    #plt.suptitle(outbase, fontsize=12)
    plt.tight_layout()
    #fig.colorbar(im, ax=ax, orientation="horizontal")
    for ftype in ['png', 'pdf', 'svg']:
        print('Save file {}'.format(outbase))
        plt.savefig('{}_func.{}'.format(outbase, ftype), bbox_inches='tight')
    plt.show()
    