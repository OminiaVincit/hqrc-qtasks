import sys
import os
import glob
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotutils as plu
# Plot for quantum memory
MEM_FUNC_DATA='/data/zoro/qrep/quan_capa_abc'

cycle = [
'#e41a1c',
'#377eb8',
'#4daf4a',
'#984ea3',
'#ff7f00',
'#ffff33'
]

def plotContour(fig, ax, data, title, fontsize, vmin, vmax, cmap):
    ax.set_title(title, fontsize=fontsize)
    t, s = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
    mp = ax.contourf(s, t, np.transpose(data), 15, cmap=cmap, levels=np.linspace(vmin, vmax, 60), extend="both", zorder=-20)
    #fig.colorbar(mp, ax=ax)
    ax.set_rasterization_zorder(-10)
    #ax.set_xlabel(r"Time", fontsize=fontsize)
    return mp

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default=MEM_FUNC_DATA)
    parser.add_argument('--ymin', type=float, default='0.0')
    parser.add_argument('--ymax', type=float, default='1.0')

    parser.add_argument('--amin', type=float, default=0.0, help='Minimum of alpha')
    parser.add_argument('--amax', type=float, default=2.0, help='Maximum of alpha')
    parser.add_argument('--nas', type=int, default=20, help='Number of alpha')

    parser.add_argument('--bcmin', type=float, default=0.0, help='Minimum of bc')
    parser.add_argument('--bcmax', type=float, default=2.0, help='Maximum of bc')
    parser.add_argument('--nbcs', type=int, default=40, help='Number of bc')

    parser.add_argument('--tauB', type=float, default=5.0)
    parser.add_argument('--thres', type=float, default=0.0)
    parser.add_argument('--V', type=int, default=1)
    parser.add_argument('--nspins', type=int, default=6)
    parser.add_argument('--nenvs', type=int, default=2)

    parser.add_argument('--dmax', type=int, default=10)
    parser.add_argument('--nticks', type=int, default=20, help='Number of xticks')
    parser.add_argument('--prefix', type=str, default='quanrc_ion_trap_nspins')
    parser.add_argument('--posfix', type=str, default='len_1000_3000_100_dmax_10')
    parser.add_argument('--ntrials', type=int, default=10)
    args = parser.parse_args()
    print(args)

    folder, prefix, posfix = args.folder, args.prefix, args.posfix
    
    posfix = 'V_{}_{}'.format(args.V, posfix)
    ymin, ymax, dmax = args.ymin, args.ymax, args.dmax
    amin, amax, nas = args.amin, args.amax, args.nas
    bcmin, bcmax, nbcs = args.bcmin, args.bcmax, args.nbcs
    
    tauB = args.tauB
    nspins, nenvs, ntrials, nticks = args.nspins, args.nenvs, args.ntrials, args.nticks
    
    als = list(np.linspace(amin, amax, nas + 1))
    als = als[1:]
    
    bcls = list(np.linspace(bcmin, bcmax, nbcs + 1))
    bcls = bcls[1:]

    binfolder = os.path.join(folder, 'binary')
    
    cmap = plt.get_cmap("rainbow")
    fig, axs = plt.subplots(2, 1, figsize=(20, 12), squeeze=False)
    axs = axs.ravel()
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=16

    ntitle = '{}_nspins_{}_{}_tauB_{}_thres_{}_{}_tod_{}'.format(prefix, nspins, nenvs, tauB, args.thres, posfix, dmax)
    sprefix = '{}_{}_{}'.format(prefix, nspins, nenvs)
    
    ax1, ax2 = axs[0], axs[1]
    memarr = []
    dcl = 0
    for a in als:
        mcs_avg, mcs_std, ts = [], [], []
        for bc in bcls:
            valbase = '{}_a_{:.3f}_bc_{:.3f}_tauB_{:.3f}_{}'.format(sprefix, a, bc, tauB, posfix)
            loc_arr = []
            loc_sum = []
            for n in range(ntrials):
                memfile = os.path.join(binfolder, '{}_trial_{}.npy'.format(valbase, n))
                if os.path.isfile(memfile) == False:
                    print('Not found {}'.format(memfile))
                    continue
                arr = np.load(memfile)
                print('read {} with shape'.format(memfile), arr.shape)
                loc_arr.append(arr[:, 1])
                loc_sum.append(np.sum(arr[:(dmax+1), 1]))
            loc_sum = np.array(loc_sum)
            loc_arr = np.array(loc_arr)

            mcs_avg.append(np.mean(loc_sum))
            mcs_std.append(np.std(loc_sum))
            ts.append(bc)
        
        if len(mcs_avg) == 0:
            continue
        mcs_avg, mcs_std = np.array(mcs_avg), np.array(mcs_std)
        
        if a in [0.2, 0.5, 1.0, 2.0]:
            color = cycle[dcl]
            dcl += 1
            ax1.fill_between(ts, mcs_avg - mcs_std, mcs_avg + mcs_std, facecolor=color, alpha=0.2)
            ax1.plot(ts, mcs_avg, alpha=0.8, marker='o', markeredgecolor='k', color=color,\
                markersize=0, linewidth=4, label='$\\alpha$={}'.format(a))
        memarr.append(mcs_avg)
    
    ax1.grid(axis='x')
    ax1.legend()
    ax1.set_title(ntitle)
    xticks = np.linspace(bcmin, bcmax, nticks+1)
    xticklabels = ['{:.1f}'.format(t) for t in xticks]
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(labels=xticklabels)
    ax1.set_xlim([ts[0], ts[-1]])

    memarr = np.array(memarr)
    
    # Plot MC heatmap contour
    ymin, ymax = np.min(memarr), np.max(memarr)
    im = plotContour(fig, ax2, memarr, '{}'.format(ntitle), 16, ymin, ymax, cmap)
    fig.colorbar(im, ax=ax2, orientation="horizontal")
    xticks = np.linspace(1, nbcs-1, nticks)
    xticklabels = ['{:.1f}'.format((t+1)/20) for t in xticks]
    yticks = range(4, nas, 5)
    yticklabels = ['{:.1f}'.format((t+1)/10) for t in yticks]

    ax2.set_xticks(xticks)
    ax2.set_xticklabels(labels=xticklabels)
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(labels=yticklabels)

    ax2.set_xlim([0, nbcs-1])
    for bx in axs:
        bx.tick_params('both', length=10, width=1.0, which='major', labelsize=20)
        #bx.set_xlim([ts[0], ts[-1]])

    fig_folder = os.path.join(folder, 'figs')
    if os.path.isdir(fig_folder) == False:
        os.mkdir(fig_folder)

    outbase = '{}/{}'.format(fig_folder, ntitle)
    #plt.suptitle(outbase, fontsize=12)
    plt.tight_layout()
    #fig.colorbar(im, ax=ax, orientation="horizontal")
    for ftype in ['png', 'pdf', 'svg']:
        print('Save file {}'.format(outbase))
        plt.savefig('{}_memabc.{}'.format(outbase, ftype), bbox_inches='tight')
    plt.show()
    