import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import plotutils as plu
import re
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


if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default=MEM_FUNC_DATA)
    parser.add_argument('--ymin', type=float, default='0.0')
    parser.add_argument('--ymax', type=float, default='1.0')
    parser.add_argument('--tmin', type=float, default=0.0, help='Minimum of tauB')
    parser.add_argument('--tmax', type=float, default=5.0, help='Maximum of tauB')
    parser.add_argument('--ntaus', type=int, default=50, help='Number of tausB')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--bcoef', type=float, default=2.0)
    parser.add_argument('--thres', type=float, default=0.0)
    parser.add_argument('--V', type=int, default=1)
    parser.add_argument('--width', type=float, default=1.0)
    parser.add_argument('--Nspins', type=str, default='3,4,5,6')
    parser.add_argument('--nenv', type=int, default=2)
    parser.add_argument('--prefix', type=str, default='quanrc_ion_trap_nspins')
    parser.add_argument('--posfix', type=str, default='len_1000_3000_100_trials_5')
    
    args = parser.parse_args()
    print(args)

    folder, prefix, posfix = args.folder, args.prefix, args.posfix
    ymin, ymax = args.ymin, args.ymax
    tmin, tmax, ntaus = args.tmin, args.tmax, args.ntaus
    alpha, bc, width, nenv = args.alpha, args.bcoef, args.width, args.nenv
    tauBs = list(np.linspace(tmin, tmax, ntaus + 1))
    vals = tauBs[1:]

    Ns = [int(x) for x in args.Nspins.split(',')]

    xticks = np.linspace(0, 5, 26)
    xticklabels = ['{:.1f}'.format(t) for t in xticks]
    cmap = plt.get_cmap("nipy_spectral")
    Vs = [1]
    fig, axs = plt.subplots(len(Vs), 1, figsize=(20, 6), squeeze=False)
    axs = axs.ravel()
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=24

    ntitle = 'nspins_{}_a_{}_bc_{}_{}_thres_{}'.format(prefix, alpha, bc, posfix, args.thres)
    #xticks = range(4, ntaus, 5)
    #xticklabels = ['{:.2f}'.format((x+1)/20) for x in xticks]
    
    xticks = np.linspace(tmin, tmax, 25+1)
    xticklabels = ['{:.1f}'.format(x) for x in xticks]
    
    for i in range(len(Vs)):
        V = Vs[i]
        ax = axs[i]
        spinarr = []

        for nspin in Ns:
            Vposfix = 'tmax_{}_tmin_{}_ntaus_{}_Vs_{}_{}'.format(tmax, tmin, ntaus, V, posfix)
            ts, mcs_avg, mcs_std = [], [], []
            valbase = '{}_{}_{}_a_{:.1f}_bc_{:.1f}_{}.log'.format(prefix, nspin, nenv, alpha, bc, Vposfix)
            logfile = os.path.join(folder, 'log')
            logfile = os.path.join(logfile, valbase)
            if os.path.isfile(logfile) == False:
                print('Not found {}'.format(logfile))
                continue
            with open(logfile, 'r') as rf:
                lines = rf.readlines()
                for line in lines:
                    if 'tauB=' in line:
                        tauB = float(re.search(r"tauB=([0-9.]+)", line).group(1))
                        capa_avg = float(re.search(r"capa_avg=([0-9.]+)", line).group(1))
                        capa_std = float(re.search(r"capa_std=([0-9.]+)", line).group(1))
                        ts.append(tauB)
                        mcs_avg.append(capa_avg)
                        mcs_std.append(capa_std)
            if len(mcs_avg) == 0:
                continue
            mcs_avg, mcs_std = np.array(mcs_avg), np.array(mcs_std)
            ax.fill_between(ts, mcs_avg - mcs_std, mcs_avg + mcs_std, facecolor='gray', alpha=0.5)
            ax.plot(ts, mcs_avg, alpha=0.8, marker='o', markeredgecolor='k', \
                markersize=0, linewidth=5, label='$N_m$={}'.format(nspin - nenv))
            
        ax.legend()
        ax.set_xticks(xticks)
        #ax.set_xticklabels(xticklabels, fontsize=16)
        ax.set_xlabel('$\\tau B$', fontsize=24)
        ax.set_ylabel('QMC', fontsize=24)
        ax.tick_params(axis='y', labelsize=18, length=10, width=2)
        ax.tick_params(axis='x', labelsize=18, length=10, width=2)
        ax.set_xlim([tmin, tmax])
        #fig.colorbar(im, ax=ax)

    fig_folder = os.path.join(folder, 'figs')
    if os.path.isdir(fig_folder) == False:
        os.mkdir(fig_folder)
    

    outbase = '{}/{}'.format(fig_folder, ntitle)
    #plt.suptitle(outbase, fontsize=12)
    plt.tight_layout()
    #fig.colorbar(im, ax=ax, orientation="horizontal")
    for ftype in ['png', 'pdf', 'svg']:
        print('Save file {}'.format(outbase))
        plt.savefig('{}_qmc.{}'.format(outbase, ftype), bbox_inches='tight')
    plt.show()
    