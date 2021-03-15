import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
#import holoviews as hv
#hv.extension('matplotlib')
#plt.set_cmap(cmap='nipy_spectral')
import re
import colorcet as cc

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
    parser.add_argument('--folder', type=str, default='qrep')
    parser.add_argument('--taskname', type=str, default='delay_tasks2')
    parser.add_argument('--prefix', type=str, default='eig_a_1.0_bc_1.0')
    parser.add_argument('--posfix', type=str, default='od_10_dl_1_delay-depolar')
    parser.add_argument('--Vs', type=str, default='1,5')
    parser.add_argument('--Nenv', type=int, default=2)
    parser.add_argument('--Nspins', type=str, default='3,4,5,6,7')
    
    parser.add_argument('--ymin', type=float, default=0.0)
    parser.add_argument('--ymax', type=float, default=1.0)
    parser.add_argument('--ptype', type=int, default=0)
    args = parser.parse_args()
    print(args)

    folder, prefix, posfix, ptype, taskname = args.folder, args.prefix, args.posfix, args.ptype, args.taskname
    
    Nenv = args.Nenv

    Vs = [int(x) for x in args.Vs.split(',')]
    Ns = [int(x) for x in args.Nspins.split(',')]

    M = len(Vs)
    ymin, ymax = args.ymin, args.ymax
    #cmap = plt.get_cmap("YlGnBu_r")
    if Nenv == 1:
        cmap = plt.get_cmap("RdBu")
    elif Nenv == 2:
        cmap = plt.get_cmap("Spectral")
    else:
        cmap = plt.get_cmap("PRGn")
    fig, axs = plt.subplots(M, 1, figsize=(20, 4*M), sharex=True)
    axs = np.array(axs).ravel()
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=16

    ntitle = 'single_{}_Nspins_{}_Nenv_{}_Vs_{}_{}'.format(prefix,\
        '_'.join([str(x) for x in Ns]), Nenv,\
        '_'.join([str(x) for x in Vs]), posfix)

    lo, hi = 10000, 0
    for j in range(M):
        V = Vs[j]
        ax = axs[j]
        ax.set_xlabel('$\\tau B$', fontsize=24)
        if ptype == 1:
            ax.set_ylabel('RMSF', fontsize=16)
        else:
            ax.set_ylabel('$N_m$', fontsize=24)
        if ptype > 0 and ymin < ymax:
            ax.set_ylim([ymin, ymax])
        ax.set_title('$M$={}, {}'.format(V, ntitle), fontsize=16)
        if ptype > 0:
            ax.grid(True, which="major", ls="-", color='0.65')
        fidarr = []
        for Nspin in Ns:
            subfolder = os.path.join(folder, taskname)
            subfolder = os.path.join(subfolder, '{}_{}_{}_{}'.format(prefix, Nspin, Nenv, posfix))
            subfolder = os.path.join(subfolder, 'log')
            print(subfolder)
            if os.path.isdir(subfolder) == False:
                continue
            for rfile in glob.glob('{}/qtasks_*_nspins_{}_{}_*_Vs_{}_*.log'.format(subfolder, Nspin, Nenv, V)):
                print(rfile)
                xs, ys, zs = [], [], []
                with open(rfile, mode='r') as rf:
                    lines = rf.readlines()
                    for line in lines:
                        if 'INFO' in line and 'Average RMSF' in line and 'avg-' in line:
                            tauB = float(re.search(r"tauB=([0-9.]+)", line).group(1))
                            avg_val_fid = float(re.search(r"avg-val=([0-9.]+)", line).group(1))
                            std_val_fid = float(re.search(r"std-val=([0-9.]+)", line).group(1))
                            xs.append(tauB)
                            ys.append(avg_val_fid)
                            zs.append(std_val_fid)
                xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
                lo = min(lo, np.min(xs))
                hi = max(hi, np.max(xs))
                sids = np.argsort(xs)
                xs, ys, zs = xs[sids], ys[sids], zs[sids]
                fidarr.append(ys)
                # Plot
                if ptype == 1:
                    ax.plot(xs, ys, alpha = 0.8, linewidth=3.0, marker='o', markersize=6, mec='k', mew=0.5, label='$N_m$={}'.format(Nspin - Nenv))
                # ax.errorbar(xs, ys, yerr=zs, alpha = 0.8, marker='o', elinewidth=2, linewidth=2, markersize=6, \
                #     mec='k', mew=0.5, label='$N_m$={}'.format(Nspin - Nenv))
        fidarr = np.array(fidarr)
        print('Fidarr shape', fidarr.shape)
        scale = 3
        extent = [10*lo, 10*hi, 0, 10]
        print(extent)
        if ptype is not 1:
            if len(fidarr) > 0:
                if ptype == 3:
                    im = plotContour(fig, ax, fidarr, '$N_e = {}, M = {}$'.format(Nenv, V), 16, ymin, ymax, cmap)
                else:
                    im = ax.imshow(fidarr, origin='lower', cmap=cmap, vmin=ymin, vmax=ymax, extent=extent)
            
            urange = np.linspace(1, 9, len(Ns))
            vrange = ['{}'.format(x-Nenv) for x in Ns]
            if ptype == 0:
                ax.set_yticks(urange)
            ax.set_yticklabels(vrange, fontsize=20)
            xticklist = np.linspace(0, 50.0, num=21)
            ax.set_xticks(xticklist)
            ax.set_xticklabels(labels=['{:.1f}'.format(x/5) for x in  xticklist], fontsize=18)
            ax.set_xlim([0.0, 50.0])
        else:
            ax.legend()
        ax.tick_params('both', length=10, width=1, which='major')
        ax.set_title('$N_e = {}$, Multiplex = ${}$'.format(Nenv, V), fontsize=24)
    
    fig_folder = os.path.join(folder, taskname)
    fig_folder = os.path.join(fig_folder, 'figs')
    if os.path.isdir(fig_folder) == False:
        os.mkdir(fig_folder)

    outbase = '{}/{}'.format(fig_folder, ntitle)
    #plt.suptitle(outbase, fontsize=12)
    plt.tight_layout()

    if ptype == 0:
        fig.colorbar(im, ax=axs, orientation="vertical")
    
    for ftype in ['png', 'svg', 'pdf']:
        transparent = True
        if ftype == 'png':
            transparent = False
        plt.savefig('{}_fidelity_{}.{}'.format(outbase, ptype, ftype), bbox_inches='tight', transparent=transparent, dpi=600)
    plt.show()
    