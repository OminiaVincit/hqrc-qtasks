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

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='qrep')
    parser.add_argument('--taskname', type=str, default='delay_tasks')
    parser.add_argument('--prefix', type=str, default='eig')
    parser.add_argument('--posfix', type=str, default='od_10_dl_1_delay-depolar')
    parser.add_argument('--V', type=int, default=1)
    parser.add_argument('--Nenv', type=int, default=1)
    parser.add_argument('--Nspins', type=str, default='2,3,4,5,6')
    
    parser.add_argument('--ymin', type=float, default=0.0)
    parser.add_argument('--ymax', type=float, default=1.0)
    parser.add_argument('--ptype', type=int, default=0)
    args = parser.parse_args()
    print(args)

    folder, prefix, posfix, ptype, taskname = args.folder, args.prefix, args.posfix, args.ptype, args.taskname
    V, Nenv = args.V, args.Nenv

    As = [0.2, 0.5, 1.0, 2.0]
    Bcs = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    Ns = [int(x) for x in args.Nspins.split(',')]

    Ma, Mbc = len(As), len(Bcs)
    ymin, ymax = args.ymin, args.ymax
    #cmap = plt.get_cmap("YlGnBu_r")
    if Nenv == 1:
        cmap = plt.get_cmap("RdBu")
    elif Nenv == 2:
        cmap = plt.get_cmap("Spectral")
    else:
        cmap = plt.get_cmap("PRGn")
    
    fig, axs = plt.subplots(Mbc, Ma, figsize=(6*Mbc, 6*Ma))
    #axs = np.array(axs).ravel()
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=16

    ntitle = 'abc_{}_Nspins_{}_Nenv_{}_V_{}_{}'.format(prefix,\
        '_'.join([str(x) for x in Ns]), Nenv, V, posfix)

    lo, hi = 10000, 0
    for i in range(Mbc):
        for j in range(Ma):
            a, bc = As[j], Bcs[i]
            ax = axs[i, j]
            ax.set_xlabel('$\\tau B$', fontsize=24)
            if ptype > 0:
                ax.set_ylabel('RMSF', fontsize=16)
            else:
                ax.set_ylabel('$N_m$', fontsize=24)
            if ptype > 0 and ymin < ymax:
                ax.set_ylim([ymin, ymax])
            ax.set_title('$\\alpha$={}, $J_b/B$={}'.format(a, bc), fontsize=24)
            if ptype > 0:
                ax.grid(True, which="major", ls="-", color='0.65')
            fidarr = []
            for Nspin in Ns:
                subfolder = os.path.join(folder, taskname)
                subfolder = os.path.join(subfolder, '{}_a_{}_bc_{}_{}_{}_{}'.format(prefix, a, bc, Nspin, Nenv, posfix))
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
                    if len(xs) == 0:
                        continue
                    lo = min(lo, np.min(xs))
                    hi = max(hi, np.max(xs))
                    sids = np.argsort(xs)
                    xs, ys, zs = xs[sids], ys[sids], zs[sids]
                    fidarr.append(ys)
                    # Plot
                    if ptype > 0:
                        ax.plot(xs, ys, alpha = 0.8, linewidth=3.0, marker='o', markersize=6, mec='k', mew=0.5, label='$N_m$={}'.format(Nspin - Nenv))
                    # ax.errorbar(xs, ys, yerr=zs, alpha = 0.8, marker='o', elinewidth=2, linewidth=2, markersize=6, \
                    #     mec='k', mew=0.5, label='$N_m$={}'.format(Nspin - Nenv))
            if len(fidarr) == 0:
                continue
            fidarr = np.array(fidarr)
            scale = 3
            extent = [lo, hi, 0, 10]
            print(extent)
            if ptype == 0:
                im = ax.imshow(fidarr, origin='lower', cmap=cmap, vmin=ymin, vmax=ymax, extent=extent)
                
                urange = np.linspace(1, 9, len(Ns))
                vrange = ['{}'.format(x-Nenv) for x in Ns]
                ax.set_yticks(urange)
                ax.set_yticklabels(vrange, fontsize=16)
            else:
                ax.legend()
            ax.tick_params('both', length=10, width=1, which='major')
            xticklist = np.linspace(0, 24.0, num=13)
            ax.set_xlim([0, 25.0])
            ax.set_xticks(xticklist)
            ax.set_xticklabels(labels=['{:.0f}'.format(x) for x in  xticklist], fontsize=16)
    
    fig_folder = os.path.join(folder, 'figs')
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
    