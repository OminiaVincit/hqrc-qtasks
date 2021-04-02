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
import plotutils as putils

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='qrep')
    parser.add_argument('--taskname', type=str, default='delay_tasks2')
    parser.add_argument('--prefix', type=str, default='eig')
    parser.add_argument('--posfix', type=str, default='od_10_dl_1_delay-depolar')
    parser.add_argument('--Vs', type=str, default='1,5')
    
    parser.add_argument('--valmax', type=float, default=10.0, help='Maximum val')
    parser.add_argument('--valmin', type=float, default=0.0, help='Minimum val')
    parser.add_argument('--nvals', type=int, default=100, help='Number of bc')
    
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--bc', type=float, default=1.0)

    parser.add_argument('--Nenvs', type=str, default='1,2,3')
    parser.add_argument('--Nms', type=str, default='4,5')
    parser.add_argument('--Nticks', type=int, default=20, help='Number of xticks')
    
    parser.add_argument('--ymin', type=float, default=0.9)
    parser.add_argument('--ymax', type=float, default=1.0)
    args = parser.parse_args()
    print(args)

    folder, prefix, posfix, taskname = args.folder, args.prefix, args.posfix, args.taskname
    valmin, valmax, nvals, nticks = args.valmin, args.valmax, args.nvals, args.Nticks
    alpha, bc = args.alpha, args.bc

    tauBs = list(np.linspace(valmin, valmax, nvals + 1))
    tauBs = tauBs[1:]

    Vs = [int(x) for x in args.Vs.split(',')]
    
    Nms = [int(x) for x in args.Nms.split(',')]
    Nenvs = [int(x) for x in args.Nenvs.split(',')]

    ymin, ymax = args.ymin, args.ymax

    fig, axs = plt.subplots(1, 1, figsize=(20, 6), squeeze=False)
    axs = np.array(axs).ravel()
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=24

    ntitle = 'taus_{}_a_{}_bc_{}_Vs_{}_Nenvs_{}_Nms_{}_{}'.format(prefix, alpha, bc, \
        '_'.join([str(v) for v in Vs]), \
        '_'.join([str(v) for v in Nenvs]), \
        '_'.join([str(v) for v in Nms]), posfix)

    ax = axs[0]
    ax.set_xlabel('$\\tau B$', fontsize=24)
    ax.set_ylabel('Error', fontsize=24)
    ax.set_xlim([valmin, valmax])
    if ymin < ymax:
        ax.set_ylim([ymin, ymax])
    ax.set_title('{}'.format(ntitle), fontsize=16)
    ax.grid(True, which="major", ls="-", color='0.65')

    dcl = 0
    for Nenv in Nenvs:
        for Nm in Nms:
            Nspin = Nenv + Nm
            for V in Vs:
                xs, ys, zs = [], [], []
                subfolder = os.path.join(folder, taskname)
                subfolder = os.path.join(subfolder, '{}_a_{}_bc_{}_{}_{}_{}'.format(prefix, alpha, bc, Nspin, Nenv, posfix))
                subfolder = os.path.join(subfolder, 'log')
                print(subfolder)
                if os.path.isdir(subfolder) == False:
                    continue
                for rfile in sorted(glob.glob('{}/qtasks_*_nspins_{}_{}_*_Vs_{}_*.log'.format(subfolder, Nspin, Nenv, V)), key=os.path.basename):
                    print(rfile)
                    if 'corr_0' in rfile:
                        K = Nm
                    elif 'corr' in rfile:
                        K = int(Nm*(Nm+1) / 2)
                    else:
                        K = Nm
                    xs, ys, zs = [], [], []
                    with open(rfile, mode='r') as rf:
                        lines = rf.readlines()
                        for line in lines:
                            if 'INFO' in line and 'Average RMSF' in line and 'avg-' in line:
                                tauB = float(re.search(r"tauB=([0-9.]+)", line).group(1))
                                avg_val_fid = float(re.search(r"avg-val=([0-9.]+)", line).group(1))
                                std_val_fid = float(re.search(r"std-val=([0-9.]+)", line).group(1))
                                xs.append(tauB)
                                ys.append(1.0 - avg_val_fid)
                                zs.append(std_val_fid)
                    xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
                    sids = np.argsort(xs)
                    xs, ys, zs = xs[sids], ys[sids], zs[sids]
                    # Plot
                    if len(xs) > 0:
                        ax.plot(xs, ys, alpha = 0.8, marker='o', markeredgecolor='k', \
                            linewidth=4, markersize=0, color=putils.cycle[dcl],\
                            label='$(N_e,N_m)=({}, {}), (M,K)=({},{})$'.format(Nenv, Nm, V, K))
                        bg = 0
                        ax.fill_between(xs[bg:], ys[bg:] - zs[bg:], ys[bg:] + zs[bg:], facecolor=putils.cycle[dcl], alpha=0.2)
                        dcl = (dcl+1) % len(putils.cycle)
    ax.set_yscale('log')
    ax.legend()
    ax.tick_params('both', length=10, width=1, which='both', labelsize=20)
    xticks = np.linspace(valmin, valmax, nticks+1)
    ax.set_xticks(xticks)

    fig_folder = os.path.join(folder, taskname)
    fig_folder = os.path.join(fig_folder, 'afigs')
    if os.path.isdir(fig_folder) == False:
        os.mkdir(fig_folder)

    outbase = '{}/{}'.format(fig_folder, ntitle)
    #plt.suptitle(outbase, fontsize=12)
    plt.tight_layout()
    
    for ftype in ['png', 'svg', 'pdf']:
        transparent = True
        if ftype == 'png':
            transparent = False
        plt.savefig('{}_fidelity.{}'.format(outbase, ftype), bbox_inches='tight', transparent=transparent, dpi=600)
    plt.show()
    