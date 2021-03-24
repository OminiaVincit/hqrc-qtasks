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
    parser.add_argument('--taskname', type=str, default='delay_tasks3')
    parser.add_argument('--prefix', type=str, default='eig')
    parser.add_argument('--posfix', type=str, default='od_10_dl_1_delay-depolar')
    parser.add_argument('--Vs', type=str, default='1,5')
    parser.add_argument('--als', type=str, default='0.5,1.0,2.0')

    parser.add_argument('--bcmax', type=float, default=2.0, help='Maximum bc')
    parser.add_argument('--bcmin', type=float, default=0.0, help='Minimum bc')
    parser.add_argument('--nbcs', type=int, default=40, help='Number of bc')
    parser.add_argument('--tauB', type=float, default=10.0)

    parser.add_argument('--Nenv', type=int, default=2)
    parser.add_argument('--Nspin', type=int, default=6)
    parser.add_argument('--Nticks', type=int, default=20, help='Number of xticks')
    
    parser.add_argument('--ymin', type=float, default=0.9)
    parser.add_argument('--ymax', type=float, default=1.0)
    args = parser.parse_args()
    print(args)

    folder, prefix, posfix, taskname = args.folder, args.prefix, args.posfix, args.taskname
    Nenv, Nspin, bcmin, bcmax, nbcs, tauB, nticks = args.Nenv, args.Nspin, args.bcmin, args.bcmax, args.nbcs, args.tauB, args.Nticks

    bcs = list(np.linspace(bcmin, bcmax, nbcs + 1))
    bcs = bcs[1:]

    Vs = [int(x) for x in args.Vs.split(',')]
    As = [float(x) for x in args.als.split(',')]
    ymin, ymax = args.ymin, args.ymax

    fig, axs = plt.subplots(1, 1, figsize=(20, 8), squeeze=False)
    axs = np.array(axs).ravel()
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=16

    ntitle = 'bcs_{}_tauB_{}_Nspin_{}_Nenv_{}_Vs_{}_{}'.format(prefix, tauB, Nspin, Nenv,\
        '_'.join([str(x) for x in Vs]), posfix)

    ax = axs[0]
    ax.set_xlabel('$J/B$', fontsize=24)
    ax.set_ylabel('RMSF', fontsize=24)
    ax.set_xlim([bcmin, bcmax])
    if ymin < ymax:
        ax.set_ylim([ymin, ymax])
    ax.set_title('{}'.format(ntitle), fontsize=16)
    ax.grid(True, which="major", ls="-", color='0.65')

    for V in Vs:
        for a in As:
            xs, ys, zs = [], [], []
            for bc in bcs:
                subfolder = os.path.join(folder, taskname)
                subfolder = os.path.join(subfolder, '{}_a_{}_bc_{:.2f}_{}_{}_{}'.format(prefix, a, bc, Nspin, Nenv, posfix))
                subfolder = os.path.join(subfolder, 'log')
                print(subfolder)
                if os.path.isdir(subfolder) == False:
                    continue
                for rfile in glob.glob('{}/tauB_{:.3f}_{:.3f}_V_{}_*_nspins_{}_{}_*.log'.format(subfolder, tauB, tauB, V, Nspin, Nenv)):
                    print(rfile)
                    with open(rfile, mode='r') as rf:
                        lines = rf.readlines()
                        for line in lines:
                            if 'INFO' in line and 'Average RMSF' in line and 'avg-' in line:
                                avg_val_fid = float(re.search(r"avg-val=([0-9.]+)", line).group(1))
                                std_val_fid = float(re.search(r"std-val=([0-9.]+)", line).group(1))
                                xs.append(bc)
                                ys.append(avg_val_fid)
                                zs.append(std_val_fid)
                                break
            xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
            # Plot
            ax.plot(xs, ys, alpha = 0.8, marker='o', markeredgecolor='k', \
                linewidth=4, markersize=6, \
                label='$\\alpha$={}, $M={}$'.format(a, V))
            ax.fill_between(xs, ys - zs, ys + zs, facecolor='gray', alpha=0.5)
        
    ax.legend()
    ax.tick_params('both', length=10, width=1, which='major', labelsize=20)
    xticks = np.linspace(bcmin, bcmax, nticks+1)
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
    