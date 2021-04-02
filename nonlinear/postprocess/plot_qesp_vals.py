import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import plotutils as plu

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--valmin', type=float, default=0.0, help='Minimum of val')
    parser.add_argument('--valmax', type=float, default=12.5, help='Maximum of val')
    parser.add_argument('--nvals', type=int, default=125, help='Number of vals')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--bcoef', type=float, default=1.0)
    parser.add_argument('--tauB', type=float, default=0.0)
    parser.add_argument('--nspins', type=int, default=6)
    parser.add_argument('--nenvs', type=int, default=2)
    parser.add_argument('--nticks', type=int, default=25, help='Number of xticks')
    parser.add_argument('--prefix', type=str, default='esp_ion_trap_nspins')
    parser.add_argument('--ntrials', type=int, default=100)
    args = parser.parse_args()
    print(args)

    folder, prefix = args.folder, args.prefix
    valmin, valmax, nvals = args.valmin, args.valmax, args.nvals
    alpha, bc, tauB = args.alpha, args.bcoef, args.tauB
    nspins, nenvs, nticks, ntrials = args.nspins, args.nenvs, args.nticks, args.ntrials
    vals = list(np.linspace(valmin, valmax, nvals + 1))
    vals = vals[1:]
    binfolder = os.path.join(folder, 'binary')
    
    if alpha == 0.0:
        # variable = a
        vlabel = '$\\alpha$'
    elif bc == 0.0:
        #vals = np.arange(0.2, 5.1, 0.2)
        #yticks = [4, 9, 14, 19]
        #yticklabels = ['{:.1f}'.format((t+1)/5) for t in yticks]
        vlabel = '$J_b/B$'
    elif tauB == 0.0:
        vlabel = '$\\tau B$'
        #xticks = range(0, ntaus, 5)
    else:
        exit(1)
    cmap = plt.get_cmap("twilight")
    fig, axs = plt.subplots(2, 1, figsize=(24, 14), squeeze=False)
    axs = axs.ravel()
    ax1, ax2 = axs[0], axs[1]
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=20
    plt.rcParams['xtick.labelsize'] = 20 # 軸だけ変更されます
    plt.rcParams['ytick.labelsize'] = 20 # 軸だけ変更されます

    sprefix = '{}_{}_{}'.format(prefix, nspins, nenvs)
    Ls = range(10, 101, 5)
    Ls = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    N = len(Ls)
    colors = plt.cm.viridis(np.linspace(0, 1, N+5))
    esparr = []
    for i in range(N):
        L, col = Ls[i], colors[i]
        posfix = 'V_1_len_{}_ntrials_{}'.format(L, ntrials)
        ntitle = '{}_nspins_{}_{}_a_{}_bc_{}_tauB_{}_{}'.format(prefix, nspins, nenvs, alpha, bc, tauB, posfix)
        drate, ts = [], []
        for val in vals:
            if alpha == 0.0:
                valbase = '{}_a_{:.3f}_bc_{:.3f}_tauB_{:.3f}_{}'.format(sprefix, val, bc, tauB, posfix)
            elif bc == 0.0:
                valbase = '{}_a_{:.3f}_bc_{:.3f}_tauB_{:.3f}_{}'.format(sprefix, alpha, val, tauB, posfix)
            else:
                valbase = '{}_a_{:.3f}_bc_{:.3f}_tauB_{:.3f}_{}'.format(sprefix, alpha, bc, val, posfix)

            memfile = os.path.join(binfolder, '{}.npy'.format(valbase))
            if os.path.isfile(memfile) == False:
                print('Not found {}'.format(memfile))
                continue
            arr = np.load(memfile)
            print('read {} with shape'.format(memfile), arr.shape)
            drate.append(arr)
            ts.append(val)

        drate = np.array(drate).T
        if len(ts) > 0:
            #for i in range(len(drate)):
            #    ax.plot(ts, drate[i], color=GREEN, alpha=0.5, linestyle='dashdot')
            ax1.plot(ts, np.mean(drate, axis=0).ravel(), marker='o', markersize=0, alpha=0.8, color=col, label='Time steps = {}'.format(L), linewidth=5.0)
            esparr.append(np.mean(drate, axis=0).ravel())
    esparr = np.array(esparr)

    ax1.set_yscale('log')
    ax1.grid(axis='x')
    #ax.legend()
    ax1.set_title(ntitle)

    cmap1 = plt.get_cmap("RdBu_r")
    cmap2 = plt.get_cmap("rainbow")
    cmap3 = plt.get_cmap("CMRmap")
    cmap4 = plt.get_cmap("PRGn")
    

    im = plu.plotContour(fig, ax2, np.log10(esparr), 'Rate', 24, None, None, cmap1)
    fig.colorbar(im, ax=ax2, orientation="horizontal", format='%d')

    xticks = np.linspace(valmin, valmax, nticks+1)
    xticklabels = ['{:.1f}'.format(t) for t in xticks]
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(labels=xticklabels)
    ax1.set_xlim([valmin, valmax])
    
    #ax2.set_yscale('log')
    yticks = range(1, N, 2)
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(labels=['{:d}'.format(Ls[t]) for t in yticks], fontsize=24)
    
    xticks = range(-1, len(vals), 5)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(labels=['{:.1f}'.format((t+1)/10) for t in xticks], fontsize=24)
    

    for bx in axs:   
        #bx.tick_params(axis='both', which='major', labelsize=16)
        #bx.tick_params(axis='both', which='minor', labelsize=12)
        bx.tick_params('both', length=10, width=1.0, which='major', labelsize=20)
        bx.set_xlabel('$\\tau B$', fontsize=24)
    
    fig_folder = os.path.join(folder, 'figs')
    if os.path.isdir(fig_folder) == False:
        os.mkdir(fig_folder)

    outbase = '{}/{}'.format(fig_folder, ntitle)
    #plt.suptitle(outbase, fontsize=12)
    plt.tight_layout()
    #fig.colorbar(im, ax=ax, orientation="horizontal")
    for ftype in ['png', 'pdf', 'svg']:
        print('Save file {}'.format(outbase))
        plt.savefig('{}_qesp.{}'.format(outbase, ftype), bbox_inches='tight')
    plt.show()
    