import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import plotutils as plu

BLUE= [x/255.0 for x in [0, 114, 178]]
VERMILLION= [x/255.0 for x in [213, 94, 0]]
GREEN= [x/255.0 for x in [0, 158, 115]]
BROWN = [x/255.0 for x in [72, 55, 55]]

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
    parser.add_argument('--nspins', type=int, default=5)
    parser.add_argument('--nenvs', type=int, default=1)
    parser.add_argument('--nticks', type=int, default=25, help='Number of xticks')
    parser.add_argument('--prefix', type=str, default='esp_ion_trap_nspins')
    parser.add_argument('--posfix', type=str, default='V_1_len_50_ntrials_100')
    args = parser.parse_args()
    print(args)

    folder, prefix, posfix = args.folder, args.prefix, args.posfix
    valmin, valmax, nvals = args.valmin, args.valmax, args.nvals
    alpha, bc, tauB = args.alpha, args.bcoef, args.tauB
    nspins, nenvs, nticks = args.nspins, args.nenvs, args.nticks
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
    fig, axs = plt.subplots(1, 1, figsize=(20, 5), squeeze=False)
    axs = axs.ravel()
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=16

    ntitle = '{}_nspins_{}_{}_a_{}_bc_{}_tauB_{}_{}'.format(prefix, nspins, nenvs, alpha, bc, tauB, posfix)
    sprefix = '{}_{}_{}'.format(prefix, nspins, nenvs)

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
    ax = axs[0]
    for i in range(len(drate)):
        ax.plot(ts, drate[i], color=GREEN, alpha=0.5, linestyle='dashdot')
    ax.plot(ts, np.median(drate, axis=0).ravel(), color=VERMILLION, alpha=1.0, label='Median value', linewidth=3.0)
    ax.set_yscale('log')
    ax.grid(axis='x')
    ax.legend()
    ax.set_title(ntitle)

    for bx in axs:
        xticks = np.linspace(valmin, valmax, nticks+1)
        xticklabels = ['{:.1f}'.format(t) for t in xticks]
        bx.set_xticks(xticks)
        bx.set_xticklabels(labels=xticklabels)
        #bx.tick_params(axis='both', which='major', labelsize=16)
        #bx.tick_params(axis='both', which='minor', labelsize=12)
        bx.tick_params('both', length=10, width=1.0, which='major', labelsize=20)
        bx.set_xlim([valmin, valmax])

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
    