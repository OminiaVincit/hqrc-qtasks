import os
import argparse
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
#import holoviews as hv
#hv.extension('matplotlib')
#plt.set_cmap(cmap='nipy_spectral')
import re
import plotutils as putils

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--spins', type=int, default=6, help='Number of the spins in the total system')
    parser.add_argument('--envs', type=int, default=1, help='Number of the spins in the environmental system')
    parser.add_argument('--rho', type=int, default=0, help='Flag for initializing the density matrix')
    parser.add_argument('--max_energy', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1e-14)

    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha of coupled strength, 0 for random coupling')
    parser.add_argument('--bcoef', type=float, default=1.0, help='bcoeff nonlinear term (non-diagonal term)')
    parser.add_argument('--dynamic', type=str, default='ion_trap')
    parser.add_argument('--virtuals', type=int, default=1)

    parser.add_argument('--trainlen', type=int, default=500)
    parser.add_argument('--vallen', type=int, default=200)
    parser.add_argument('--buffer', type=int, default=500)
    parser.add_argument('--delay', type=int, default=10)

    parser.add_argument('--nproc', type=int, default=125)
    parser.add_argument('--ntrials', type=int, default=10)

    parser.add_argument('--folder', type=str, default='QTasks_repeated')

    parser.add_argument('--tauB', type=float, default=10.0, help='tauB')

    parser.add_argument('--usecorr', type=int, default=0, help='Use correlator operators')
    parser.add_argument('--reservoir', type=int, default=1, help='Use quantum reservoir to predict')
    parser.add_argument('--postprocess', type=int, default=1, help='Use post processing')
    parser.add_argument('--lastrho', type=int, default=1, help='Use last rho in test phase')

    parser.add_argument('--fvmin', type=float, default=0.0, help='vmin for fidelity mat')
    parser.add_argument('--fvmax', type=float, default=0.0, help='max for fidelity mat')

    parser.add_argument('--nvmin', type=float, default=0.0, help='vmin for negativity diff mat')
    parser.add_argument('--nvmax', type=float, default=0.0, help='max for negativity diff mat')

    parser.add_argument('--data', type=str, default='rand')
    args = parser.parse_args()
    print(args)

    n_spins, n_envs, max_energy, beta, alpha, bcoef, init_rho = args.spins, args.envs, args.max_energy, args.beta, args.alpha, args.bcoef, args.rho
    dynamic, tauB, usecorr = args.dynamic, args.tauB, args.usecorr
    use_reservoir, use_postprocess, test_lastrho = args.reservoir, args.postprocess, args.lastrho

    fvmin, fvmax, nvmin, nvmax = args.fvmin, args.fvmax, args.nvmin, args.nvmax


    train_len, val_len, buffer, delay = args.trainlen, args.vallen, args.buffer, args.delay
    ntrials, folder = args.ntrials, args.folder
    V, dat_label = args.virtuals, args.data
    log_folder = os.path.join(folder, 'log')

    basename = 'qrc_{}_{}_post_{}_lastr_{}_{}_corr_{}_nspins_{}_{}_a_{}_bc_{}_tauB_{}_V_{}_len_{}_{}_{}_maxd_{}_trials_{}'.format(\
        args.reservoir, args.data, args.postprocess, args.lastrho,\
        dynamic, usecorr, n_spins, n_envs, alpha, bcoef, tauB, \
        V, buffer, train_len, val_len, delay, ntrials)

    
    fid_arr = np.zeros((delay+1, delay+1))
    neg_arr = np.zeros((delay+1, delay+1))
    for d1 in range(delay+1):
        for d2 in range(delay+1):
            log_filename = os.path.join(log_folder, 'delay_{}_{}_{}.log'.format(d1, d2, basename))
            with open(log_filename, mode='r') as rf:
                lines = rf.readlines()
                for line in lines:
                    if 'INFO' in line and 'Average RMSF' in line and 'avg-' in line:
                        avg_val_fid = float(re.search(r"avg-val=([0-9.]+)", line).group(1))
                        fid_arr[d1, d2] = avg_val_fid
                    if 'INFO' in line and 'Average RMS negativity' in line and 'avg-' in line:
                        avg_val_neg = float(re.search(r"avg-val=([0-9.]+)", line).group(1))
                        neg_arr[d1, d2] = avg_val_neg

    fig, axs = plt.subplots(1, 2, figsize=(22, 10), squeeze=False)
    axs = np.array(axs).ravel()
    putils.setPlot()
    fmt = ticker.StrMethodFormatter("{x}")


    for m in range(len(axs)):
        ax = axs[m]
        color = 'k'
        if m == 0:
            ax.set_title('Avg. RMSF (%)')
            arr = fid_arr
            cmap = plt.cm.get_cmap('summer')
            if fvmin >= fvmax:
                vmin, vmax = None, None
            else:
                vmin, vmax = fvmin, fvmax
        else:
            ax.set_title('Avg. RMSE Negativity (%)')
            arr = neg_arr
            cmap = plt.cm.get_cmap('viridis_r')
            if nvmin >= nvmax:
                vmin, vmax = None, None
            else:
                vmin, vmax = nvmin, nvmax

        ax.imshow(arr, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
        ax.set_xticks(range(delay+1))
        ax.set_yticks(range(delay+1))
        ax.tick_params('both', length=8, width=1, which='major', labelsize=20)
        ax.tick_params('both', length=6, width=1, which='minor')
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)

        # Loop over data dimensions and create text annotations.
        for i in range(delay+1):
            for j in range(delay+1):
                text = ax.text(j, i, '{:.1f}'.format(100*arr[i, j]), ha="center", va="center", color=color, fontsize=14)

    #plt.colorbar()

    fig_folder = os.path.join(folder, 'figs')
    os.makedirs(fig_folder, exist_ok=True)

    outbase = os.path.join(fig_folder, basename)
    #plt.suptitle(outbase, fontsize=12)
    plt.tight_layout()
    
    for ftype in ['png', 'svg']:
        transparent = True
        if ftype == 'png':
            transparent = False
        plt.savefig('{}.{}'.format(outbase, ftype), bbox_inches='tight', transparent=transparent, dpi=600)
    plt.show()
    