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
    parser.add_argument('--ymax', type=float, default='0.0')
    parser.add_argument('--valmin', type=float, default=0.0, help='Minimum of val')
    parser.add_argument('--valmax', type=float, default=25.0, help='Maximum of val')
    parser.add_argument('--nvals', type=int, default=125, help='Number of vals')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--bcoef', type=float, default=1.0)
    parser.add_argument('--tauB', type=float, default=1.0)
    parser.add_argument('--thres', type=float, default=0.0)
    parser.add_argument('--V', type=int, default=1)
    parser.add_argument('--width', type=float, default=1.0)
    parser.add_argument('--nspins', type=str, default='3,4,5,6')
    parser.add_argument('--nenvs', type=int, default=2)
    parser.add_argument('--dmax', type=int, default=10)
    parser.add_argument('--nticks', type=int, default=25, help='Number of xticks')
    parser.add_argument('--prefix', type=str, default='quanrc_ion_trap_nspins')
    parser.add_argument('--posfix', type=str, default='len_1000_3000_100_dmax_20')
    parser.add_argument('--ntrials', type=int, default=10)
    args = parser.parse_args()
    print(args)

    folder, prefix, posfix = args.folder, args.prefix, args.posfix
    Ns = [int(x) for x in args.nspins.split(',')]

    posfix = 'V_{}_{}'.format(args.V, posfix)
    ymin, ymax, dmax = args.ymin, args.ymax, args.dmax
    valmin, valmax, nvals = args.valmin, args.valmax, args.nvals
    alpha, bc, tauB, width = args.alpha, args.bcoef, args.tauB, args.width
    nenvs, ntrials, nticks = args.nenvs, args.ntrials, args.nticks
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
    fig, axs = plt.subplots(2, 1, figsize=(20, 12), squeeze=False)
    axs = axs.ravel()
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=20

    ntitle = '{}_nenvs_{}_a_{}_bc_{}_tauB_{}_thres_{}_{}_tod_{}'.format(prefix, nenvs, alpha, bc, tauB, args.thres, posfix, dmax)
    
    ax = axs[1]
    for nspin in Ns:
        sprefix = '{}_{}_{}'.format(prefix, nspin, nenvs)
        memarr, ts = [], []
        mcs_avg, mcs_std = [], []
        for val in vals:
            if alpha == 0.0:
                valbase = '{}_a_{:.3f}_bc_{:.3f}_tauB_{:.3f}_{}'.format(sprefix, val, bc, tauB, posfix)
            elif bc == 0.0:
                valbase = '{}_a_{:.3f}_bc_{:.3f}_tauB_{:.3f}_{}'.format(sprefix, alpha, val, tauB, posfix)
            else:
                valbase = '{}_a_{:.3f}_bc_{:.3f}_tauB_{:.3f}_{}'.format(sprefix, alpha, bc, val, posfix)
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
            avg_loc_arr = np.mean(loc_arr, axis=0)
            memarr.append(avg_loc_arr)

            mcs_avg.append(np.mean(loc_sum))
            mcs_std.append(np.std(loc_sum))
            ts.append(val)
        if len(mcs_avg) == 0:
            continue
        mcs_avg, mcs_std = np.array(mcs_avg), np.array(mcs_std)
        ax.fill_between(ts, mcs_avg - mcs_std, mcs_avg + mcs_std, facecolor='gray', alpha=0.5)
        ax.plot(ts, mcs_avg, alpha=0.8, marker='o', markeredgecolor='k', \
            markersize=0, linewidth=5, label='$N_m$={}'.format(nspin - nenvs))

        memarr = np.array(memarr).T
    ax.grid(axis='x')
    ax.legend()

    # Plot MC bar
    ax = axs[0]
    # ymin, ymax = np.min(memarr), np.max(memarr)
    # im = plotContour(fig, ax, memarr.T, '{}'.format(ntitle), 16, ymin, ymax, cmap)
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
    
    #ax.set_yticklabels(labels='')
    ax.set_title('{}_spins_{}'.format(ntitle, Ns[-1]), fontsize=12)
    #ax.grid(True, which="both", ls="-", color='0.65')
    #ax.legend()

    # Plot MC lines

    for bx in axs:
        xticks = np.linspace(valmin, valmax, nticks+1)
        xticklabels = ['{:.1f}'.format(t) for t in xticks]
        bx.set_xticks(xticks)
        bx.set_xticklabels(labels=xticklabels)
        #bx.tick_params(axis='both', which='major', labelsize=16)
        #bx.tick_params(axis='both', which='minor', labelsize=12)
        bx.tick_params('both', length=10, width=1.0, which='major', labelsize=24)
        bx.set_xlim([valmin, valmax])
        if ymax > ymin:
            bx.set_ylim([ymin, ymax])

    fig_folder = os.path.join(folder, 'figs')
    if os.path.isdir(fig_folder) == False:
        os.mkdir(fig_folder)

    outbase = '{}/{}_spins_{}'.format(fig_folder, ntitle, '_'.join([str(x) for x in Ns]))
    #plt.suptitle(outbase, fontsize=12)
    plt.tight_layout()
    #fig.colorbar(im, ax=ax, orientation="horizontal")
    for ftype in ['png', 'pdf', 'svg']:
        print('Save file {}'.format(outbase))
        plt.savefig('{}_func.{}'.format(outbase, ftype), bbox_inches='tight')
    plt.show()
    