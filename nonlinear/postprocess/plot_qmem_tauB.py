import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

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
    parser.add_argument('--tmax', type=float, default=25.0, help='Maximum of tauB')
    parser.add_argument('--ntaus', type=int, default=25, help='Number of tausB')
    parser.add_argument('--prefix', type=str, default='quanrc_ion_trap_nspins_5_1_a_1.0_bc_2.0')
    parser.add_argument('--posfix', type=str, default='V_1_len_1000_3000_1000_trials_5')
    
    args = parser.parse_args()
    print(args)

    folder, prefix, posfix = args.folder, args.prefix, args.posfix
    ymin, ymax = args.ymin, args.ymax
    tmin, tmax, ntaus = args.tmin, args.tmax, args.ntaus
    tauBs = list(np.linspace(tmin, tmax, ntaus + 1))
    tauBs = tauBs[1:]

    cmap = plt.get_cmap("twilight")
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [3, 1]})
    ax = axs[0]
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=16

    ntitle = '{}_{}'.format(prefix, posfix)
    memarr, ts, mcs = [], [], []
    for tauB in tauBs:
        memfile = os.path.join(folder, '{}_tauB_{:.3f}_{}.txt'.format(prefix, tauB, posfix))
        arr = np.loadtxt(memfile)
        print('read {} with shape'.format(memfile), arr.shape)
        loc_arr = arr[:, 1]
        memarr.append(np.log10(loc_arr))
        #loc_arr = loc_arr - np.min(loc_arr)
        loc_arr[loc_arr < 10**(-1)] = 0.0
        mcs.append(np.sum(loc_arr))
        ts.append(tauB)

    memarr = np.array(memarr)
    ymin, ymax = np.min(memarr), np.max(memarr)
    im = plotContour(fig, ax, memarr, '{}'.format(ntitle), 16, ymin, ymax, cmap)
    
    ax.set_xlabel('$d$', fontsize=24)
    ax.set_ylabel('$\\tau B$', fontsize=24)
    ax.set_xlim([0, 30])
    #ax.set_ylim([np.min(avg_tests)/2, 2*np.max(avg_tests)])
    #ax.set_yscale('log')
    #ax.set_ylim([5*10**(-2), 10**0])
    yticks = [4, 9, 14, 19, 24]
    ax.set_yticks(yticks)
    ax.set_yticklabels(labels=['{}'.format(t+1) for t in yticks])
    #ax.set_yticklabels(labels='')
    #ax.set_title('{}'.format(ntitle), fontsize=12)
    #ax.grid(True, which="both", ls="-", color='0.65')
    #ax.legend()

    ax2 = axs[1]
    ax2.set_xlabel('QMC', fontsize=24)
    ax2.set_ylabel('$\\tau B$', fontsize=24)
    ax2.barh(ts, mcs, height=1.0, edgecolor='k', alpha=0.8)
    ax2.set_ylim(tauBs[0], tauBs[-1])
    
    for bx in axs:
        bx.tick_params(axis='both', which='major', labelsize=16)
        bx.tick_params(axis='both', which='minor', labelsize=12)
        bx.tick_params('both', length=10, width=1.0, which='major')

    outbase = '{}/{}'.format(folder, ntitle)
    #plt.suptitle(outbase, fontsize=12)
    plt.tight_layout()
    fig.colorbar(im, ax=ax, orientation="vertical")
    for ftype in ['png', 'pdf', 'svg']:
        print('Save file {}'.format(outbase))
        plt.savefig('{}_func.{}'.format(outbase, ftype), bbox_inches='tight')
    plt.show()
    