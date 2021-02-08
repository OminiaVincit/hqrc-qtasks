import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Plot for quantum memory
MEM_FUNC_DATA='/data/zoro/qrep/quan_capacity'

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default=MEM_FUNC_DATA)
    parser.add_argument('--taus', type=str, default='3.0')
    parser.add_argument('--ymin', type=float, default='0.0')
    parser.add_argument('--ymax', type=float, default='1.0')
    parser.add_argument('--prefix', type=str, default='quanrc_ion_trap_nspins_5_1_a_0.2_bc_1.0')
    parser.add_argument('--posfix', type=str, default='V_1_len_1000_3000_1000_trials_5')
    
    args = parser.parse_args()
    print(args)

    folder, prefix, posfix = args.folder, args.prefix, args.posfix
    ymin, ymax = args.ymin, args.ymax
    tauBs = [float(x) for x in args.taus.split(',')]

    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(16, 6))
    
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=16

    ntitle = '{}_tauBs_{}_{}'.format(prefix, args.taus, posfix)
    for tauB in tauBs:
        memfile = os.path.join(folder, '{}_tauB_{:.3f}_{}.txt'.format(prefix, tauB, posfix))
        arr = np.loadtxt(memfile)
        print('read {} with shape'.format(memfile), arr.shape)
        ax.plot(arr[:,0], arr[:,1], linewidth=2, markersize=3, marker='o',alpha=0.8,\
                    label='$\\tau B$ = {}'.format(tauB))
        
    ax.set_xlabel('$d$', fontsize=14)
    ax.set_ylabel('QMF$(d)$', fontsize=14)
    #ax.set_ylim([np.min(avg_tests)/2, 2*np.max(avg_tests)])
    ax.set_yscale('log')
    ax.set_ylim([5*10**(-2), 10**0])
    #ax.set_xticks([2**n for n in range(-4, 8)])
    #ax.set_xticklabels(labels='')
    #ax.set_yticklabels(labels='')
    ax.set_title('{}'.format(ntitle), fontsize=12)
    ax.grid(True, which="both", ls="-", color='0.65')
    ax.legend()

    outbase = '{}/{}'.format(folder, ntitle)
    #plt.suptitle(outbase, fontsize=12)
    
    for ftype in ['png', 'pdf', 'svg']:
        print('Save file {}'.format(outbase))
        plt.savefig('{}_func.{}'.format(outbase, ftype), bbox_inches='tight')
    plt.show()
    