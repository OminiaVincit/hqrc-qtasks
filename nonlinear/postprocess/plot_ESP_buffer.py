import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker, cm

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='echo_repeated')
    parser.add_argument('--prefix', type=str, default='qrc_echo_ion_trap')
    
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--bcoef', type=float, default=0.42)
    parser.add_argument('--tmax', type=float, default=25, help='Maximum of tauB')
    parser.add_argument('--ntaus', type=int, default=125, help='Number of tausB')

    parser.add_argument('--posfix', type=str, default='esp_trials_1_2_esp')
    parser.add_argument('--Ts', type=str, default='10,20,50,100,200,500,1000,2000,5000,10000')
    args = parser.parse_args()
    print(args)

    folder, posfix = args.folder, args.posfix
    prefix = '{}_a_{}_bc_{}_tmax_{}_ntaus_{}'.format(args.prefix, args.alpha, args.bcoef, args.tmax, args.ntaus)
    
    Ts = [int(x) for x in args.Ts.split(',')]

    cmap = plt.get_cmap("RdBu")
    fig, axs = plt.subplots(1, 1, figsize=(12, 4), squeeze=False)
    axs = axs.ravel()
    ax = axs[0]
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=12

    ntitle = ''
    B = 1.0 / args.bcoef

    for T in Ts:
        for rfile in glob.glob('{}/binary/{}*_T_{}_{}*{}.txt'.format(folder, prefix, T, T+1000, posfix)):
            print(rfile)
            ntitle = os.path.basename(rfile)
            #nidx = ntitle.find('layers')
            #ntitle = ntitle[nidx:]
            ntitle = ntitle.replace('.txt', '')
            tmp = np.loadtxt(rfile)
            print(tmp.shape)
            ts, avs, stds = tmp[:, 2], tmp[:, -2], tmp[:, -1]
            ts = ts * B
            ax.plot(ts, avs, label='T={}'.format(T))

    ax.set_title('{}'.format(ntitle), fontsize=10)
    #ax.set_xscale("log", basex=2)
    ax.set_yscale("log", basey=10)
    #ax.set_xticks([2**x for x in np.arange(-7,7.1,1.0)])
    #ax.set_xticks([2**x for x in np.arange(0,14.1,1.0)])
    
    ax.minorticks_on()
    ax.tick_params('both', length=6, width=1, which='major')
    ax.tick_params('both', length=3, width=1, which='minor')
    ax.legend()
    ax.set_xlabel('$\\tau B$', fontsize=16)
    ax.set_ylabel('QESP', fontsize=16)
    plt.tight_layout()
    #ax.set_xlim([2**(0), 2**(7)])
    figsave = os.path.join(folder, 'figs')
    if os.path.isdir(figsave) == False:
        os.mkdir(figsave)

    outbase = os.path.join(figsave, ntitle)
    for ftype in ['png']:
        plt.savefig('{}_v1.{}'.format(outbase, ftype), bbox_inches='tight')
    plt.show()
    