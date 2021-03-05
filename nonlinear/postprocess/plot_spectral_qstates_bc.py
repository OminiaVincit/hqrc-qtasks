import sys
import os
import glob
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.set_cmap(cmap='nipy_spectral')
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import pickle

BLUE= [x/255.0 for x in [0, 114, 178]]
VERMILLION= [x/255.0 for x in [213, 94, 0]]
GREEN= [x/255.0 for x in [0, 158, 115]]
BROWN = [x/255.0 for x in [72, 55, 55]]

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--nspins', type=int, default=6, help='Number of spins')
    parser.add_argument('--nenvs', type=int, default=2, help='Number of env spins')
    parser.add_argument('--bcmax', type=float, default=2.5, help='Maximum bc')
    parser.add_argument('--bcmin', type=float, default=0.0, help='Minimum bc')
    parser.add_argument('--nbcs', type=int, default=125, help='Number of bc')
    parser.add_argument('--prefix', type=str, default='spec')
    parser.add_argument('--selbcs', type=str, default='0.1, 0.2, 0.5, 1.0, 2.0')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--tauB', type=float, default=10.0)
    parser.add_argument('--posfix', type=str, default='eig_id_124')
    parser.add_argument('--ptype', type=int, default=1, help='Plot type, 0:spectral, 1:line')
    
    args = parser.parse_args()
    print(args)
    folder, prefix, posfix, ptype = args.folder, args.prefix, args.posfix, args.ptype
    binfolder = os.path.join(folder, 'binary')
    Nspins, Nenvs, alpha, tauB = args.nspins, args.nenvs, args.alpha, args.tauB
    bcmin, bcmax, nbcs = args.bcmin, args.bcmax, args.nbcs
    posfix = 'bcmin_{}_bcmax_{}_nbcs_{}_tauB_{}_{}'.format(bcmin, bcmax, nbcs, tauB, posfix)
    ild2, ld23 = [], []

    sel_bcs = [float(x) for x in args.selbcs.split(',')]
    eigs = dict()
    for bc in sel_bcs:
        eigs[bc] = []
    
    M = len(sel_bcs)
    lo, hi = bcmin, bcmax
    for p in range(100):
        t1, t2 = [], []
        filename = os.path.join(binfolder, '{}_nspins_{}_nenvs_{}_a_{}_{}_tot_{}.binaryfile'.format(\
            prefix, Nspins, Nenvs, alpha, posfix, p))
        if os.path.isfile(filename) == False:
            print('Not found {}'.format(filename))
            continue
        with open(filename, 'rb') as rrs:
            z = pickle.load(rrs)
        #print(z.keys())
        xs = []
        for bc in z.keys():
            if bc > hi or bc < lo:
                continue
            xs.append(bc)
            egvals = z[bc]
            egvals = sorted(egvals, key=abs, reverse=True)
            if p == 0 and bc in sel_bcs:
                eigs[bc] = egvals
            
            la = 1.0/np.abs(egvals[1])
            
            # salpha = []
            # for n in range(len(egvals)-1):
            #     salpha.append(np.abs(egvals[n]) - np.abs(egvals[n+1]))
            
            # ralpha = []
            # for n in range(2, len(salpha)):
            #     minval = min(salpha[n], salpha[n-1])
            #     maxval = max(salpha[n], salpha[n-1])
            #     ralpha.append(minval / maxval)

            ralpha = []
            for n in range(1, len(egvals)):
                ralpha.append(np.abs(egvals[n]) / np.abs(egvals[n-1]))
            
            #lb = np.abs(egvals[1])/np.abs(egvals[2])
            lb = np.mean(ralpha)
            t1.append([la])
            t2.append([lb])
            # if lb > la and lb > 1.2:
            #     print(p, tau, lb, la)

        t1, t2 = np.array(t1).ravel(), np.array(t2).ravel()
        ild2.append(t1)
        ld23.append(t2)
    
    ild2 = np.array(ild2)
    ld23 = np.array(ld23)
    print(ild2.shape, ld23.shape)    
    
    # Plot file
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams["font.size"] = 20 # 全体のフォントサイズが変更されます
    plt.rcParams['xtick.labelsize'] = 20 # 軸だけ変更されます
    plt.rcParams['ytick.labelsize'] = 20 # 軸だけ変更されます
    fig = plt.figure(figsize=(24, 15), dpi=600)
    #vmin, vmax = 1.0, np.max(ild2)
    vmin, vmax = 1.0, 1.4
    extent = [lo, hi, 0, 4.0]
    fig.suptitle(os.path.basename(filename), fontsize=16, horizontalalignment='left')
    # Plot Nspins largest eigenvectors
    ax1 = plt.subplot2grid((3,M), (0,0), colspan=M, rowspan=1)
    ax1.set_title('$1 / |\lambda_2|$', fontsize=24)
    if ptype == 0:
        im1 = ax1.imshow(ild2, origin='lower', vmin=vmin, vmax=vmax, extent=extent)
        ax1.set_ylabel('Index', fontsize=16)
        ax1.set_xticks(list(range(extent[0], extent[1] + 1)))
        urange = np.linspace(extent[2], extent[3], 6)
        vrange = ['{:.1f}'.format(x) for x in np.linspace(0, 1, 6)]
    else:
        for i in range(len(ild2)):
            ax1.plot(xs, ild2[i], color=GREEN, alpha=0.5, linestyle='dashdot')
        ax1.plot(xs, np.median(ild2, axis=0).ravel(), color=VERMILLION, alpha=1.0, label='Median value', linewidth=3.0)
        ax1.legend()
        ax1.grid(which='major',color='black',linestyle='-', axis='x', linewidth=1.0, alpha=0.5)
    #ax1.scatter3D(ild2[:, 0], ild2[:, 1], ild2[:, 2], cmap='Green', rasterized=True)

    # # Plot 1/|lambda_2|, |lambda_2| / |lambda_3|
    ax2 = plt.subplot2grid((3,M), (1,0), colspan=M, rowspan=1)
    ax2.set_title('$\langle |\lambda_{k+1}| / |\lambda_{k}| \\rangle$', fontsize=24)
    if ptype == 0:
        #ax2.scatter3D(ld23[:, 0], ld23[:, 1], ld23[:, 2], cmap='Green', rasterized=True)
        im2 = ax2.imshow(ld23, origin='lower', vmin=vmin, vmax=vmax, extent=extent)
        ax2.set_ylabel('$u$', fontsize=16)
        ax2.set_xticks(list(range(extent[0], extent[1] + 1)))
        fig.colorbar(im1, ax=[ax1, ax2])
    else:
        for i in range(len(ld23)):
            ax2.plot(xs, ld23[i], color=BLUE, alpha=0.5, linestyle='dashdot')
        ax2.plot(xs, np.median(ld23, axis=0).ravel(), color=VERMILLION, alpha=1.0, label='Median value', linewidth=3.0)
        ax2.legend()
        ax2.grid(which='major',color='black',linestyle='-', axis='x', linewidth=1.0, alpha=0.5)
        
    for ax in [ax1, ax2]:
        ax.set_xlabel('$J/B$', fontsize=24)
        if ptype == 0:
            ax.set_yticks(urange)
            ax.set_yticklabels(vrange)
        else:
            ax.set_xlim([lo, hi])
            ax.set_xticks(np.linspace(lo, hi, 26))
    
    for i in range(M):
        ax = plt.subplot2grid((3, M), (2,i), colspan=1, rowspan=1)
        circle = Circle((0, 0), 1.0)
        p = PatchCollection([circle], cmap=matplotlib.cm.jet, alpha=0.1)
        ax.add_collection(p)
        ax.axis('equal')
        w = eigs[sel_bcs[i]]
        if len(w) == 0:
            continue
        for xi, yi in zip(np.real(w), np.imag(w)):
            ax.plot(xi, yi, 'o', color='k', alpha=0.7)
        ax.set_title('$J/B$={}'.format(sel_bcs[i]), fontsize=24)
        ax.set_xlabel('Re', fontsize=24)
        ax.set_ylabel('Im', fontsize=24)
        ax.set_xlim([-1.0, 1.0])
        ax.set_ylim([-1.0, 1.0])
        ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    plt.tight_layout()
    # call subplot adjust after tight_layout
    #plt.subplots_adjust(hspace=0.0)
    
    figsave = os.path.join(folder, 'figs')
    if os.path.isdir(figsave) == False:
        os.mkdir(figsave)

    outbase = os.path.basename(filename).replace('.binaryfile', '')
    outbase = os.path.join(figsave, outbase)

    for ftype in ['png', 'pdf', 'svg']:
        plt.savefig('{}_{}_{}_ptype_{}.{}'.format(outbase, lo, hi, ptype, ftype), bbox_inches='tight', dpi=600)
    plt.show()



