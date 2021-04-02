import sys
import os
import glob
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import plotutils as putils

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--nspins', type=int, default=6, help='Number of spins')
    parser.add_argument('--nenvs', type=int, default=2, help='Number of env spins')
    
    parser.add_argument('--bcmax', type=float, default=2.5, help='Maximum bc')
    parser.add_argument('--bcmin', type=float, default=0.0, help='Minimum bc')
    parser.add_argument('--nbcs', type=int, default=125, help='Number of bc')

    parser.add_argument('--amax', type=float, default=2.0, help='Maximum a')
    parser.add_argument('--amin', type=float, default=0.0, help='Minimum a')
    parser.add_argument('--nas', type=int, default=20, help='Number of a')

    parser.add_argument('--nticks', type=int, default=20, help='Number of xticks')
    parser.add_argument('--prefix', type=str, default='spec')
    parser.add_argument('--tauB', type=float, default=10.0)
    parser.add_argument('--posfix', type=str, default='eig_id_124')
    parser.add_argument('--ptype', type=int, default=1, help='Plot type, 0:spectral, 1:line')
    
    args = parser.parse_args()
    print(args)
    folder, prefix, posfix, ptype = args.folder, args.prefix, args.posfix, args.ptype
    Nspins, Nenvs, tauB = args.nspins, args.nenvs, args.tauB
    bcmin, bcmax, nas, nbcs, nticks = args.bcmin, args.bcmax, args.nas, args.nbcs, args.nticks
    als = list(np.linspace(args.amin, args.amax, nas + 1))
    als = als[1:]
    #als.extend([3.0, 4.0, 5.0, 10.0])
    posfix = 'bcmin_{}_bcmax_{}_nbcs_{}_tauB_{}_{}'.format(bcmin, bcmax, nbcs, tauB, posfix)
    
    lo, hi = bcmin, 2.0
    ld2_abc, ld23_abc = [], []
    cmap2 = plt.get_cmap("RdBu")
    cmap2 = plt.get_cmap("rainbow")
    cmap2 = plt.get_cmap("CMRmap")
    cmap1 = plt.get_cmap("PRGn")

    fig, axs = plt.subplots(2, 1, figsize=(24, 12), squeeze=False)
    axs = axs.ravel()
    ax1, ax2 = axs[0], axs[1]
    dcl = 0
    for alpha in als:
        afolder = os.path.join(folder, 'eig_a_{:.1f}_tauB_{}_{}_{}'.format(alpha, tauB, Nspins, Nenvs))
        binfolder = os.path.join(afolder, 'binary')
        
        ild2, ld23 = [], []
        for p in range(10):
            t1, t2 = [], []
            filename = os.path.join(binfolder, '{}_nspins_{}_nenvs_{}_a_{:.1f}_{}_tot_{}.binaryfile'.format(\
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
                t1.append(la)
                t2.append(lb)
                # if lb > la and lb > 1.2:
                #     print(p, tau, lb, la)

            t1, t2 = np.array(t1).ravel(), np.array(t2).ravel()
            ild2.append(t1)
            ld23.append(t2)
    
        ild2_avg, ild2_std  = np.mean(np.array(ild2), axis=0), np.std(np.array(ild2), axis=0)
        ld23_avg, ild23_std = np.mean(np.array(ld23), axis=0), np.std(np.array(ld23), axis=0)
        print(alpha, ild2_avg.shape, ld23_avg.shape)

        if alpha in [0.5, 1.0, 1.5, 2.0]:
            color = putils.cycle[dcl]
            dcl += 1
            ax2.fill_between(xs, ld23_avg - ild23_std, ld23_avg + ild23_std, facecolor=color, alpha=0.2)
            ax2.plot(xs, ld23_avg, alpha=0.8, marker='o', markeredgecolor='k', color=color,\
                markersize=0, linewidth=4, label='$\\alpha$={}'.format(alpha))

        ld2_abc.append(ild2_avg)
        ld23_abc.append(ld23_avg)
    ld2_abc = np.array(ld2_abc)
    ld23_abc = np.array(ld23_abc)

    # Plot file
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams["font.size"] = 24 # 全体のフォントサイズが変更されます
    plt.rcParams['xtick.labelsize'] = 24 # 軸だけ変更されます
    plt.rcParams['ytick.labelsize'] = 24 # 軸だけ変更されます

    #vmin, vmax = 1.0, np.max(ild2)
    vmin, vmax = 1.0, 1.7
    extent = [lo, hi, 0, 4]
    # fig.suptitle(os.path.basename(filename), fontsize=16, horizontalalignment='left')

    if False:
        # Plot the second largest eigenvectors
        ax1 = plt.subplot2grid((2,1), (0,0), colspan=1, rowspan=1)
        ax1.set_title('$1 / |\lambda_2|$', fontsize=24)
        if ptype == 0:
            im1 = ax1.imshow(ld2_abc, origin='lower', vmin=vmin, vmax=vmax, extent=extent)
            ax1.set_ylabel('Index', fontsize=16)
            ax1.set_xticks(list(range(extent[0], extent[1] + 1)))
            urange = np.linspace(extent[2], extent[3], 6)
            vrange = ['{:.1f}'.format(x) for x in np.linspace(0, 1, 6)]
        elif ptype == 1:
            for i in range(len(ld2_abc)):
                ax1.plot(xs, ld2_abc[i], color=GREEN, alpha=0.8, linestyle='dashdot')
        else:
            im = plotContour(fig, ax1, ld2_abc, '$1 / |\lambda_2|$', 24, None, None, cmap1)
            fig.colorbar(im, ax=ax1, orientation="vertical")

    # Plot average |\lambda_{k+1}| / |\lambda_{k}
    #ax2 = plt.subplot2grid((1,1), (0,0), colspan=1, rowspan=1)
    #ax2.set_title('$\langle |\lambda_{k+1}| / |\lambda_{k}| \\rangle$', fontsize=24)
    # if ptype == 0:
    #     #ax2.scatter3D(ld23[:, 0], ld23[:, 1], ld23[:, 2], cmap='Green', rasterized=True)
    #     im2 = ax2.imshow(ld23, origin='lower', vmin=vmin, vmax=vmax, extent=extent)
    #     ax2.set_ylabel('$u$', fontsize=16)
    #     ax2.set_xticks(list(range(extent[0], extent[1] + 1)))
    #     fig.colorbar(im1, ax=[ax1, ax2])
    # elif ptype == 1:
    #     for i in range(len(ld23_abc)):
    #         ax2.plot(xs, ld23_abc[i], color=BLUE, alpha=0.8, linestyle='dashdot')
    # else:
    
    im = putils.plotContour(fig, ax1, ld23_abc, '$\langle |\lambda_{k+1}| / |\lambda_{k}| \\rangle$', 16, None, None, cmap2)
    fig.colorbar(im, ax=ax1, orientation="vertical", format='%.3f')

    
    ax1.set_ylabel('$\\alpha$', fontsize=24)
    yticks = range(4, nas, 5)
    xticks = range(-1, len(xs), 5)
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(labels=['{:.1f}'.format((t+1)/10) for t in yticks], fontsize=24)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(labels=['{:.1f}'.format((t+1)/50) for t in xticks], fontsize=24)
    
    ax2.set_xlim([lo, hi])
    x2ticks = np.linspace(lo, hi, nticks+1)
    ax2.set_xticks(x2ticks)
    ax2.legend()
    ax2.grid(True, axis='x', which="major", ls="-", color='0.65')

    for ax in axs:
        ax.tick_params('both', length=10, width=1.0, which='major', direction='out', labelsize=24)
        ax.set_xlabel('$J/B$', fontsize=24)
    
    plt.tight_layout()
    # call subplot adjust after tight_layout
    #plt.subplots_adjust(hspace=0.0)
    
    figsave = os.path.join(folder, 'figs')
    if os.path.isdir(figsave) == False:
        os.mkdir(figsave)

    outbase = os.path.basename(filename).replace('.binaryfile', '')
    outbase = os.path.join(figsave, outbase)

    for ftype in ['png', 'pdf', 'svg']:
        plt.savefig('{}_abc_ptype_{}.{}'.format(outbase, ptype, ftype), bbox_inches='tight', dpi=600)
    plt.show()



