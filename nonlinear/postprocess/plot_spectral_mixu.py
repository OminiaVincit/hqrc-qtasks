import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.set_cmap(cmap='nipy_spectral')
import pickle

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--nspins', type=int, default=5, help='Number of spins')
    parser.add_argument('--prefix', type=str, default='spec_nspins_5')
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--bcoef', type=float, default=0.42)
    parser.add_argument('--posfix', type=str, default='tmax_25.0_ntaus_250_eig_id_124_tot.binaryfile')
    #parser.add_argument('--posfix', type=str, default='J_1.0_tmax_0.0_ntaus_0_eig_id_124_tot.binaryfile')
    
    args = parser.parse_args()
    print(args)
    folder, prefix, posfix = args.folder, args.prefix, args.posfix
    binfolder = os.path.join(folder, 'binary')
    Nspins, alpha, bc = args.nspins, args.alpha, args.bcoef

    pstates = [i/100 for i in range(101)]
    ild2, ld23 = [], []

    lo, hi = 0, 25
    for p in pstates:
        t1, t2 = [], []
        filename = os.path.join(binfolder, '{}_state_{:.2f}_a_{}_bc_{}_{}'.format(prefix, p, alpha, bc, posfix))
        if os.path.isfile(filename) == False:
            print('Not found {}'.format(filename))
            continue
        with open(filename, 'rb') as rrs:
            z = pickle.load(rrs)
        #print(z.keys())
        for taub in z.keys():
            if taub > hi or taub < lo:
                continue
            egvals = z[taub]
            egvals = sorted(egvals, key=abs, reverse=True)
            la = 1.0/np.abs(egvals[1])
            lb = np.abs(egvals[1])/np.abs(egvals[2])
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
    plt.rcParams["font.size"] = 16 # 全体のフォントサイズが変更されます
    plt.rcParams['xtick.labelsize'] = 14 # 軸だけ変更されます
    plt.rcParams['ytick.labelsize'] = 14 # 軸だけ変更されます
    fig = plt.figure(figsize=(24, 8), dpi=600)
    #vmin, vmax = 1.0, np.max(ild2)
    vmin, vmax = 1.0, 1.4
    extent = [lo, hi, 0, 4.0]
    fig.suptitle(os.path.basename(filename), fontsize=16, horizontalalignment='left')
    # Plot Nspins largest eigenvectors
    ax1 = plt.subplot2grid((2,1), (0,0), colspan=1, rowspan=1)
    ax1.set_title('$|\lambda_1|/|\lambda_2|$', fontsize=16)
    im1 = ax1.imshow(ild2, origin='lower', vmin=vmin, vmax=vmax, extent=extent)
    ax1.set_ylabel('$u$', fontsize=16)
    ax1.set_xticks(list(range(extent[0], extent[1] + 1)))
    
    urange = np.linspace(extent[2], extent[3], 6)
    vrange = ['{:.1f}'.format(x) for x in np.linspace(0, 1, 6)]
    ax1.set_yticks(urange)
    ax1.set_yticklabels(vrange)
    ax1.set_xlabel('$\\tau B$', fontsize=16)
    #ax1.scatter3D(ild2[:, 0], ild2[:, 1], ild2[:, 2], cmap='Green', rasterized=True)

    # # Plot 1/|lambda_2|, |lambda_2| / |lambda_3|
    ax2 = plt.subplot2grid((2,1), (1,0), colspan=1, rowspan=1)
    ax2.set_title('$|\lambda_2| / |\lambda_3|$', fontsize=16)
    #ax2.scatter3D(ld23[:, 0], ld23[:, 1], ld23[:, 2], cmap='Green', rasterized=True)
    
    ax2.set_xlabel('$\\tau B$', fontsize=16)
    im2 = ax2.imshow(ld23, origin='lower', vmin=vmin, vmax=vmax, extent=extent)
    ax2.set_ylabel('$u$', fontsize=16)
    ax2.set_xticks(list(range(extent[0], extent[1] + 1)))
    ax2.set_yticks(urange)
    ax2.set_yticklabels(vrange)

    #ax2.legend()
    plt.tight_layout()
    # call subplot adjust after tight_layout
    #plt.subplots_adjust(hspace=0.0)
    fig.colorbar(im1, ax=[ax1, ax2])
    
    figsave = os.path.join(folder, 'figs')
    if os.path.isdir(figsave) == False:
        os.mkdir(figsave)

    outbase = os.path.basename(filename).replace('.binaryfile', '')
    outbase = os.path.join(figsave, outbase)

    for ftype in ['png']:
        plt.savefig('{}_{}_{}_heat.{}'.format(outbase, lo, hi, ftype), bbox_inches='tight', dpi=600)
    plt.show()



