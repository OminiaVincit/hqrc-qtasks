import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--prefix', type=str, default='spec_nspins_5')
    #parser.add_argument('--posfix', type=str, default='a_0.2_bc_0.42_tmax_50.0_ntaus_501_eig_id_100_tot.binaryfile')
    parser.add_argument('--posfix', type=str, default='a_0.2_bc_0.42_tmax_0.0_ntaus_0_eig_id_100_tot.binaryfile')
    
    args = parser.parse_args()
    print(args)
    folder, prefix, posfix = args.folder, args.prefix, args.posfix
    
    pstates = [i/100 for i in range(101)]
    ild2, ld23 = [], []

    lexp1, lexp2 = 0, 14
    for p in pstates:
        t1, t2 = [], []
        filename = os.path.join(folder, '{}_state_{:.2f}_{}'.format(prefix, p, posfix))
        if os.path.isfile(filename) == False:
            print('Not found {}'.format(filename))
            continue
        with open(filename, 'rb') as rrs:
            z = pickle.load(rrs)

        for tau in z.keys():
            if tau > 2**(lexp2) or tau < 2**(lexp1):
                continue
            egvals = z[tau]
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
    plt.rc('font', family='serif', size=14)
    plt.rc('mathtext', fontset='cm')
    fig = plt.figure(figsize=(24, 7), dpi=600)
    vmin, vmax = 1.0, max(np.max(ild2), np.max(ld23))
    extent = [lexp1, lexp2, 0, 3]
    # Plot Nspins largest eigenvectors
    ax1 = plt.subplot2grid((2,1), (0,0), colspan=1, rowspan=1)
    ax1.set_title('$1/|\lambda_2|$ {}'.format(os.path.basename(filename)), size=8)
    im1 = ax1.imshow(ild2, origin='lower', vmin=vmin, vmax=vmax, extent=extent)
    ax1.set_ylabel('$u$', fontsize=16)
    ax1.set_xticks(list(range(extent[0], extent[1] + 1)))
    
    urange = np.linspace(extent[2], extent[3], 6)
    vrange = ['{:.1f}'.format(x) for x in np.linspace(0, 1, 6)]
    ax1.set_yticks(urange)
    ax1.set_yticklabels(vrange)
    ax1.set_xlabel('$\log_2(\\tau)$', fontsize=16)
    #ax1.scatter3D(ild2[:, 0], ild2[:, 1], ild2[:, 2], cmap='Green', rasterized=True)

    # # Plot 1/|lambda_2|, |lambda_2| / |lambda_3|
    ax2 = plt.subplot2grid((2,1), (1,0), colspan=1, rowspan=1)
    ax2.set_title('$|\lambda_2| / |\lambda_3|$')
    #ax2.scatter3D(ld23[:, 0], ld23[:, 1], ld23[:, 2], cmap='Green', rasterized=True)
    
    ax2.set_xlabel('$\log_2(\\tau)$', fontsize=16)
    im2 = ax2.imshow(ld23, origin='lower', vmin=vmin, vmax=vmax, extent=extent)
    ax2.set_ylabel('$u$', fontsize=16)
    ax2.set_xticks(list(range(extent[0], extent[1] + 1)))
    ax2.set_yticks(urange)
    ax2.set_yticklabels(vrange)

    #ax2.legend()
    fig.colorbar(im1, ax=[ax1, ax2])

    outbase = filename.replace('.binaryfile', '')
    for ftype in ['png']:
        plt.savefig('{}_{}_{}_heat.{}'.format(outbase, lexp1, lexp2, ftype), bbox_inches='tight', dpi=600)
    plt.show()



