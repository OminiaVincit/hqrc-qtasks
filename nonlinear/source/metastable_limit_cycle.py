# Run export OMP_NUM_THREADS=1
from qutip import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
import argparse
from loginit import get_module_logger
import multiprocessing
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import os
from qutils import *

BLUE= [x/255.0 for x in [0, 114, 178]]
VERMILLION= [x/255.0 for x in [213, 94, 0]]
GREEN= [x/255.0 for x in [0, 158, 115]]

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--nspins', type=int, default=5, help='Number of spins')
    parser.add_argument('--binary', type=int, default=0, help='Binary input or not')
    parser.add_argument('--strength', type=float, default=1.0, help='Input strength')
    parser.add_argument('--pstate', type=float, default=2.0, help='Mixed coefficient in the swap state, in [0, 1], -1 is random')
    parser.add_argument('--tauB', type=float, default=10.0, help='Time between the input')
    parser.add_argument('--Tsteps', type=int, default=100, help='Number of time steps')
    parser.add_argument('--Ntrials', type=int, default=10, help='Number of trials')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha of coupled strength')
    parser.add_argument('--bcoef', type=float, default=2.0)
    parser.add_argument('--max_energy', type=float, default=1.0)

    parser.add_argument('--savedir', type=str, default='meta_limit')
    parser.add_argument('--basename', type=str, default='qrc')
    parser.add_argument('--plot', type=int, default=1, help='Flag to plot')
    parser.add_argument('--seed', type=int, default=0, help='Seed for random')
    args = parser.parse_args()
    print(args)

    Nspins, alpha, bc, J, Ntrials = args.nspins, args.alpha, args.bcoef, args.max_energy, args.Ntrials
    tauB, T, pstate, strength = args.tauB, args.Tsteps, args.pstate, args.strength
    savedir, seed, binary = args.savedir, args.seed, args.binary
    basename = '{}_spins_{}_trials_{}_seed_{}_strength_{}_pstate_{:.2f}_a_{}_bc_{}_tauB_{}_T_{}_bin_{}'.format(args.basename, \
        Nspins, Ntrials, seed, strength, pstate, alpha, bc, tauB, T, binary)
    
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    if os.path.isfile(savedir) == False:
        logdir = os.path.join(savedir, 'log')
        if os.path.isdir(logdir) == False:
            os.mkdir(logdir)
        log_filename = os.path.join(logdir, '{}.log'.format(basename))
        logger = get_module_logger(__name__, log_filename)
        logger.info(log_filename)
        logger.info('Nspins={},pstate={},binary={},strength={},alpha={},bcoef={},tauB={},Tsteps={},Ntrials={},seed={}'.format(Nspins, \
            pstate, binary, strength, alpha, bc, tauB, T, Ntrials, seed))

        B = J/bc # Magnetic field
        tau = tauB/B
        L, Mx, My, Mz  = getLiouv_IsingOpen(Nspins, alpha, B, Nspins-1, J)
        S = (tau*L).expm()

        # swap with environment
        q0 = getSci(basis(2, 0), 0, Nspins)
        q1 = getSci(basis(2, 1), 0, Nspins)
        s0 = sprepost(q0, q0.dag())
        s1 = sprepost(q1, q1.dag())
        tc = tensor_contract(S, (0, Nspins))
        
        nobs = Nspins - 1
        
        # Create two basis density matrix
        sp = (basis(2, 0)).unit()
        su = (basis(2, 1)).unit()
        rho_sp = ket2dm(tensor(sp, sp, sp, sp))
        rho_su = ket2dm(tensor(su, su, su, su))
        print('rho_sp', 'rho_su', rho_sp.tr(), rho_su.tr())

        # Initialize density matrix
        tr_spls, tr_suls, fid_spls, fid_suls = [], [], [], []
        mx_ls, my_ls, mz_ls = [], [], []
        
        np.random.seed(seed=abs(seed))
        #avgms = []
        opran = None
        ntimes = 5
        nT = int(T/ntimes)
        eigs = []
        
        if binary > 0:
            us = np.random.randint(2, size=T)
        else:
            us = np.random.rand(T) * strength
        
        rstates = []
        for n in range(T):
            rket = rand_ket(2)
            rstates.append(rket)

        for i in range(Ntrials):
            if seed >= 0:
                rho = rand_dm(2**nobs, density=0.2, dims=rho_sp.dims)
            else:
                betas = np.random.rand(nobs)
                print(i, betas)
                rho = basis(2, 0) * betas[0] + basis(2, 1) * (1.0 - betas[0])
                rho = rho.unit()
                for j in range(1, len(betas)):
                    tmp = basis(2, 0) * betas[j] + basis(2, 1) * (1.0 - betas[j])
                    tmp = tmp.unit()
                    rho = tensor(rho, tmp)
                rho = ket2dm(rho)
            
            print(i, rho.shape, rho.type, rho.tr())
            rho = operator_to_vector(rho)
            # fs_sp, fs_su, tr_sp, tr_su = [], [], [], []

            mxs, mys, mzs = [], [], []
            for n in range(T):
                # v = pstate 
                # if v < 0 or v > 1:
                #     v = us[n]
                # s_prep = v * s0 + (1.0 - v) * s1
                rket = rstates[n]
                q = getSci(rket, 0, Nspins)
                s_prep = sprepost(q, q.dag())
                ts = tc * s_prep
                if i == 0:
                    if opran == None:
                        opran = ts
                    else:
                        opran = ts * opran
                    if n % nT == 0:
                        evs = opran.eigenstates()[0]
                        evs = sorted(evs, key=abs, reverse=True)
                        eigs.append(evs)
                #print(ts.dims, ts.shape, ts.iscp, ts.istp, ts.iscptp)
                rho = ts * rho
                rh1 =  vector_to_operator(rho)
                #print(x, rh1.shape, rh1.type, rh1.tr())
                obmx = (Mx * rh1).tr()
                obmy = (My * rh1).tr()
                obmz = (Mz * rh1).tr()
                mxs.append(np.real(obmx))
                mys.append(np.real(obmy))
                mzs.append(np.real(obmz))

                # fs_sp.append(fidelity(rho_sp, rh1))
                # fs_su.append(fidelity(rho_su, rh1))

                # tr_sp.append(tracedist(rho_sp, rh1))
                # tr_su.append(tracedist(rho_su, rh1))

            #print('after xs: ', rh1.shape, rh1.type, rh1.tr())

            # tr_spls.append(tr_sp)
            # tr_suls.append(tr_su)
            # fid_spls.append(fs_sp)
            # fid_suls.append(fs_su)
            mx_ls.append(mxs)
            my_ls.append(mys)
            mz_ls.append(mzs)

    # Plot file
    if args.plot > 0:
        plt.rc('font', family='serif', size=12)
        plt.rc('mathtext', fontset='cm')
        fig = plt.figure(figsize=(25, 20), dpi=600)
        outbase = os.path.join(savedir, basename)
        xs = list(range(T))

        # # Plot fidelity and distance line
        # ax1 = plt.subplot2grid((4,1), (0,0), colspan=1, rowspan=1)
        # # ax1.plot(xs, fs, 'o-', label='Fidelity-ss')
        # # ax1.plot(xs, fs_sp, 'o-', label='Fs_sp')
        # # ax1.plot(xs, fs_su, 'o-', label='Fs_su')

        # for i in range(len(tr_spls)):
        #     if i == 0:
        #         ax1.plot(xs, tr_spls[i],  color=VERMILLION, alpha=0.7, label='Basis 0')
        #         ax1.plot(xs, tr_suls[i],  color=BLUE, alpha=0.7, label='Basis 1')
        #     else:
        #         ax1.plot(xs, tr_spls[i], color=VERMILLION, alpha=0.7)
        #         ax1.plot(xs, tr_suls[i], color=BLUE, alpha=0.7)

        # ax1.set_xlabel('$T$')
        # ax1.legend()  
        # ax1.set_title('Trace: {}'.format(outbase))

        # ax2 = plt.subplot2grid((4,1), (1,0), colspan=1, rowspan=1)
        # for i in range(len(fid_spls)):
        #     if i == 0:
        #         ax2.plot(xs, fid_spls[i], color=VERMILLION, alpha=0.7, label='Basis 0')
        #         ax2.plot(xs, fid_suls[i], color=BLUE, alpha=0.7, label='Basis 1')
        #     else:
        #         ax2.plot(xs, fid_spls[i], color=VERMILLION, alpha=0.7)
        #         ax2.plot(xs, fid_suls[i], color=BLUE, alpha=0.7)

        # ax2.set_xlabel('$T$')
        # ax2.legend()  
        # ax2.set_title('Fidelity: {}'.format(outbase))

        M = len(eigs)
        mlss = [mx_ls, my_ls, mz_ls]
        lbs = ['Mx', 'My', 'Mz']
        cols = [VERMILLION, BLUE, GREEN]
        # viridis = cm.get_cmap('viridis', Ntrials)
        # cmaps = viridis(np.linspace(0, 1, Ntrials))

        for j in range(3):
            jnext = (j+1)%3
            mls, label, col = mlss[j], lbs[j], cols[j]
            mls_next, label_next, col_next = mlss[jnext], lbs[jnext], cols[jnext]
            
            ax = plt.subplot2grid((5, M), (j, 0), colspan=M-1, rowspan=1)
            for i in range(len(mls)):
                if i == 0:
                    ax.plot(xs, mls[i], color=col, alpha=0.7, label=label)
                else:
                    ax.plot(xs, mls[i], color=col, alpha=0.7)

            ax.set_xlabel('$T$')
            ax.legend()  
            ax.set_title('Average magnezation $\langle {}\\rangle$: {}'.format(label, outbase))

            # Plot 3D plot
            bx = plt.subplot2grid((5, M), (j, M-1), projection='3d', colspan=1, rowspan=1)
            for i in range(len(mls)):
                #bx.plot(mls[i], mls_next[i], np.arange(len(mls[i])), alpha=0.5, color='gray', marker='o', mec='k', mfc=cols[j])
                bx.plot(mls[i], mls_next[i], range(1, 1 + len(mls[i])), alpha=0.5, color=cols[j])
                
                #bx.scatter(mls[i], mls_next[i], c=np.linspace(0, 1, len(mls[i])), alpha=0.7, edgecolor='k', cmap='PuBu')
                #bx.scatter3D(mls[i], mls_next[i], np.arange(len(mls[i])), alpha=0.7, edgecolor='k')
                bx.set_xticklabels([])
                bx.set_yticklabels([])
                bx.set_xticks([])
                bx.set_yticks([])
                bx.grid(False)
                bx.tick_params(axis='z', which='both', direction='out', length=6)
            bx.set_xlabel(label)
            bx.set_ylabel(label_next)
            

        for i in range(M):
            ax4 = plt.subplot2grid((5, M), (3,i), colspan=1, rowspan=1)
            
            circle = Circle((0, 0), 1.0)
            p = PatchCollection([circle], cmap=matplotlib.cm.jet, alpha=0.1)
            ax4.add_collection(p)
            ax4.axis('equal')
            w = eigs[i]
            for xi, yi in zip(np.real(w), np.imag(w)):
                ax4.plot(xi, yi, 'o', color='k', alpha=0.7)
            ax4.set_title('Eig-dist T={}'.format(i*nT))
            ax4.set_xlabel('Real')
            ax4.set_ylabel('Img')

            ax5 = plt.subplot2grid((5, M), (4,i), colspan=1, rowspan=1)
            for j in range(len(w)):
                ax5.plot(j+1, abs(w[j]), 'o', color='k', alpha=0.7)
            ax5.set_xlabel('Index')
            ax5.set_ylabel('Abs')
            ax5.set_title('Spectral')

        for ftype in ['png']:
            plt.savefig('{}_v2.{}'.format(outbase, ftype), bbox_inches='tight', dpi=600)
        plt.show()

  