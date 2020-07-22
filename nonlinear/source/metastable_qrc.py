# Run export OMP_NUM_THREADS=1
from qutip import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import argparse
from loginit import get_module_logger
import multiprocessing
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
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
    parser.add_argument('--strength', type=float, default=0.2, help='Input strength')
    parser.add_argument('--pstate', type=float, default=0.2, help='Mixed coefficient in the swap state, in [0, 1], -1 is random')
    parser.add_argument('--tau', type=float, default=10.0, help='Time between the input')
    parser.add_argument('--Tsteps', type=int, default=100, help='Number of time steps')
    parser.add_argument('--Ntrials', type=int, default=10, help='Number of trials')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha of coupled strength')
    parser.add_argument('--bcoef', type=float, default=0.42)

    parser.add_argument('--savedir', type=str, default='metaqrc')
    parser.add_argument('--basename', type=str, default='qrc')
    parser.add_argument('--plot', type=int, default=1, help='Flag to plot')
    parser.add_argument('--seed', type=int, default=0, help='Seed for random')
    args = parser.parse_args()
    print(args)

    Nspins, alpha, bc, Ntrials = args.nspins, args.alpha, args.bcoef, args.Ntrials
    tau, T, pstate, strength = args.tau, args.Tsteps, args.pstate, args.strength
    savedir, seed = args.savedir, args.seed
    basename = '{}_spins_{}_trials_{}_seed_{}_strength_{}_pstate_{:.2f}_a_{}_bc_{}_tau_{}_T_{}'.format(args.basename, \
        Nspins, Ntrials, seed, strength, pstate, alpha, bc, tau, T)
    
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    if os.path.isfile(savedir) == False:
        logdir = os.path.join(savedir, 'log')
        if os.path.isdir(logdir) == False:
            os.mkdir(logdir)
        log_filename = os.path.join(logdir, '{}.log'.format(basename))
        logger = get_module_logger(__name__, log_filename)
        logger.info(log_filename)
        logger.info('Nspins={},pstate={},strength={},alpha={},bcoef={},tau={},Tsteps={},Ntrials={},seed={}'.format(Nspins, \
            pstate, strength, alpha, bc, tau, T, Ntrials, seed))


        L, Mx, Mz = getLiouv_IsingOpen(Nspins, alpha, bc)
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
        mx_ls, mz_ls = [], []
        
        np.random.seed(seed=abs(seed))
        #avgms = []
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

            mxs, mzs = [], []
            us = np.random.rand(T) * strength
            for u in us:
                v = pstate 
                if v < 0 or v > 1:
                    v = u
                s_prep = v * s0 + (1.0 - v) * s1
                ts = tc * s_prep
                #print(ts.dims, ts.shape, ts.iscp, ts.istp, ts.iscptp)
                rho = ts * rho
                rh1 =  vector_to_operator(rho)
                #print(x, rh1.shape, rh1.type, rh1.tr())
                obmx = (Mx * rh1).tr()
                obmy = (Mz * rh1).tr()
                mxs.append(np.real(obmx))
                mzs.append(np.real(obmy))

                # fs_sp.append(fidelity(rho_sp, rh1))
                # fs_su.append(fidelity(rho_su, rh1))

                # tr_sp.append(tracedist(rho_sp, rh1))
                # tr_su.append(tracedist(rho_su, rh1))

            #print('after xs: ', rh1.shape, rh1.type, rh1.tr())

            tr_spls.append(tr_sp)
            tr_suls.append(tr_su)
            fid_spls.append(fs_sp)
            fid_suls.append(fs_su)
            mx_ls.append(mxs)
            mz_ls.append(mzs)

    # Plot file
    if args.plot > 0:
        plt.rc('font', family='serif', size=12)
        plt.rc('mathtext', fontset='cm')
        fig = plt.figure(figsize=(20, 20), dpi=600)
        outbase = os.path.join(savedir, basename)
        xs = list(range(T))

        # Plot fidelity and distance line
        ax1 = plt.subplot2grid((4,1), (0,0), colspan=1, rowspan=1)
        # ax1.plot(xs, fs, 'o-', label='Fidelity-ss')
        # ax1.plot(xs, fs_sp, 'o-', label='Fs_sp')
        # ax1.plot(xs, fs_su, 'o-', label='Fs_su')

        for i in range(len(tr_spls)):
            if i == 0:
                ax1.plot(xs, tr_spls[i],  color=VERMILLION, alpha=0.7, label='Basis 0')
                ax1.plot(xs, tr_suls[i],  color=BLUE, alpha=0.7, label='Basis 1')
            else:
                ax1.plot(xs, tr_spls[i], color=VERMILLION, alpha=0.7)
                ax1.plot(xs, tr_suls[i], color=BLUE, alpha=0.7)

        ax1.set_xlabel('$T$')
        ax1.legend()  
        ax1.set_title('Trace: {}'.format(outbase))

        ax2 = plt.subplot2grid((4,1), (1,0), colspan=1, rowspan=1)
        for i in range(len(fid_spls)):
            if i == 0:
                ax2.plot(xs, fid_spls[i], color=VERMILLION, alpha=0.7, label='Basis 0')
                ax2.plot(xs, fid_suls[i], color=BLUE, alpha=0.7, label='Basis 1')
            else:
                ax2.plot(xs, fid_spls[i], color=VERMILLION, alpha=0.7)
                ax2.plot(xs, fid_suls[i], color=BLUE, alpha=0.7)

        ax2.set_xlabel('$T$')
        ax2.legend()  
        ax2.set_title('Fidelity: {}'.format(outbase))

        ax3 = plt.subplot2grid((4,1), (2,0), colspan=1, rowspan=1)
        for i in range(len(fid_spls)):
            if i == 0:
                ax3.plot(xs, mx_ls[i], color=VERMILLION, alpha=0.7, label='Mx')
            else:
                ax3.plot(xs, mx_ls[i], color=VERMILLION, alpha=0.7)

        ax3.set_xlabel('$T$')
        ax3.legend()  
        ax3.set_title('Average magnezation $\langle M_x\\rangle$: {}'.format(outbase))

        ax4 = plt.subplot2grid((4,1), (3,0), colspan=1, rowspan=1)
        for i in range(len(fid_spls)):
            if i == 0:
                ax4.plot(xs, mz_ls[i], color=BLUE, alpha=0.7, label='Mz')
            else:
                ax4.plot(xs, mz_ls[i], color=BLUE, alpha=0.7)

        ax4.set_xlabel('$T$')
        ax4.legend()  
        ax4.set_title('Average magnezation $\langle M_z\\rangle$: {}'.format(outbase))

        for ftype in ['png']:
            plt.savefig('{}_v1.{}'.format(outbase, ftype), bbox_inches='tight', dpi=600)
        plt.show()

  