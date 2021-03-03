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
    parser.add_argument('--nspins', type=int, default=5, help='Number of total spins')
    parser.add_argument('--nenvs', type=int, default=1, help='Number of env spins')
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

    Nspins, Nenvs, alpha, bc, J, Ntrials = args.nspins, args.nenvs, args.alpha, args.bcoef, args.max_energy, args.Ntrials
    tauB, T, pstate, strength = args.tauB, args.Tsteps, args.pstate, args.strength
    savedir, seed, binary = args.savedir, args.seed, args.binary
    basename = '{}_spins_{}_envs_{}_trials_{}_seed_{}_strength_{}_pstate_{:.2f}_a_{}_bc_{}_tauB_{}_T_{}_bin_{}'.format(args.basename, \
        Nspins, Nenvs, Ntrials, seed, strength, pstate, alpha, bc, tauB, T, binary)
    
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    if os.path.isfile(savedir) == False:
        logdir = os.path.join(savedir, 'log')
        if os.path.isdir(logdir) == False:
            os.mkdir(logdir)
        log_filename = os.path.join(logdir, '{}.log'.format(basename))
        logger = get_module_logger(__name__, log_filename)
        logger.info(log_filename)
        logger.info('Nspins={},Nenvs={},pstate={},binary={},strength={},alpha={},bcoef={},tauB={},Tsteps={},Ntrials={},seed={}'.format(\
            Nspins, Nenvs,\
            pstate, binary, strength, alpha, bc, tauB, T, Ntrials, seed))

        B = J/bc # Magnetic field
        tau = tauB/B
        L, Mx, My, Mz  = getLiouv_IsingOpen(Nspins, alpha, B, Nspins-Nenvs, J)
        S = (tau*L).expm()

        # swap with environment
        q0 = getSci(basis(2, 0), 0, Nspins)
        q1 = getSci(basis(2, 1), 0, Nspins)
        s0 = sprepost(q0, q0.dag())
        s1 = sprepost(q1, q1.dag())
        if Nenvs == 1:
            tc = tensor_contract(S, (0, Nspins))
        elif Nenvs == 2:
            tc = tensor_contract(S, (0, Nspins), (1, Nspins + 1))
        elif Nenvs == 3:
            tc = tensor_contract(S, (0, Nspins), (1, Nspins + 1), (2, Nspins + 2))
        else:
            tc = tensor_contract(S, (0, Nspins))
        nobs = Nspins - Nenvs
        
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
            rkets = []
            for m in range(Nenvs):
                rkets.append(rand_ket(2))
            rstates.append(rkets)

        iop = identity(2)
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
                rkets = rstates[n]
                q = rkets[0]
                for j in range(Nenvs - 1):
                    q = tensor(q, rkets[j+1])
                for j in range(Nspins - Nenvs):
                    q = tensor(q, iop)
                
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

    # min, max scale
    mx_ls = np.array(mx_ls)
    my_ls = np.array(my_ls)
    mz_ls = np.array(mz_ls)

    mx_ls = (mx_ls - np.min(mx_ls)) / (np.max(mx_ls) - np.min(mx_ls)) * 2 - 1.0
    my_ls = (my_ls - np.min(my_ls)) / (np.max(my_ls) - np.min(my_ls)) * 2 - 1.0
    mz_ls = (mz_ls - np.min(mz_ls)) / (np.max(mz_ls) - np.min(mz_ls)) * 2 - 1.0

    # Plot file
    if args.plot > 0:
        plt.rc('font', family='serif')
        plt.rc('mathtext', fontset='cm')
        plt.rcParams["font.size"] = 20 # 全体のフォントサイズが変更されます
        plt.rcParams['xtick.labelsize'] = 24 # 軸だけ変更されます
        plt.rcParams['ytick.labelsize'] = 24 # 軸だけ変更されます
        fig = plt.figure(figsize=(30, 3), dpi=600)
        outbase = os.path.join(savedir, basename)
        xs = list(range(1, T+1))

        mlss = [mx_ls, my_ls, mz_ls]
        lbs = ['m_x', 'm_y', 'm_z']
        cols = [VERMILLION, BLUE, GREEN]
        # viridis = cm.get_cmap('viridis', Ntrials)
        # cmaps = viridis(np.linspace(0, 1, Ntrials))

        for j in range(3):
            jnext = (j+1)%3
            mls, label, col = mlss[j], lbs[j], cols[j]
            mls_next, label_next, col_next = mlss[jnext], lbs[jnext], cols[jnext]
            
            ax = plt.subplot2grid((1, 3), (0, j), colspan=1, rowspan=1)
            for i in range(len(mls)):
                if i == 0:
                    ax.plot(xs, mls[i], color=col, alpha=0.7, label=label)
                else:
                    ax.plot(xs, mls[i], color=col, alpha=0.7)

            ax.set_xlabel('$n$', fontsize=24)
            ax.set_ylabel(lbs[j], fontsize=24)
            #ax.set_yticklabels([])
            ax.tick_params(axis='x', which='both', direction='out', length=6)
            ax.tick_params(axis='y', which='both', direction='out', length=6)
            #ax.legend()  
            #ax.set_title('Average magnezation $\langle {}\\rangle$: {}'.format(label, outbase))

        for ftype in ['png','pdf','svg']:
            plt.savefig('{}_scaled.{}'.format(outbase, ftype), bbox_inches='tight', dpi=600)
        plt.show()

  