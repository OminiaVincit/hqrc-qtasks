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
    parser.add_argument('--strength', type=float, default=1.0, help='Input strength')
    parser.add_argument('--tmax', type=float, default=25.0, help='Maximum of tauB')
    parser.add_argument('--tmin', type=float, default=0.0, help='Minimum of tauB')
    parser.add_argument('--ntaus', type=int, default=250, help='Number of discrete evolution')
    parser.add_argument('--Ntrials', type=int, default=10, help='Number of trials')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha of coupled strength')
    parser.add_argument('--bcoef', type=float, default=1.0)
    parser.add_argument('--max_energy', type=float, default=1.0)

    parser.add_argument('--savedir', type=str, default='qtrans')
    parser.add_argument('--basename', type=str, default='trans')
    parser.add_argument('--plot', type=int, default=1, help='Flag to plot')
    parser.add_argument('--seed', type=int, default=0, help='Seed for random')
    args = parser.parse_args()
    print(args)

    Nspins, alpha, bc, J, Ntrials = args.nspins, args.alpha, args.bcoef, args.max_energy, args.Ntrials
    tmin, tmax, ntaus, strength = args.tmin, args.tmax, args.ntaus, args.strength
    savedir, seed = args.savedir, args.seed
    basename = '{}_spins_{}_trials_{}_seed_{}_strength_{}_a_{}_bc_{}_tmax_{}_tmin_{}_ntaus_{}'.format(args.basename, \
        Nspins, Ntrials, seed, strength, alpha, bc, tmax, tmin, ntaus)
    
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    if os.path.isfile(savedir) == False:
        logdir = os.path.join(savedir, 'log')
        if os.path.isdir(logdir) == False:
            os.mkdir(logdir)
        log_filename = os.path.join(logdir, '{}.log'.format(basename))
        logger = get_module_logger(__name__, log_filename)
        logger.info(log_filename)
        logger.info('Nspins={},strength={},alpha={},bcoef={},tmax={},tmin={},ntaus={},Ntrials={},seed={}'.format(Nspins, \
            strength, alpha, bc, tmax, tmin, ntaus, Ntrials, seed))


        
        tauB = (tmax - tmin) / ntaus
        Nalpha = getNormCoef(Nspins, alpha)
        # Wrong setting
        # B = J / bc 

        # Setting from papers (B. Zunkovic, PRL 120, 130601 (2018) DQPT)
        B = J / bc
        L, Mx, Mz = getLiouv_IsingOpen(Nspins, alpha, B, Nspins-1, J)
        
        tau = tauB / B
        S = (tau*L).expm()
        tc = tensor_contract(S, (0, Nspins))
        print('TC', tc.type, tc.shape)
                
        nobs = Nspins 
        
        # Create two basis density matrix
        sp = (basis(2, 0)).unit()
        su = (basis(2, 1)).unit()
        print('sp, su: ', sp, su)
        rho_sp = ket2dm(sp)
        rho_su = ket2dm(su)
        print('rho_sp', 'rho_su', rho_sp, rho_su)
        print('rho_sp', 'rho_su', rho_sp.dims, rho_su.dims)

        # Initialize density matrix
        mx_ls, mz_ls = [], []
        
        np.random.seed(seed=abs(seed))

        for i in range(Ntrials):
            betas = np.random.rand(nobs)
            #betas = np.zeros(nobs) + 0.5
            print(i, betas)

            if seed >= 0:
                print('Generate pure states')
                rho = basis(2, 0) * np.sqrt(betas[0]) + basis(2, 1) * np.sqrt(1.0 - betas[0])
                rho = rho.unit()
                for j in range(1, len(betas)):
                    tmp = basis(2, 0) * np.sqrt(betas[j]) + basis(2, 1) * np.sqrt(1.0 - betas[j])
                    tmp = tmp.unit()
                    rho = tensor(rho, tmp)
                rho = ket2dm(rho)
            else:
                print('Generate mixed matrices')
                #rho = rand_dm(2**nobs, density=0.2, dims=rho_sp.dims)
                rho = rho_sp * betas[0] + rho_su * (1.0 - betas[0])
                for j in range(1, len(betas)):
                    tmp = rho_sp * betas[j] + rho_su * (1.0 - betas[j])
                    rho = tensor(rho, tmp)

            print(i, 'rho', rho.shape, rho.type, rho.tr(), rho.dims)
            rho = operator_to_vector(rho)
            
            mxs, mzs, xs = [], [], []
            for n in range(ntaus):
                rho = S * rho
                rha = tc * rho
                #print(n, rha.shape, rha.type, rha.dims)
                rh1 =  vector_to_operator(rha)
                #print(n, rh1.shape, rh1.type, rh1.tr())
                obmx = (Mx * rh1).tr()
                obmy = (Mz * rh1).tr()
                mxs.append(np.real(obmx))
                mzs.append(np.real(obmy))
                xs.append(tmin + tauB * (n+1))

            mx_ls.append(mxs)
            mz_ls.append(mzs)

    # Plot file
    if args.plot > 0:
        plt.rc('font', family='serif', size=16)
        plt.rc('mathtext', fontset='cm')
        fig =  plt.figure(figsize=(24, 16), dpi=600)
        outbase = os.path.join(savedir, basename)
        
        ax1 = plt.subplot2grid((2, 1), (0,0), colspan=1, rowspan=1)
        for i in range(len(mx_ls)):
            ax1.plot(xs, mx_ls[i], color=GREEN, alpha=0.5, linestyle='dashdot')
        
        ax1.plot(xs, np.mean(mx_ls, axis=0).ravel(), color=VERMILLION, alpha=1.0, label='Average Mx', linewidth=3.0)
            

        ax1.set_xlabel('$\\tau B$', size=24)
        ax1.set_ylabel('$\langle M_x\\rangle$', size=24)
        ax1.axhline(y=0, color='k', linestyle='-')

        ax1.legend()  
        ax1.set_title('{}'.format(outbase), size=14)
        ax1.set_xlim([tmin, tmax])
        #ax1.set_xticks(list(np.linspace(tmin, tmax, int(tmax) + 1)))
        ax1.grid(which='major',color='black',linestyle='-', axis='x')
        #ax1.grid(which='minor',color='black',linestyle='-')

        ax2 = plt.subplot2grid((2, 1), (1,0), colspan=1, rowspan=1)
        for i in range(len(mz_ls)):
            ax2.plot(xs, mz_ls[i], color=GREEN, alpha=0.5, linestyle='dashdot')
        ax2.plot(xs, np.mean(mz_ls, axis=0).ravel(), color=BLUE, alpha=1.0, label='Average Mz', linewidth=3.0)

        ax2.set_xlabel('$\\tau B$', size=24)
        ax2.set_ylabel('$\langle M_z\\rangle$', size=24)
        ax2.legend()  
        ax2.set_title('{}'.format(outbase), size=14)
        ax2.set_xlim([tmin, tmax])
        #ax2.set_xticks(list(np.linspace(tmin, tmax, int(tmax)  + 1)))
        ax2.grid(which='major',color='black',linestyle='-', axis='x')
        #ax2.grid(which='minor',color='black',linestyle='-')

        plt.tight_layout()

        for ftype in ['png']:
            plt.savefig('{}_v1.{}'.format(outbase, ftype), bbox_inches='tight', dpi=600)
        plt.show()

  