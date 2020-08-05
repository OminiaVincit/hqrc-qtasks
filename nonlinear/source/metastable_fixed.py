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
    parser.add_argument('--pstate', type=float, default=0.5, help='Mixed coefficient in the swap state, in [0, 1], -1 is random')
    parser.add_argument('--tau', type=float, default=10.0, help='Time between the input')
    parser.add_argument('--Tsteps', type=int, default=100, help='Number of time steps')
    parser.add_argument('--Ntrials', type=int, default=10, help='Number of trials')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha of coupled strength')
    parser.add_argument('--bcoef', type=float, default=0.42)

    parser.add_argument('--savedir', type=str, default='metastable')
    parser.add_argument('--basename', type=str, default='meta')
    parser.add_argument('--plot', type=int, default=1, help='Flag to plot')
    parser.add_argument('--seed', type=int, default=0, help='Seed for random')
    args = parser.parse_args()
    print(args)

    Nspins, alpha, bc, Ntrials = args.nspins, args.alpha, args.bcoef, args.Ntrials
    tau, T, pstate = args.tau, args.Tsteps, args.pstate
    savedir, seed = args.savedir, args.seed
    basename = '{}_nspins_{}_ntrials_{}_seed_{}_pstate_{:.2f}_a_{}_bc_{}_tau_{}_T_{}'.format(args.basename, Nspins, Ntrials, seed, pstate, alpha, bc, tau, T)
    
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    if os.path.isfile(savedir) == False:
        logdir = os.path.join(savedir, 'log')
        if os.path.isdir(logdir) == False:
            os.mkdir(logdir)
        log_filename = os.path.join(logdir, '{}.log'.format(basename))
        logger = get_module_logger(__name__, log_filename)
        logger.info(log_filename)
        logger.info('Nspins={},pstate={},alpha={},bcoef={},tau={},Tsteps={},Ntrials={},seed={}'.format(Nspins, \
            pstate, alpha, bc, tau, T, Ntrials, seed))

        L, Mx, Mz = getLiouv_IsingOpen(Nspins, alpha, bc)
        S = (tau*L).expm()

        # swap with environment
        q0 = getSci(basis(2, 0), 0, Nspins)
        q1 = getSci(basis(2, 1), 0, Nspins)
        s0 = sprepost(q0, q0.dag())
        s1 = sprepost(q1, q1.dag())
        
        # t1 = tensor_contract(S, (0, Nspins)) * s0
        # t2 = tensor_contract(S, (0, Nspins)) * s1
        
        s_prep = pstate * s0 + (1.0-pstate) * s1
        ts = tensor_contract(S, (0, Nspins)) * s_prep
        print(ts.dims, ts.shape, ts.iscp, ts.istp, ts.iscptp)

        ev = ts.eigenstates()[1]
        dims = ev[-1].dims
        
        w, vl, vr = eigenstates(ts.full())
        rhoss = Qobj(vr[:, 0], type='ket', dims=dims)
        print('rhoss', rhoss.dims, rhoss.shape, rhoss.type, ev[-1].dims)
        rhoss = vector_to_operator(rhoss)
        # print('ev[-1]', ev[-1].dims, ev[-1].shape, ev[-1].type)
        #rhoss = vector_to_operator(ev[-1])
        rhoss = rhoss / rhoss.tr()
        print('rhoss', rhoss.dims, rhoss.shape, rhoss.type, rhoss.tr())

        ldl2 = Qobj(vl[:, 1], type='ket', dims=dims)
        ldl2 = vector_to_operator(ldl2)
        ldl2 = ldl2 / ldl2.tr()
        wl, _, _ = eigenstates(ldl2.full())
        ws = np.sort(np.real(wl))

        c2min, c2max = ws[0], ws[-1]
        chro1 = rhoss + c2max*ldl2
        chro2 = rhoss + c2min*ldl2

        # chro1 = vector_to_operator(chro1)
        # chro2 = vector_to_operator(chro2)

        chro1 = chro1 / chro1.tr()
        chro2 = chro2 / chro2.tr()
        print(c2max, c2min)

        nobs = Nspins - 1
        I = Qobj(identity(2**nobs), type='op', dims=ldl2.dims)
        cP1 = (ldl2 - c2min*I)/(c2max - c2min)
        cP2 = (-ldl2 + c2max*I)/(c2max - c2min)
        # rho1 = ts * ev[-1]
        # print('rho1', rho1.dims, rho1.shape, rho1.type)
        # Definite the initial state
        #rho = rand_dm(2**(Nspins-1), density=0.3)

        # Create two basis density matrix
        sp = (basis(2, 0)).unit()
        su = (basis(2, 1)).unit()
        rho_sp = ket2dm(tensor(sp, sp, sp, sp))
        rho_su = ket2dm(tensor(su, su, su, su))
        print('rho_sp', 'rho_su', rho_sp.tr(), rho_su.tr())

        # Initialize density matrix
        trace_ls, tr_spls, tr_suls, tr_smls = [], [], [], []
        fid_ls, fid_spls, fid_suls, fid_smls = [], [], [], []
        
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
            
            p1 = (cP1*rho).tr()
            p2 = (cP2*rho).tr()
            chrom = p1 * chro1 + p2 * chro2
            
            print(i, rho.shape, rho.type, rho.tr(), p1+p2, chrom.tr())
            rho = operator_to_vector(rho)
            #print(ts.shape, rho.dims, rho.shape, rho.type)
            #rho = ev[-1]
            fs, traces = [], []
            fs_sp, fs_su, tr_sp, tr_su = [], [], [], []
            fs_sm, tr_sm = [], []
            xs = list(range(T))
            avgm = []
            for x in xs:
                rho = ts * rho
                rh1 =  vector_to_operator(rho)
                #print(x, rh1.shape, rh1.type, rh1.tr())
                # obmx = (Mx * rh1).tr()
                # obmy = (Mz * rh1).tr()
                # avgm.append([x, abs(obmx), abs(obmy)])

                fs.append(fidelity(rhoss, rh1))
                traces.append(tracedist(rhoss, rh1))

                fs_sp.append(fidelity(chro1, rh1))
                fs_su.append(fidelity(chro2, rh1))
                fs_sm.append(fidelity(chrom, rh1))

                tr_sp.append(tracedist(chro1, rh1))
                tr_su.append(tracedist(chro2, rh1))
                tr_sm.append(tracedist(chrom, rh1))
            #avgms.append(np.array(avgm))
            print('after xs: ', rh1.shape, rh1.type, rh1.tr())

            trace_ls.append(traces)
            tr_spls.append(tr_sp)
            tr_suls.append(tr_su)
            tr_smls.append(tr_sm)

            fid_ls.append(fs)
            fid_spls.append(fs_sp)
            fid_suls.append(fs_su)
            fid_smls.append(fs_sm)
        print(c2max, c2min, tracedist(chro1, chro2))
        # Plot file
    if args.plot > 0:
        plt.rc('font', family='serif', size=12)
        plt.rc('mathtext', fontset='cm')
        fig = plt.figure(figsize=(20, 12), dpi=600)
        outbase = os.path.join(savedir, basename)

        # Plot fidelity and distance line
        ax1 = plt.subplot2grid((2,3), (0,0), colspan=2, rowspan=1)
        # ax1.plot(xs, fs, 'o-', label='Fidelity-ss')
        # ax1.plot(xs, fs_sp, 'o-', label='Fs_sp')
        # ax1.plot(xs, fs_su, 'o-', label='Fs_su')

        for i in range(len(trace_ls)):
            if i == 0:
                ax1.plot(xs, trace_ls[i], color='k', alpha=0.7, label='SS')
                ax1.plot(xs, tr_spls[i],  color=GREEN, alpha=0.7, label='c2max={}'.format(c2max))
                ax1.plot(xs, tr_suls[i],  color=BLUE, alpha=0.7, label='c2min={}'.format(c2min))
                ax1.plot(xs, tr_smls[i],  color=VERMILLION, alpha=0.7, label='MM')
            else:
                ax1.plot(xs, trace_ls[i], color='k', alpha=0.7)
                ax1.plot(xs, tr_spls[i], color=GREEN, alpha=0.7)
                ax1.plot(xs, tr_suls[i], color=BLUE, alpha=0.7)
                ax1.plot(xs, tr_smls[i],  color=VERMILLION, alpha=0.7)
            #ax1.plot(xs, tr_sp, 'o-', label='Tr_sp')
            #ax1.plot(xs, tr_su, 'o-', label='Tr_su')

        ax1.set_xlabel('$T$')
        ax1.legend()  
        ax1.set_title('Trace: {}'.format(outbase))

        ax2 = plt.subplot2grid((2,3), (1,0), colspan=2, rowspan=1)
        # ax1.plot(xs, fs, 'o-', label='Fidelity-ss')
        # ax1.plot(xs, fs_sp, 'o-', label='Fs_sp')
        # ax1.plot(xs, fs_su, 'o-', label='Fs_su')

        for i in range(len(fid_ls)):
            ax2.plot(xs, fid_ls[i], color='k', alpha=0.7)
            ax2.plot(xs, fid_spls[i], color=GREEN, alpha=0.7)
            ax2.plot(xs, fid_suls[i], color=BLUE, alpha=0.7)
            ax2.plot(xs, fid_smls[i], color=VERMILLION, alpha=0.7)

            # avgm = avgms[i]
            # ax2.plot(avgm[:, 0], avgm[:, 1], label='Mx', color='k')
            # ax2.plot(avgm[:, 0], avgm[:, 2], label='Mz', color='r')
        ax2.set_title('Fidelity: {}'.format(outbase))

        # Plot distribution of eigenvalues
        ax3 = plt.subplot2grid((2,3), (0,2), colspan=1, rowspan=1)
        circle = Circle((0, 0), 1.0)
        p = PatchCollection([circle], cmap=matplotlib.cm.jet, alpha=0.1)
        ax3.add_collection(p)
        ax3.axis('equal')
        for xi, yi in zip(np.real(w), np.imag(w)):
            ax3.plot(xi, yi, 'o', color='k', alpha=0.7)
        ax3.set_title('Eigenvalue dist')
        ax3.set_xlabel('Real')
        ax3.set_ylabel('Img')
        
        ax4 = plt.subplot2grid((2,3), (1,2), colspan=1, rowspan=1)
        for i in range(len(w)):
            ax4.plot(i, abs(w[i]), 'o', color='k', alpha=0.7)
        ax4.set_xlabel('Index')
        ax4.set_ylabel('Abs')
        ax4.set_title('Spectral')

        # ax5 = plt.subplot2grid((3,3), (2,0), colspan=2, rowspan=1)
        # for i in range(len(avgms)):
        #     avgm = avgms[i]
        #     if i == 0:
        #         ax5.plot(avgm[:, 0], avgm[:, 1], color='r', alpha=0.7, label='Mx')
        #         ax5.plot(avgm[:, 0], avgm[:, 2], color='b', alpha=0.7, label='Mz')
        #     else:
        #         ax5.plot(avgm[:, 0], avgm[:, 1], color='r', alpha=0.7)
        #         ax5.plot(avgm[:, 0], avgm[:, 2], color='b', alpha=0.7)
        # ax5.set_xlabel('T')
        # ax5.set_title('Average magnetization')
        # ax5.legend()

        for ftype in ['png']:
            plt.savefig('{}_v2.{}'.format(outbase, ftype), bbox_inches='tight', dpi=600)
        plt.show()

  