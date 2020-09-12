# Run export OMP_NUM_THREADS=1
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from loginit import get_module_logger
import multiprocessing
from collections import defaultdict
import os

def getSci(sc, i, Nspins):
    iop = identity(2)
    sci = iop
    if i == 0:
        sci = sc
    for j in range(1, Nspins):
        tmp = iop
        if j == i:
            tmp = sc
        sci = tensor(sci, tmp)
    return sci

def dump_eigenval_job(savedir, basename, tauls, L, s_prep, idx, send_end):
    """
    Dump eigenvalues of superoperator
    """
    print('Start pid={} with size {} (from {} to {})'.format(idx, len(tauls), tauls[0], tauls[-1]))
    results = dict()
    for tau in tauls:
        S = (tau*L).expm()
        ts = tensor_contract(S, (0, Nspins)) * s_prep
        # if tau == 1.0:
        #     print(tau, ts.shape, ts.iscp, ts.istp, ts.iscptp)
        egvals = ts.eigenstates()[0] # Eigenvalues sorted from low to high (magnitude)
        results[tau] = egvals
    filename = '{}_eig_id_{}.binaryfile'.format(basename, idx)
    filename = os.path.join(savedir, filename)
    with open(filename, 'wb') as wrs:
        pickle.dump(results, wrs)
    send_end.send(filename)
    print('Finished pid={} with size {} (from {} to {})'.format(idx, len(tauls), tauls[0], tauls[-1]))
    
if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--nspins', type=int, default=5, help='Number of spins')
    parser.add_argument('--pstate', type=float, default=0, help='Mixed coefficient in the swap state')
    parser.add_argument('--tmax', type=float, default=50, help='Maximum of tau')
    parser.add_argument('--ntaus', type=int, default=51, help='Number of taus')
    parser.add_argument('--nproc', type=int, default=51)
    parser.add_argument('--max_energy', type=float, default=1.0, help='Magnitude coupled strength')
    parser.add_argument('--savedir', type=str, default='spectral_random')
    parser.add_argument('--basename', type=str, default='spec')
    parser.add_argument('--nc', type=int, default=5, help='Number of components')
    parser.add_argument('--plot', type=int, default=0, help='Flag to plot')
    args = parser.parse_args()
    print(args)

    np.random.seed(seed=0)

    Nspins, max_energy = args.nspins, args.max_energy
    tmax, ntaus, nc, pstate = args.tmax, args.ntaus, args.nc, args.pstate
    basename = '{}_nspins_{}_state_{:.2f}_J_{}_tmax_{}_ntaus_{}'.format(args.basename, Nspins, pstate, max_energy, tmax, ntaus)
    savedir = args.savedir
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    if os.path.isfile(savedir) == False:
        logdir = os.path.join(savedir, 'log')
        if os.path.isdir(logdir) == False:
            os.mkdir(logdir)
        log_filename = os.path.join(logdir, '{}.log'.format(basename))
        logger = get_module_logger(__name__, log_filename)
        logger.info(log_filename)
        logger.info('Nspins={},pstate={},max_energy={},tmax={},ntaus={}'.format(Nspins, pstate, max_energy, tmax, ntaus))

        X = sigmax()
        Z = sigmaz()
        I = identity(2)

        # Create Hamiltonian
        H0 = getSci(I, 0, Nspins) * 0.0
        H1 = getSci(I, 0, Nspins) * 0.0
        for i in range(Nspins):
            Szi = getSci(Z, i, Nspins)
            H0 = H0 + (np.random.rand()-0.5) * 2 * max_energy * Szi # Hamiltonian for the magnetic field
            for j in range(i+1, Nspins):
                Sxi = getSci(X, i, Nspins)
                Sxj = getSci(X, j, Nspins)
                hij = (np.random.rand()-0.5) * 2 * max_energy
                H1 = H1 + hij * Sxi * Sxj # Interaction Hamiltonian
        H = H0 + H1 # Total Hamiltonian
        L = liouvillian(H, [])

        # swap with environment
        q0 = getSci(basis(2, 0), 0, Nspins)
        q1 = getSci(basis(2, 1), 0, Nspins)
        
        s_prep = pstate * sprepost(q0, q0.dag()) + (1.0-pstate) * sprepost(q1, q1.dag())

        #tauls = list(np.linspace(0.0, tmax, ntaus))
        tx = list(np.arange(-7, 5.001, 0.01))
        tauls = [2**x for x in tx]

        nproc = min(len(tauls), args.nproc)
        lst = np.array_split(tauls, nproc)

        jobs, pipels = [], []
        for pid in range(nproc):
            ts = lst[pid]
            recv_end, send_end = multiprocessing.Pipe(False)
            p = multiprocessing.Process(target=dump_eigenval_job, \
                args=(savedir, basename, ts, L, s_prep, pid, send_end))
            jobs.append(p)
            pipels.append(recv_end)

        # Start the process
        for p in jobs:
            p.start()

        # Ensure all processes have finished execution
        for p in jobs:
            p.join()

        # Join dumbpled pickle files
        z = dict()
        for px in pipels:
            filename = px.recv()
            with open(filename, 'rb') as rrs:
                tmp = pickle.load(rrs)
                z = dict(list(z.items()) + list(tmp.items()))
            # Delete file
            os.remove(filename)
            print('zlen={}, Deleted {}'.format(len(z), filename))

        filename = filename.replace('.binaryfile', '_tot.binaryfile')
        with open(filename, 'wb') as wrs:
            pickle.dump(z, wrs)
    else:
        # Loadfile
        filename = savedir
        with open(filename, 'rb') as rrs:
            z = pickle.load(rrs)
    
    print(z.keys())
    # Plot file
    if args.plot > 0:
        plt.rc('font', family='serif', size=12)
        plt.rc('mathtext', fontset='cm')
        fig = plt.figure(figsize=(8, 8), dpi=600)
        
        # Plot Nspins largest eigenvectors
        ax1 = plt.subplot2grid((2,1), (0,0), colspan=1, rowspan=1)
        ax1.set_title('{} spec; {}'.format(nc, os.path.basename(filename)), size=8)

        xs = []
        ysd = defaultdict(list)
        for tau in z.keys():
            xs.append(tau)
            egvals = z[tau]
            egvals = sorted(egvals, key=abs)
            for n in range(nc):
                ysd[n].append(np.abs(egvals[-(n+1)]))
        for n in range(nc):
            print(n, len(ysd[n]))
            ax1.plot(xs, ysd[n], 'o-', label='{}th'.format(n+1), alpha=0.8)
        #ax1.set_xlabel('$\\tau$', fontsize=16)
        ax1.legend()
        
        # # Plot 1/|lambda_2|, |lambda_2| / |lambda_3|
        ax2 = plt.subplot2grid((2,1), (1,0), colspan=1, rowspan=1)
        ax2.set_title('$1/|\lambda_2|$ and $|\lambda_2| / |\lambda_3|$')
        ild2 = [1.0/a for a in ysd[1]]
        ld23 = [ysd[1][i]/ysd[2][i] for i in range(min(len(ysd[1]), len(ysd[2])))]
        ax2.plot(xs, ysd[0], label='$|\lambda_1|$')
        ax2.plot(xs, ild2, 'o-', label='$1/|\lambda_2|$', alpha=0.8)
        ax2.plot(xs, ld23, 'o-', label='$|\lambda_2| / |\lambda_3|$', alpha=0.8)
        ax2.set_xlabel('$\\tau$', fontsize=16)
        ax2.legend()
        
        outbase = filename.replace('.binaryfile', '')
        for ftype in ['png']:
            plt.savefig('{}_v2.{}'.format(outbase, ftype), bbox_inches='tight', dpi=600)
        plt.show()



