# Run export OMP_NUM_THREADS=1
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from datetime import timedelta
import argparse
from loginit import get_module_logger
import multiprocessing
from collections import defaultdict
import os
from utils import *
from qutils import *

def dump_eigenval_job(savedir, basename, Nspins, Nenvs, tauls, L, B, s_prep, idx, send_end):
    """
    Dump eigenvalues of superoperator
    """
    print('Start pid={} with size {} (from {} to {})'.format(idx, len(tauls), tauls[0], tauls[-1]))
    results = dict()
    for tauB in tauls:
        tau = tauB / B
        S = (tau*L).expm()
        if Nenvs == 1:
            ts = tensor_contract(S, (0, Nspins))
        elif Nenvs == 2:
            ts = tensor_contract(S, (0, Nspins), (1, Nspins + 1))
        elif Nenvs == 3:
            ts = tensor_contract(S, (0, Nspins), (1, Nspins + 1), (2, Nspins + 2))
        else:
            return
        ts = ts * s_prep
        # if tau == 1.0:
        #     print(tau, ts.shape, ts.iscp, ts.istp, ts.iscptp)
        egvals = ts.eigenstates()[0] # Eigenvalues sorted from low to high (magnitude)
        results[tauB] = egvals
    filename = '{}_eig_id_{}.binaryfile'.format(basename, idx)
    filename = os.path.join(savedir, filename)
    with open(filename, 'wb') as wrs:
        pickle.dump(results, wrs)
    send_end.send(filename)
    print('Finished pid={} with size {} (from {} to {}) Num eigs = {}'.format(idx, len(tauls), tauls[0], tauls[-1], len(egvals)))

def plot_eigenvalues(z, nc, filename):
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

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--nspins', type=int, default=5, help='Number of spins')
    parser.add_argument('--nenvs', type=int, default=1, help='Number of env spins')
    parser.add_argument('--tmin', type=float, default=0, help='Maximum of tauB')
    parser.add_argument('--tmax', type=float, default=25, help='Maximum of tauB')
    parser.add_argument('--ntaus', type=int, default=250, help='Number of tausB')
    parser.add_argument('--nproc', type=int, default=125)
    parser.add_argument('--bgidx', type=int, default=0)
    parser.add_argument('--edidx', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha of coupled strength, 0 for random coupling')
    parser.add_argument('--bcoef', type=float, default=2.0)
    parser.add_argument('--max_energy', type=float, default=1.0)
    parser.add_argument('--savedir', type=str, default='spectral')
    parser.add_argument('--basename', type=str, default='spec')
    parser.add_argument('--nc', type=int, default=5, help='Number of components')
    parser.add_argument('--plot', type=int, default=0, help='Flag to plot')
    args = parser.parse_args()
    print(args)

    start_time = time.monotonic()

    J, Nspins, Nenvs, alpha, bc = args.max_energy, args.nspins, args.nenvs, args.alpha, args.bcoef
    tmin, tmax, ntaus, nc, edidx, bgidx = args.tmin, args.tmax, args.ntaus, args.nc, args.edidx, args.bgidx
    basename = '{}_nspins_{}_nenvs_{}_a_{}_bc_{}_tmin_{}_tmax_{}_ntaus_{}'.format(args.basename, Nspins, Nenvs, alpha, bc, tmin, tmax, ntaus)

    np.random.seed(seed=args.seed)

    savedir = args.savedir
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    if os.path.isfile(savedir) == False:
        bindir = os.path.join(savedir, 'binary')
        if os.path.isdir(bindir) == False:
                os.mkdir(bindir)

        logdir = os.path.join(savedir, 'log')
        if os.path.isdir(logdir) == False:
            os.mkdir(logdir)
        log_filename = os.path.join(logdir, '{}.log'.format(basename))
        logger = get_module_logger(__name__, log_filename)
        logger.info(log_filename)
        logger.info('Nspins={},Nenvs={},alpha={},bcoef={},tmin={},tmax={},ntaus={}'.format(Nspins, Nenvs, alpha, bc, tmin, tmax, ntaus))

        # Create coupling strength
        Nalpha = getNormCoef(Nspins, alpha)
        B = J/bc # Magnetic field

        tauls = list(np.linspace(tmin, tmax, ntaus + 1))
        tauls = tauls[1:]
        #tx = list(np.arange(-7, 14.1, 0.05))
        #tauls = [2**x for x in tx]

        nproc = min(len(tauls), args.nproc)
        lst = np.array_split(tauls, nproc)
        L, _, _, _ = getLiouv_IsingOpen(Nspins, alpha, B, Nspins-Nenvs, J)

        # swap with environment
        #rhos = generate_one_qubit_states(ranseed=args.seed, Nitems=nitems)
        iop = identity(2)
        for ridx in range(bgidx, edidx):
            q = rand_ket(2)
            for i in range(Nenvs - 1):
                q = tensor(q, rand_ket(2))
            for j in range(Nspins - Nenvs):
                q = tensor(q, iop)

            s_prep = sprepost(q, q.dag())
            print(s_prep.type, s_prep.dims)
            
            jobs, pipels = [], []
            for pid in range(nproc):
                ts = lst[pid]
                recv_end, send_end = multiprocessing.Pipe(False)
                p = multiprocessing.Process(target=dump_eigenval_job, \
                    args=(bindir, basename, Nspins, Nenvs, ts, L, B, s_prep, pid, send_end))
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

            filename = filename.replace('.binaryfile', '_tot_{}.binaryfile'.format(ridx))
            with open(filename, 'wb') as wrs:
                pickle.dump(z, wrs)
            logger.debug(z.keys())
            end_time = time.monotonic()
            logger.info('{}: Finish write to results. Executed time {}'.format(filename, timedelta(seconds=end_time - start_time)))
            # Plot file
            if args.plot > 0:
                plot_eigenvalues(z, nc, filename)
    else:
        # Loadfile
        filename = savedir
        with open(filename, 'rb') as rrs:
            z = pickle.load(rrs)
    
        # Plot file
        if args.plot > 0:
            plot_eigenvalues(z, nc, filename)
    
    


