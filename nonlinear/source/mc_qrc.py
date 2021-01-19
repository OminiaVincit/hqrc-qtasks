#!/usr/bin/env python
"""
    Calculate memory capacity for higher-order quantum reservoir
    See run_MC_repeated.sh for an example to run the script
"""

import sys
import numpy as np
import os
import scipy
import argparse
import multiprocessing
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import time
import datetime
import hqrc as hqrc
from loginit import get_module_logger
import utils
from utils import *

def memory_compute(taskname, qparams, nqrc, layer_strength,\
        train_len, val_len, buffer, dlist, ranseed, pid, send_end):
    btime = int(time.time() * 1000.0)
    rsarr = hqrc.memory_function(taskname, qparams, train_len=train_len, val_len=val_len, buffer=buffer, \
        dlist=dlist, nqrc=nqrc, gamma=layer_strength, sparsity=1.0, sigma_input=1.0, ranseed=ranseed)
    C = np.sum(rsarr[:, 1])
    etime = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))
    print('{} Finished process {} in {} s with J={}, taudelta={}, V={}, layers={}, strength={}, dmin={}, dmax={}, capacity={}'.format(\
        datestr, pid, etime-btime, \
        qparams.max_energy, qparams.tau, qparams.virtual_nodes, nqrc, layer_strength, dlist[0], dlist[-1], C))
    send_end.send('{}'.format(C))

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--spins', type=int, default=5, help='Number of the hidden units')
    parser.add_argument('--coupling', type=float, default=1.0)

    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha of coupled strength, 0 for random coupling')
    parser.add_argument('--bcoef', type=float, default=0.42, help='bcoeff nonlinear term (non-diagonal term)')
    parser.add_argument('--dynamic', type=str, default='ion_trap')

    parser.add_argument('--rho', type=int, default=0, help='Flag for initializing the density matrix')
    parser.add_argument('--beta', type=float, default=1e-14, help='reg term')
    parser.add_argument('--solver', type=str, default=LINEAR_PINV, \
        help='regression solver by linear_pinv,ridge_pinv,auto,svd,cholesky,lsqr,sparse_cg,sag')

    parser.add_argument('--trainlen', type=int, default=3000)
    parser.add_argument('--vallen', type=int, default=1000)
    parser.add_argument('--buffer', type=int, default=1000)
    
    parser.add_argument('--mind', type=int, default=0)
    parser.add_argument('--maxd', type=int, default=10)
    parser.add_argument('--interval', type=int, default=1)

    parser.add_argument('--nproc', type=int, default=50)
    parser.add_argument('--ntrials', type=int, default=1)

    parser.add_argument('--tmin', type=float, default=0.0, help='Minimum of tauB')
    parser.add_argument('--tmax', type=float, default=25.0, help='Maximum of tauB')
    parser.add_argument('--ntaus', type=int, default=250, help='Number of tausB')

    parser.add_argument('--layers', type=str, default='1')
    parser.add_argument('--strengths', type=str, default='0.0')
    parser.add_argument('--virtuals', type=str, default='1')

    parser.add_argument('--taskname', type=str, default='qrc_stm') # Use _stm or _pc
    parser.add_argument('--savedir', type=str, default='capa_repeated')
    args = parser.parse_args()
    print(args)

    n_spins, beta = args.spins, args.beta
    train_len, val_len, buffer = args.trainlen, args.vallen, args.buffer
    nproc, init_rho, solver, dynamic = args.nproc, args.rho, args.solver, args.dynamic
    max_energy, alpha, bcoef = args.coupling, args.alpha, args.bcoef
    tmin, tmax, ntaus = args.tmin, args.tmax, args.ntaus

    minD, maxD, interval, Ntrials = args.mind, args.maxd, args.interval, args.ntrials
    dlist = list(range(minD, maxD + 1, interval))
    nproc = min(nproc, len(dlist))
    print('Divided into {} processes'.format(nproc))
    
    taskname, savedir, solver = args.taskname, args.savedir, args.solver
    if os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    B = max_energy / bcoef
    taudeltas = list(np.linspace(tmin, tmax, ntaus + 1) / B)
    taudeltas = taudeltas[1:]

    virtuals = [int(x) for x in args.virtuals.split(',')]
    layers = [int(x) for x in args.layers.split(',')]
    strengths = [float(x) for x in args.strengths.split(',')]

    # Evaluation
    timestamp = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    # datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))

    outbase = os.path.join(savedir, '{}_{}_nspins_{}_a_{}_bc_{}_tmax_{}_tmin_{}_ntaus_{}_J_{}_strength_{}_V_{}_layers_{}_len_{}_{}_{}_capa_trials_{}'.format(\
        taskname, dynamic, n_spins, alpha, bcoef, tmax, tmin, ntaus, max_energy,\
            '_'.join([str(s) for s in strengths]), \
            '_'.join([str(v) for v in virtuals]), \
            '_'.join([str(l) for l in layers]), \
            buffer, train_len, val_len, Ntrials))

    log_filename = '{}.log'.format(outbase)
    logger = get_module_logger(__name__, log_filename)
    logger.info(log_filename)

    global_rs = []
    if os.path.isfile(savedir) == False:
        for tau in taudeltas:
            for V in virtuals:
                qparams = QRCParams(n_units=n_spins - 1, n_envs=1, max_energy=max_energy, non_diag=bcoef, alpha=alpha,\
                    beta=beta, virtual_nodes=V, tau=tau, init_rho=init_rho, dynamic=dynamic, solver=solver)
                for layer_strength in strengths:
                    for nqrc in layers:
                        local_sum = []
                        for n in range(Ntrials):
                            # Multi process
                            lst = np.array_split(dlist, nproc)
                            jobs, pipels = [], []
                            for proc_id in range(nproc):
                                dsmall = lst[proc_id]
                                if dsmall.size == 0:
                                    continue
                                print('dlist: ', dsmall)
                                recv_end, send_end = multiprocessing.Pipe(False)
                                p = multiprocessing.Process(target=memory_compute, \
                                    args=(taskname, qparams, nqrc, layer_strength, train_len, val_len, buffer, dsmall, n, proc_id, send_end))
                                jobs.append(p)
                                pipels.append(recv_end)
                    
                            # Start the process
                            for p in jobs:
                                p.start()
                    
                            # Ensure all processes have finiished execution
                            for p in jobs:
                                p.join()

                            # Sleep 5s
                            time.sleep(5)

                            # Get the result
                            local_rsarr = [float(x.recv()) for x in pipels]
                            local_sum.append(np.sum(local_rsarr))
                        local_avg, local_std = np.mean(local_sum), np.std(local_sum)
                        global_rs.append([nqrc, layer_strength, V, tau, max_energy, local_avg, local_std])
                        logger.debug('J={},bcoef={},tauB={},V={},layer_strength={},layers={},capa_avg={},capa_std={}'.format(\
                            max_energy, bcoef, tau * B, V, layer_strength, nqrc, local_avg, local_std))
    
    rsarr = np.array(global_rs)
    np.savetxt('{}_capacity.txt'.format(outbase), rsarr, delimiter=' ')
    
    # save experiments setting
    with open('{}_setting.txt'.format(outbase), 'w') as sfile:
        sfile.write('solver={}, train_len={}, val_len={}, buffer={}\n'.format(\
            solver, train_len, val_len, buffer))
        sfile.write('beta={}, Ntrials={}\n'.format(beta, Ntrials))
        sfile.write('n_spins={}\n'.format(n_spins))
        sfile.write('max_energy={},bcoef={},alpha={}\n'.format(max_energy, bcoef, alpha))
        sfile.write('tauBs={}\n'.format(' '.join([str(v*B) for v in taudeltas])))
        sfile.write('layers={}\n'.format(' '.join([str(l) for l in layers])))
        sfile.write('V={}\n'.format(' '.join([str(v) for v in virtuals])))
        sfile.write('minD={}, maxD={}, interval={}\n'.format(minD, maxD, interval))
        sfile.write('layer_strength={}\n'.format(' '.join([str(v) for v in strengths])))