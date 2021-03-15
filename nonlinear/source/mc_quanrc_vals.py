#!/usr/bin/env python
"""
    Calculate quantum memory capacity for quantum task
    of temporal quantum data
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
import quanrc as qrc
from loginit import get_module_logger
import utils
from utils import *

def memory_compute(outfile, qparams, train_len, val_len, buffer, dlist, trial, usecorr, send_end):
    btime = int(time.time() * 1000.0)
    rsarr = qrc.memory_function(qparams, train_len=train_len, val_len=val_len, buffer=buffer, dlist=dlist, ranseed=trial, Ntrials=1, usecorr=usecorr)
    np.save(outfile, rsarr)
    etime = int(time.time() * 1000.0)
    bcoef = qparams.non_diag
    alpha = qparams.alpha
    B = qparams.max_energy / bcoef
    tauB = qparams.tau * B
    capa = np.sum(rsarr[:, 1])
    val_fid = np.mean(rsarr[:, -1])
    train_fid = np.mean(rsarr[:, -2])

    retstr = 'Finish trial {} with alpha={},bcoef={},tauB={},train={},val={},capa={}'.format(\
        trial, alpha, bcoef, tauB, train_fid, val_fid, capa)
    send_end.send(retstr)

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--spins', type=int, default=5, help='Number of the spins in the total system')
    parser.add_argument('--envs', type=int, default=1, help='Number of the spins in the environmental system')
    parser.add_argument('--rho', type=int, default=0, help='Flag for initializing the density matrix')
    parser.add_argument('--max_energy', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1e-14)

    parser.add_argument('--dynamic', type=str, default='ion_trap')
    parser.add_argument('--solver', type=str, default=LINEAR_PINV, \
        help='regression solver by linear_pinv,ridge_pinv,auto,svd,cholesky,lsqr,sparse_cg,sag')

    parser.add_argument('--trainlen', type=int, default=3000)
    parser.add_argument('--vallen', type=int, default=1000)
    parser.add_argument('--buffer', type=int, default=1000)
    
    parser.add_argument('--mind', type=int, default=0)
    parser.add_argument('--maxd', type=int, default=125)
    parser.add_argument('--interval', type=int, default=1)

    parser.add_argument('--nproc', type=int, default=50)
    parser.add_argument('--bgidx', type=int, default=0)
    parser.add_argument('--edidx', type=int, default=1)

    parser.add_argument('--amin', type=float, default=0.9, help='Minimum Alpha power in Ising model')
    parser.add_argument('--amax', type=float, default=1.0, help='Maximum Alpha power in Ising model')
    parser.add_argument('--nas', type=int, default=1, help='Number of Alpha')

    parser.add_argument('--bcmin', type=float, default=0.9, help='Minimum bcoeff nonlinear term (non-diagonal term)')
    parser.add_argument('--bcmax', type=float, default=1.0, help='Maximum bcoeff nonlinear term (non-diagonal term)')
    parser.add_argument('--nbcs', type=int, default=1, help='Number of bcoeff')

    parser.add_argument('--tmin', type=float, default=0.0, help='Minimum of tauB')
    parser.add_argument('--tmax', type=float, default=25.0, help='Maximum of tauB')
    parser.add_argument('--ntaus', type=int, default=125, help='Number of tausB')

    parser.add_argument('--virtuals', type=str, default='1')
    parser.add_argument('--als', type=str, default='')
    parser.add_argument('--bcls', type=str, default='')
    parser.add_argument('--tauls', type=str, default='')

    parser.add_argument('--basename', type=str, default='quanrc') 
    parser.add_argument('--savedir', type=str, default='capa_repeated')
    parser.add_argument('--usecorr', type=int, default=0, help='Use correlator operators')

    args = parser.parse_args()
    print(args)

    n_spins, n_envs, beta = args.spins, args.envs, args.beta
    train_len, val_len, buffer = args.trainlen, args.vallen, args.buffer
    nproc, init_rho, solver, dynamic = args.nproc, args.rho, args.solver, args.dynamic
    max_energy, usecorr = args.max_energy, args.usecorr
    
    bgidx, edidx = args.bgidx, args.edidx
    amin, amax, nas = args.amin, args.amax, args.nas
    bcmin, bcmax, nbcs = args.bcmin, args.bcmax, args.nbcs
    tmin, tmax, ntaus = args.tmin, args.tmax, args.ntaus

    mind, maxd, interval = args.mind, args.maxd, args.interval
    dlist = list(range(mind, maxd + 1, interval))
    
    bname, savedir, solver = args.basename, args.savedir, args.solver
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    if os.path.isdir(savedir) == True:
        logdir = os.path.join(savedir, 'log')
        bindir = os.path.join(savedir, 'binary')
    else:
        logdir = os.path.join(os.path.dirname(__file__), 'log')
    if os.path.isdir(logdir) == False:
        os.mkdir(logdir)

    if os.path.isdir(bindir) == False:
        os.mkdir(bindir)

    virtuals = [int(x) for x in args.virtuals.split(',')]
    if args.als != '':
        als = [float(x) for x in args.als.split(',')]
    else:
        als = list(np.linspace(amin, amax, nas + 1))[1:]

    if args.bcls != '':
        bcls = [float(x) for x in args.bcls.split(',')]
    else: 
        bcls = list(np.linspace(bcmin, bcmax, nbcs + 1))[1:]
    
    if args.tauls != '':
        tauls = [float(x) for x in args.tauls.split(',')]
    else:
        tauls = list(np.linspace(tmin, tmax, ntaus + 1))[1:]
    

    # Evaluation
    timestamp = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    # datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))
    if usecorr > 0:
        corr = '_corr_{}'.format(usecorr)
    else:
        corr = ''
    
    basename = '{}_{}{}_nspins_{}_{}_Vs_{}_len_{}_{}_{}_dmax_{}'.format(\
        bname, dynamic, corr, n_spins, n_envs, '_'.join([str(v) for v in virtuals]), buffer, train_len, val_len, maxd)

    log_filename = os.path.join(logdir, '{}.log'.format(basename))
    logger = get_module_logger(__name__, log_filename)
    logger.info(log_filename)
    
    for idx in range(bgidx, edidx):
        jobs, pipels = [], []
        if os.path.isfile(savedir) == False:
            for alpha in als:
                for bcoef in bcls:
                    for tauB in tauls:
                        for V in virtuals:
                            outfile = '{}_{}{}_nspins_{}_{}_a_{:.3f}_bc_{:.3f}_tauB_{:.3f}_V_{}_len_{}_{}_{}_dmax_{}_trial_{}.npy'.format(\
                                bname, dynamic, corr, n_spins, n_envs, alpha, bcoef, tauB, V, buffer, train_len, val_len, maxd, idx)
                            outfile = os.path.join(bindir, outfile)
                            if os.path.isfile(outfile) == True:
                                print('File existed {}'.format(outfile))
                                continue
                            B = max_energy / bcoef
                            qparams = QRCParams(n_units=n_spins-n_envs, n_envs=n_envs, max_energy=max_energy, non_diag=bcoef, alpha=alpha,\
                                beta=beta, virtual_nodes=V, tau=tauB/B, init_rho=init_rho, dynamic=dynamic, solver=solver)
                            recv_end, send_end = multiprocessing.Pipe(False)
                            p = multiprocessing.Process(target=memory_compute, \
                                args=(outfile, qparams, train_len, val_len, buffer, dlist, idx, usecorr, send_end))
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
        for recv_end in pipels:
            retstr = recv_end.recv()
            logger.debug(retstr)
