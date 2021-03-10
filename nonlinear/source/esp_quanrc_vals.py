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

def diff_states_compute(outfile, qparams, length, ranseed, state_trials, send_end):
    btime = int(time.time() * 1000.0)
    dP = qrc.esp_states(qparams, length=length, ranseed=ranseed, state_trials=state_trials)
    np.save(outfile, dP)

    etime = int(time.time() * 1000.0)
    bcoef = qparams.non_diag
    alpha = qparams.alpha
    B = qparams.max_energy / bcoef
    tauB = qparams.tau * B
    
    diff_avg, diff_std = np.mean(dP), np.std(dP)

    retstr = 'Finish job with alpha={},bcoef={},tauB={},length={},diff_avg={:.10f},diff_std={:.10f}'.format(\
        alpha, bcoef, tauB, length, diff_avg, diff_std)
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
    parser.add_argument('--length', type=int, default=1000)
    
    parser.add_argument('--amin', type=float, default=0.9, help='Minimum Alpha power in Ising model')
    parser.add_argument('--amax', type=float, default=1.0, help='Maximum Alpha power in Ising model')
    parser.add_argument('--nas', type=int, default=1, help='Number of Alpha')

    parser.add_argument('--bcmin', type=float, default=0.9, help='Minimum bcoeff nonlinear term (non-diagonal term)')
    parser.add_argument('--bcmax', type=float, default=1.0, help='Maximum bcoeff nonlinear term (non-diagonal term)')
    parser.add_argument('--nbcs', type=int, default=1, help='Number of bcoeff')

    parser.add_argument('--tmin', type=float, default=0.0, help='Minimum of tauB')
    parser.add_argument('--tmax', type=float, default=12.5, help='Maximum of tauB')
    parser.add_argument('--ntaus', type=int, default=125, help='Number of tausB')

    parser.add_argument('--virtuals', type=str, default='1')
    parser.add_argument('--als', type=str, default='')
    parser.add_argument('--bcls', type=str, default='')
    parser.add_argument('--tauls', type=str, default='')

    parser.add_argument('--basename', type=str, default='esp') 
    parser.add_argument('--savedir', type=str, default='esp_states')
    parser.add_argument('--ntrials', type=int, default=2, help='Number of trials')
    args = parser.parse_args()
    print(args)

    n_spins, n_envs, beta = args.spins, args.envs, args.beta
    init_rho, length, dynamic, max_energy = args.rho, args.length, args.dynamic, args.max_energy
    
    amin, amax, nas = args.amin, args.amax, args.nas
    bcmin, bcmax, nbcs = args.bcmin, args.bcmax, args.nbcs
    tmin, tmax, ntaus, ntrials = args.tmin, args.tmax, args.ntaus, args.ntrials
    
    bname, savedir = args.basename, args.savedir
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

    basename = '{}_{}_nspins_{}_{}_Vs_{}_len_{}_ntrials_{}'.format(\
        bname, dynamic, n_spins, n_envs, '_'.join([str(v) for v in virtuals]), length, ntrials)

    log_filename = os.path.join(logdir, '{}.log'.format(basename))
    logger = get_module_logger(__name__, log_filename)
    logger.info(log_filename)
    
    jobs, pipels = [], []
    if os.path.isfile(savedir) == False:
        for alpha in als:
            for bcoef in bcls:
                for tauB in tauls:
                    for V in virtuals:
                        outfile = '{}_{}_nspins_{}_{}_a_{:.3f}_bc_{:.3f}_tauB_{:.3f}_V_{}_len_{}_ntrials_{}.npy'.format(\
                            bname, dynamic, n_spins, n_envs, alpha, bcoef, tauB, V, length, ntrials)
                        outfile = os.path.join(bindir, outfile)
                        if os.path.isfile(outfile) == True:
                            print('File existed {}'.format(outfile))
                            continue
                        B = max_energy / bcoef
                        qparams = QRCParams(n_units=n_spins-n_envs, n_envs=n_envs, max_energy=max_energy, non_diag=bcoef, alpha=alpha,\
                            beta=beta, virtual_nodes=V, tau=tauB/B, init_rho=init_rho, dynamic=dynamic)
                        recv_end, send_end = multiprocessing.Pipe(False)
                        p = multiprocessing.Process(target=diff_states_compute, \
                            args=(outfile, qparams, length, 0, ntrials, send_end))
                        jobs.append(p)
                        pipels.append(recv_end)
                        
    # Start the process
    for p in jobs:
        p.start()

    # Ensure all processes have finished execution
    for p in jobs:
        p.join()

    # Sleep 5s
    time.sleep(5)

    # Get the result
    for recv_end in pipels:
        retstr = recv_end.recv()
        logger.debug(retstr)
