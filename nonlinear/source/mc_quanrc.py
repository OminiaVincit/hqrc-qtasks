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

def memory_compute(qparams, train_len, val_len, buffer, dlist, ranseed, pid, send_end):
    btime = int(time.time() * 1000.0)
    rsarr = qrc.memory_function(qparams, train_len=train_len, val_len=val_len, buffer=buffer, dlist=dlist, ranseed=ranseed, Ntrials=1)
    C = np.sum(rsarr[:, 1])
    val_fid = np.mean(rsarr[:, -1])
    train_fid = np.mean(rsarr[:, -2])
    
    etime = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))
    bcoef = qparams.non_diag
    alpha = qparams.alpha
    B = qparams.max_energy / bcoef
    tauB = qparams.tau * B

    # print('{} Finished process {} in {} s with tauB={}, bcoef={}, alpha={}, V={}, dmin={}, dmax={}, Fid train={} val={}, capacity={}'.format(\
    #    datestr, pid, etime-btime, \
    #    tauB, bcoef, alpha, qparams.virtual_nodes, dlist[0], dlist[-1], train_fid, val_fid, C))
    
    # Send list of MFd
    MFds = rsarr[:, 1].ravel()
    MFd_str = ','.join([str(x) for x in MFds])
    send_end.send('{}'.format(MFd_str))

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--spins', type=int, default=5, help='Number of the spins in the total system')
    parser.add_argument('--envs', type=int, default=1, help='Number of the spins in the environmental system')
    parser.add_argument('--rho', type=int, default=0, help='Flag for initializing the density matrix')
    parser.add_argument('--max_energy', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1e-14)

    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha of coupled strength, 0 for random coupling')
    parser.add_argument('--bcoef', type=float, default=1.0, help='bcoeff nonlinear term (non-diagonal term)')
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
    parser.add_argument('--ntrials', type=int, default=1)

    parser.add_argument('--tmin', type=float, default=0.0, help='Minimum of tauB')
    parser.add_argument('--tmax', type=float, default=25.0, help='Maximum of tauB')
    parser.add_argument('--ntaus', type=int, default=25, help='Number of tausB')

    parser.add_argument('--virtuals', type=str, default='1')

    parser.add_argument('--basename', type=str, default='quanrc') 
    parser.add_argument('--savedir', type=str, default='capa_repeated')
    args = parser.parse_args()
    print(args)

    n_spins, n_envs, beta = args.spins, args.envs, args.beta
    train_len, val_len, buffer = args.trainlen, args.vallen, args.buffer
    nproc, init_rho, solver, dynamic = args.nproc, args.rho, args.solver, args.dynamic
    max_energy, alpha, bcoef = args.max_energy, args.alpha, args.bcoef
    tmin, tmax, ntaus = args.tmin, args.tmax, args.ntaus

    minD, maxD, interval, ntrials = args.mind, args.maxd, args.interval, args.ntrials
    dlist = list(range(minD, maxD + 1, interval))
    nproc = min(nproc, len(dlist))
    print('Divided into {} processes'.format(nproc))
    
    bname, savedir, solver = args.basename, args.savedir, args.solver
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    if os.path.isdir(savedir) == True:
        logdir = os.path.join(savedir, 'log')
    else:
        logdir = os.path.join(os.path.dirname(__file__), 'log')
    if os.path.isdir(logdir) == False:
        os.mkdir(logdir)

    B = max_energy / bcoef
    taudeltas = list(np.linspace(tmin, tmax, ntaus + 1))
    taudeltas = taudeltas[1:]

    virtuals = [int(x) for x in args.virtuals.split(',')]

    # Evaluation
    timestamp = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    # datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))

    basename = '{}_{}_nspins_{}_{}_a_{}_bc_{}_tmax_{}_tmin_{}_ntaus_{}_Vs_{}_len_{}_{}_{}_trials_{}'.format(\
        bname, dynamic, n_spins, n_envs, alpha, bcoef, tmax, tmin, ntaus,\
        '_'.join([str(v) for v in virtuals]), buffer, train_len, val_len, ntrials)

    log_filename = os.path.join(logdir, '{}.log'.format(basename))
    logger = get_module_logger(__name__, log_filename)
    logger.info(log_filename)

    global_rs = []
    if os.path.isfile(savedir) == False:
        for tauB in taudeltas:
            for V in virtuals:
                outfile = '{}_{}_nspins_{}_{}_a_{}_bc_{}_tauB_{:.3f}_V_{}_len_{}_{}_{}_trials_{}'.format(\
                    bname, dynamic, n_spins, n_envs, alpha, bcoef, \
                    tauB, V, buffer, train_len, val_len, ntrials)
                outfile = os.path.join(savedir, outfile)
                qparams = QRCParams(n_units=n_spins-n_envs, n_envs=n_envs, max_energy=max_energy, non_diag=bcoef, alpha=alpha,\
                    beta=beta, virtual_nodes=V, tau=tauB/B, init_rho=init_rho, dynamic=dynamic, solver=solver)
                
                local_sum, MFd_arr = [], []
                capacity = []
                for n in range(ntrials):
                    # Multi process
                    lst = np.array_split(dlist, nproc)
                    jobs, pipels = [], []
                    for proc_id in range(nproc):
                        dsmall = lst[proc_id]
                        if dsmall.size == 0:
                            continue
                        #print(tauB, V, n, 'dlist: ', dsmall)
                        recv_end, send_end = multiprocessing.Pipe(False)
                        p = multiprocessing.Process(target=memory_compute, \
                            args=(qparams, train_len, val_len, buffer, dsmall, n, proc_id, send_end))
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
                    dlist, MFd_ls = [], []
                    for proc_id in range(nproc):
                        dsmall = lst[proc_id]
                        if dsmall.size == 0:
                            continue
                        MFd_str = pipels[proc_id].recv()
                        MFds = [float(x) for x in MFd_str.split(',')]

                        MFd_ls.extend(MFds)
                        dlist.extend(dsmall)
                    MFd_n = np.vstack((np.array(dlist), np.array(MFd_ls))).T
                    MFd_arr.append(MFd_n)
                    local_capa = np.sum(MFd_n[:, 1])
                    capacity.append(local_capa)
                    logger.debug('Trial={}, capa={}'.format(n, local_capa))
                    #print('Mfd_n', MFd_n.shape)

                MFd_arr = np.array(MFd_arr)
                #print('MFd_arr', MFd_arr.shape)
                MFd_avg = np.mean(MFd_arr, axis=0)
                #print('MFd_arr', MFd_avg.shape)

                capa_avg = np.mean(capacity)
                std_avg = np.std(capacity)

                logger.debug('alpha={},bcoef={},tauB={},V={},capa_avg={},capa_std={}'.format(\
                    alpha, bcoef, tauB, V, capa_avg, std_avg))
    
                np.savetxt(outfile, MFd_avg, delimiter=' ')