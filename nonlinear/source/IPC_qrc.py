#!/usr/bin/env python
"""
    Calculate information processing capacity (IPC) for higher-order quantum reservoir
    See run_hqrc_ipc.sh for an example to run the script
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
from IPC import IPCParams

def IPC_compute(qparams, ipcparams, length, ntrials, ranseed, log_filename, savedir, posfix):
    logger = get_module_logger(__name__, log_filename)
    logger.info(log_filename)

    hqrc.get_IPC(qparams, ipcparams, length, logger=logger, ranseed=ranseed, Ntrials=ntrials, \
        savedir=savedir, posfix=posfix, type_input=1, label='log2tau_{:.3f}'.format(int(np.log2(tau_delta))))
    
if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--units', type=int, default=5, help='Number of the hidden units')
    parser.add_argument('--rho', type=int, default=0, help='Flag for initializing the density matrix')
    parser.add_argument('--coupling', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1e-14)

    parser.add_argument('--length', type=int, default=3000)
    parser.add_argument('--max_delay', type=int, default=100)
    parser.add_argument('--max_deg', type=int, default=2)
    parser.add_argument('--max_num_var', type=int, default=1)
    parser.add_argument('--max_window', type=int, default=50)
    parser.add_argument('--thres', type=float, default=0.0)
    parser.add_argument('--chunk', type=int, default=1000)

    parser.add_argument('--nproc', type=int, default=50)
    parser.add_argument('--ntrials', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deg_delays', type=str, default='0,100,0,10,0,5', help='delays by degree')
    parser.add_argument('--writedelay', type=bool, default=False)
    parser.add_argument('--virtuals', type=str, default='1')


    parser.add_argument('--nondiag', type=float, default=0.42, help='Nonlinear term (non-diagonal term)')
    parser.add_argument('--dynamic', type=str, default='ion_trap', help='full_random,full_const_trans,full_const_coeff,ion_trap')
    parser.add_argument('--basename', type=str, default='qrc_IPC')
    parser.add_argument('--savedir', type=str, default='res3_IPC')
    parser.add_argument('--tbg', type=float, default=-7.0)
    parser.add_argument('--ted', type=float, default=7.1)
    parser.add_argument('--interval', type=float, default=0.2)
    
    args = parser.parse_args()
    print(args)

    n_units, max_energy, beta, g, init_rho = args.units, args.coupling, args.beta, args.nondiag, args.rho
    max_delay, max_deg, max_num_var, thres, chunk = args.max_delay, args.max_deg, args.max_num_var, args.thres, args.chunk
    max_window, writedelay = args.max_window, args.writedelay
    deg_delays = [int(x) for x in args.deg_delays.split(',')]


    dynamic = args.dynamic
    length, ranseed = args.length, args.seed
    

    ntrials, basename, savedir = args.ntrials, args.basename, args.savedir
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    if os.path.isdir(savedir) == True:
        logdir = os.path.join(savedir, 'log')
    else:
        logdir = os.path.join(os.path.dirname(__file__), 'log')
    if os.path.isdir(logdir) == False:
        os.mkdir(logdir)
    
    basename = '{}_{}_seed_{}_mdeg_{}_mvar_{}_thres_{}_delays_{}_T_{}'.format(\
        basename, dynamic, ranseed, max_deg, max_num_var, thres, args.deg_delays, length)
    log_filename = os.path.join(logdir, '{}.log'.format(basename))
    logger = get_module_logger(__name__, log_filename)
    logger.info(log_filename)

    # Make IPCparams
    ipcparams = IPCParams(max_delay=max_delay, max_deg=max_deg, max_num_var=max_num_var, \
        max_window=max_window, thres=thres, deg_delays=deg_delays)
    
    #tx = list(np.arange(-7, 7.1, 0.5))
    tx = list(np.arange(args.tbg, args.ted, args.interval))
    #taudeltas = [2**x for x in tx]
    virtuals = [int(x) for x in args.virtuals.split(',')]

    if os.path.isfile(savedir) == False:
        jobs, pipels = [], []
        for V in virtuals:
            #for tau_delta in taudeltas:
            for x in tx:
                tau_delta = 2**x
                log_filename = os.path.join(logdir, 'logtau_{:.3f}_V_{}_{}.log'.format(x, V, basename))
                posfix = 'capa_logtau_{:.3f}_V_{}_{}'.format(x, V, basename)
            
                # check file
                # degfile = os.path.join(savedir, 'degree_{}.txt'.format(posfix))
                # if os.path.isfile(degfile) == True:
                #     continue
                qparams = QRCParams(n_units=n_units, max_energy=max_energy, non_diag=g,\
                            beta=beta, virtual_nodes=V, tau=tau_delta, init_rho=init_rho, dynamic=dynamic)
                p = multiprocessing.Process(target=IPC_compute, \
                    args=(qparams, ipcparams, length, ntrials, ranseed, log_filename, savedir, posfix))
                jobs.append(p)
                #pipels.append(recv_end)

        # Start the process
        for p in jobs:
            p.start()

        # Ensure all processes have finished execution
        for p in jobs:
            p.join()

        # Sleep 5s
        time.sleep(5)