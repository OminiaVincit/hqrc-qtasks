#!/usr/bin/env python
"""
    Do tomography for temporal quantum data
    See run_QTask.sh for an example to run the script
"""

import sys
import numpy as np
import os
import argparse
import multiprocessing
import matplotlib.pyplot as plt
from matplotlib import ticker
import time
import quanrc as qrc
from loginit import get_module_logger
from utils import *
from qutils import *

def fidelity_compute(qparams, train_len, val_len, buffer, ntrials, log_filename, \
    delay1, delay2, use_corr, reservoir, postprocess, test_lastrho, dat_label):
    logger = get_module_logger(__name__, log_filename)
    logger.info(log_filename)
    length = buffer + train_len + val_len
    train_rmean_ls, val_rmean_ls = [], []
    neg_train_rmean_ls, neg_val_rmean_ls = [], []

    for n in range(ntrials):
        input_data, output_data = generate_delay_tensor(qparams.n_envs, delay1, delay2, length=length, ranseed=n, dat_label=dat_label)
        Uop = get_transverse_unitary(Nspins=2*qparams.n_envs, B=2, dt=10.0)
        for k in range(len(output_data)):
            output_data[k] = Uop @ output_data[k] @ Uop.T.conj()
                    
        train_input_seq = np.array(input_data[  : (buffer + train_len)])
        train_output_seq = np.array(output_data[  : (buffer + train_len)])

        val_input_seq   = np.array(input_data[(buffer + train_len) : length])
        val_output_seq = np.array(output_data[(buffer + train_len) : length])

        train_pred_seq, train_fidls, val_pred_seq, val_fidls, pred_state_list = \
            qrc.get_fidelity(qparams, buffer, train_input_seq, train_output_seq, \
                val_input_seq, val_output_seq, ranseed=n, \
                use_corr=use_corr, reservoir=reservoir, postprocess=postprocess, test_lastrho=test_lastrho)
        
        train_rmean_square_fid = np.sqrt(np.mean(np.array(train_fidls)**2))
        val_rmean_square_fid = np.sqrt(np.mean(np.array(val_fidls)**2))
        
        neg_train_pred = np.array(negativity_compute(train_pred_seq))
        neg_train_out  = np.array(negativity_compute(train_output_seq[buffer:]))
        train_rmse_neg = np.sqrt(np.mean(np.array(neg_train_pred - neg_train_out)**2))

        neg_val_pred = np.array(negativity_compute(val_pred_seq))
        neg_val_out  = np.array(negativity_compute(val_output_seq))
        val_rmse_neg = np.sqrt(np.mean(np.array(neg_val_pred - neg_val_out)**2))

        res_title = 'Root mean square at n={}, delay1={}, delay2={}; fid_train={:.6f}, fid_val={:.6f}; neg_train={:.6f}, neg_val={:.6f}'.format(\
            n, delay1, delay2, train_rmean_square_fid, val_rmean_square_fid, train_rmse_neg, val_rmse_neg)
        logger.debug(res_title)
        
        train_rmean_ls.append(train_rmean_square_fid)
        val_rmean_ls.append(val_rmean_square_fid)

        neg_train_rmean_ls.append(train_rmse_neg)
        neg_val_rmean_ls.append(val_rmse_neg)
                    
    avg_train, avg_val = np.mean(train_rmean_ls), np.mean(val_rmean_ls)
    std_train, std_val = np.std(train_rmean_ls), np.std(val_rmean_ls)
    neg_avg_train, neg_avg_val = np.mean(neg_train_rmean_ls), np.mean(neg_val_rmean_ls)
    neg_std_train, neg_std_val = np.std(neg_train_rmean_ls), np.std(neg_val_rmean_ls)

    logger.info('Average RMSF with ntrials={}, delay1={}, delay2={}, fid-avg-train={:.6f}, fid-avg-val={:.6f}, fid-std-train={:.6f}, fid-std-val={:.6f}'.format(\
        ntrials, delay1, delay2, avg_train, avg_val, std_train, std_val))
    logger.info('Average RMS negativity with ntrials={}, delay1={}, delay2={}, neg-avg-train={:.6f}, neg-avg-val={:.6f}, neg-std-train={:.6f}, neg-std-val={:.6f}'.format(\
        ntrials, delay1, delay2, neg_avg_train, neg_avg_val, neg_std_train, neg_std_val))

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--spins', type=int, default=5, help='Number of the spins in the total system')
    parser.add_argument('--envs', type=int, default=1, help='Number of the spins in the environmental system')
    parser.add_argument('--rho', type=int, default=0, help='Flag for initializing the density matrix')
    parser.add_argument('--max_energy', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1e-14)

    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha of coupled strength, 0 for random coupling')
    parser.add_argument('--bcoef', type=float, default=0.42, help='bcoeff nonlinear term (non-diagonal term)')
    parser.add_argument('--dynamic', type=str, default='ion_trap')
    parser.add_argument('--solver', type=str, default=RIDGE_PINV, \
        help='regression solver by linear_pinv,ridge_pinv,auto,svd,cholesky,lsqr,sparse_cg,sag')

    parser.add_argument('--virtuals', type=int, default=1)

    parser.add_argument('--trainlen', type=int, default=200)
    parser.add_argument('--vallen', type=int, default=100)
    parser.add_argument('--buffer', type=int, default=100)
    parser.add_argument('--delay', type=int, default=5)

    parser.add_argument('--nproc', type=int, default=125)
    parser.add_argument('--ntrials', type=int, default=1)

    parser.add_argument('--savedir', type=str, default='QTasks_repeated')

    parser.add_argument('--tauB', type=float, default=10.0, help='tauB')

    parser.add_argument('--usecorr', type=int, default=0, help='Use correlator operators')
    parser.add_argument('--reservoir', type=int, default=1, help='Use quantum reservoir to predict')
    parser.add_argument('--postprocess', type=int, default=1, help='Use post processing')
    parser.add_argument('--lastrho', type=int, default=1, help='Use last rho in test phase')

    parser.add_argument('--data', type=str, default='rand')
    args = parser.parse_args()
    print(args)

    n_spins, n_envs, max_energy, beta, alpha, bcoef, init_rho = args.spins, args.envs, args.max_energy, args.beta, args.alpha, args.bcoef, args.rho
    dynamic, tauB, usecorr = args.dynamic, args.tauB, args.usecorr
    use_reservoir, use_postprocess, test_lastrho = True, True, True
    if args.reservoir == 0:
        use_reservoir = False
    if args.postprocess == 0:
        use_postprocess = False
    if args.lastrho == 0:
        test_lastrho = False
    
    train_len, val_len, buffer, delay = args.trainlen, args.vallen, args.buffer, args.delay
    ntrials, savedir = args.ntrials, args.savedir
    V = args.virtuals

    dat_label = None
    if args.data != 'rand':
        dat_label = args.data

    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    if os.path.isdir(savedir) == True:
        logdir = os.path.join(savedir, 'log')
    else:
        logdir = os.path.join(os.path.dirname(__file__), 'log')
    if os.path.isdir(logdir) == False:
        os.mkdir(logdir)
    
    basename = 'qrc_{}_{}_post_{}_lastr_{}_{}_corr_{}_nspins_{}_{}_a_{}_bc_{}_tauB_{}_V_{}_len_{}_{}_{}_maxd_{}_trials_{}'.format(\
        args.reservoir, args.data, args.postprocess, args.lastrho,\
        dynamic, usecorr, n_spins, n_envs, alpha, bcoef, tauB, \
        V, buffer, train_len, val_len, delay, ntrials)
    
    log_filename = os.path.join(logdir, '{}.log'.format(basename))
    logger = get_module_logger(__name__, log_filename)
    logger.info(log_filename)

    B = max_energy / bcoef
    tau = tauB/B

    jobs, pipels = [], []
    for d1 in range(delay+1):
        for d2 in range(delay+1):
            log_filename = os.path.join(logdir, 'delay_{}_{}_{}.log'.format(d1, d2, basename))
            qparams = QRCParams(n_units=n_spins-n_envs, n_envs=n_envs, max_energy=max_energy, non_diag=bcoef,alpha=alpha,\
                        beta=beta, virtual_nodes=V, tau=tau, init_rho=init_rho, dynamic=dynamic)
            
            p = multiprocessing.Process(target=fidelity_compute, \
                args=(qparams, train_len, val_len, buffer, ntrials, log_filename, \
                    d1, d2, usecorr, use_reservoir, use_postprocess, test_lastrho, dat_label))
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
