#!/usr/bin/env python
"""
    Do tomography for temporal quantum data
    See run_QTask.sh for an example to run the script
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
from utils import *
from qutils import *
from qutip import *
from IPC import IPCParams
import colorcet as cc

def generate_qtasks_delay(n_envs, ranseed, length, delay, taskname, order):
    #input_data = generate_one_qubit_states(ranseed=ranseed, Nitems=length)
    np.random.seed(seed=ranseed + 1)
    D = 2**n_envs
    # Returns a superoperator acting on vectorized dim × dim density operators, 
    # sampled from the BCSZ distribution.
    sup_ops = []
    
    for d in range(delay+1):
        sop = rand_super_bcsz(N=D, enforce_tp=True)
        #sop = rand_unitary(N=2**n_envs)
        sup_ops.append(sop)
    
    # generate coefficients
    coeffs = np.random.rand(delay + 1) 
    #coeffs[-1] = 1.0
    if 'delay' in taskname:
        coeffs = np.zeros(delay+1)
        coeffs[-1] = 1.0
    elif 'wma' in taskname: # weighted moving average filter
        coeffs = np.flip(np.arange(1, delay+2))
    elif 'sma' in taskname:
        coeffs = np.ones(delay+1)
    #print('Coeffs', coeffs)
    coeffs = coeffs / np.sum(coeffs)
    coeffs = coeffs.astype(complex)

    if n_envs == 1:
        input_data = generate_one_qubit_states(ranseed=ranseed, Nitems=length)
    else:
        input_data = generate_random_states(ranseed=ranseed, Nbase=n_envs, Nitems=length)
    #input_data = generate_random_states(ranseed=ranseed, Nbase=n_envs, Nitems=length)
    
    idrho = np.zeros((D, D)).astype(complex)
    idrho[0, 0] = 1
    output_data = [idrho] * length
    if taskname == 'sma-rand' or taskname == 'wma-rand':
        print('Task {}'.format(taskname))
        for n in range(delay, length):
            outstate = None
            for d in range(delay+1):
                mstate = Qobj(input_data[n - d])
                mstate = operator_to_vector(mstate)
                mstate = sup_ops[d] * mstate
                if outstate is not None:
                    outstate += coeffs[d] * mstate
                else:
                    outstate = coeffs[d] * mstate
            #print('Shape outstate', is_trace_one(outstate), is_hermitian(outstate), is_positive_semi(outstate))
            output_data[n] = np.array(vector_to_operator(outstate))
    elif taskname == 'sma-depolar' or taskname == 'wma-depolar' or taskname == 'delay-depolar':
        print('Make NARMA data for depolar task {}'.format(taskname))
        _, data_noise = make_data_for_narma(length=length, orders=[order])
        pnoise = data_noise[:, 0].ravel()
        idmat = np.eye(D)
        for n in range(delay, length):
            outstate = None
            for d in range(delay+1):
                mstate = pnoise[n-d] * idmat / D + (1.0 - pnoise[n-d]) * input_data[n - d]
                if outstate is not None:
                    outstate += coeffs[d] * mstate
                else:
                    outstate = coeffs[d] * mstate
                output_data[n] = outstate
    elif taskname == 'delay-id':
        print('Task {}'.format(taskname))
        output_data[delay:] = input_data[:(length-delay)]
    elif taskname == 'delay-rand':
        for n in range(delay, length):
            mstate = Qobj(input_data[n - delay])
            mstate = operator_to_vector(mstate)
            mstate = sup_ops[delay] * mstate
            output_data[n] = np.array(vector_to_operator(mstate))    
    else:
        output_data = input_data
    
    return input_data, output_data

def convert_seq(input_seq):
    ma = np.real(input_seq).reshape((input_seq.shape[0], -1))
    mb = np.imag(input_seq).reshape((input_seq.shape[0], -1))
    return np.hstack((ma, mb)).transpose()

def plot_result(fig_path, res_title, train_input_seq, train_output_seq, val_input_seq, val_output_seq, \
    train_pred_seq, val_pred_seq, train_fidls, val_fidls):
    input_seq = np.vstack((train_input_seq, val_input_seq))
    output_seq = np.vstack((train_output_seq, val_output_seq))
    pred_seq = np.vstack((train_pred_seq, val_pred_seq))
    #print('shape1', input_seq.shape, output_seq.shape)
    input_seq = convert_seq(input_seq)
    output_seq = convert_seq(output_seq)
    pred_seq = convert_seq(pred_seq)
    err_seq = np.abs(output_seq - pred_seq)
    #print('shape2', input_seq.shape, output_seq.shape)

    fidls = np.array([train_fidls, val_fidls]).ravel()
    #matplotlib.style.use('seaborn')
    cmap = plt.get_cmap("RdBu")
    #cmap = plt.get_cmap("tab20c_r")
    
    ecmap = plt.get_cmap("summer_r")
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=10
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #print('\n'.join(color for color in colors))  
    fig, axs = plt.subplots(4, 1, figsize=(20, 16), sharey=True)
    fig.subplots_adjust(hspace=0.4, wspace = 0.4)
    # Plotting the contour plot
    fsize = 14
    vmin, vmax = input_seq.min(), input_seq.max()
    vmin_error, vmax_error = err_seq.min(), err_seq.max()
    vmin = min(vmin, output_seq.min())
    vmin = min(vmin, pred_seq.min())
    vmax = max(vmax, output_seq.max())
    vmax = max(vmax, pred_seq.max())
    
    #print(vmin, vmax)
    axs = axs.ravel()
    for ax in axs:
        ax.set_ylabel('Index', fontsize=fsize)
        ax.set_xlabel('Time', fontsize=fsize)
        
    mp0 = plotContour(fig, axs[0], input_seq, "Input", fsize, vmin, vmax, cmap)
    mp1 = plotContour(fig, axs[1], output_seq, "Output", fsize, vmin, vmax, cmap)
    mp2 = plotContour(fig, axs[2], pred_seq, "Target", fsize, vmin, vmax, cmap)
    mp3 = plotContour(fig, axs[3], err_seq, "Diff. and {}".format(res_title), fsize, vmin_error, vmax_error, ecmap)
    bx = axs[3].twinx()
    nicered = (0.769, 0.306, 0.322)
    bx.plot(fidls, 'o', linestyle ='-', marker='o', color=nicered, alpha=0.8)
    bx.set_ylabel('Fidelity', color=nicered, fontsize=fsize)
    #bx.set_ylim([0.8, 1.01])

    for ftype in ['png', 'svg']:
        transparent = True
        if ftype == 'png':
            transparent = False
        plt.savefig('{}.{}'.format(fig_path, ftype), bbox_inches='tight', transparent=transparent, dpi=600)
    plt.show()

def fidelity_compute(qparams, train_len, val_len, buffer, ntrials, log_filename, B, tBs, delay, taskname, order, flagplot):
    logger = get_module_logger(__name__, log_filename)
    logger.info(log_filename)
    length = buffer + train_len + val_len
    basename = log_filename.replace('.log', '')
    for tauB in tBs:
        qparams.tau = tauB / B
        train_rmean_ls, val_rmean_ls = [], []
        for n in range(ntrials):
            input_data, output_data = generate_qtasks_delay(qparams.n_envs, ranseed=n, length=length, delay=delay, taskname=taskname, order=order)
            train_input_seq = np.array(input_data[  : (buffer + train_len)])
            train_output_seq = np.array(output_data[  : (buffer + train_len)])
            
            val_input_seq   = np.array(input_data[(buffer + train_len) : length])
            val_output_seq = np.array(output_data[(buffer + train_len) : length])
            fig_path = '{}_tauB_{:.3f}_{}'.format(basename, tauB, n)

            train_pred_seq, train_fidls, val_pred_seq, val_fidls = \
                qrc.get_fidelity(qparams, buffer, train_input_seq, train_output_seq, val_input_seq, val_output_seq, ranseed=n)
            
            #train_fid_avg, train_fid_std = np.mean(train_fidls), np.std(train_fidls)
            #val_fid_avg, val_fid_std = np.mean(val_fidls), np.std(val_fidls)
            train_rmean_square_fid = np.sqrt(np.mean(np.array(train_fidls)**2))
            val_rmean_square_fid = np.sqrt(np.mean(np.array(val_fidls)**2))
            res_title = 'Root mean square Fidelity at n={}, tauB={:.3f}, train={:.6f}, val={:.6f}'.format(n, tauB, train_rmean_square_fid, val_rmean_square_fid)
            logger.debug(res_title)
            
            train_rmean_ls.append(train_rmean_square_fid)
            val_rmean_ls.append(val_rmean_square_fid)
            if flagplot > 0 and n == 0:
                plot_result(fig_path, res_title, train_input_seq[buffer:], train_output_seq[buffer:], \
                    val_input_seq, val_output_seq, train_pred_seq, val_pred_seq,\
                    train_fidls, val_fidls)
        
        avg_train, avg_val = np.mean(train_rmean_ls), np.mean(val_rmean_ls)
        std_train, std_val = np.std(train_rmean_ls), np.std(val_rmean_ls)
        logger.info('Average RMSF with ntrials={}, tauB={:.3f}, avg-train={:.6f}, avg-val={:.6f}, std-train={:.6f}, std-val={:.6f}'.format(\
            ntrials, tauB, avg_train, avg_val, std_train, std_val))
            
    
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

    parser.add_argument('--virtuals', type=str, default='1')

    parser.add_argument('--trainlen', type=int, default=200)
    parser.add_argument('--vallen', type=int, default=100)
    parser.add_argument('--buffer', type=int, default=100)
    parser.add_argument('--delay', type=int, default=5)

    parser.add_argument('--nproc', type=int, default=125)
    parser.add_argument('--ntrials', type=int, default=1)

    parser.add_argument('--basename', type=str, default='qtasks')
    parser.add_argument('--taskname', type=str, default='delay')
    parser.add_argument('--savedir', type=str, default='QTasks_repeated')
    parser.add_argument('--order', type=int, default=5, help='Order of nonlinear in depolarizing channel')

    parser.add_argument('--tmax', type=float, default=25, help='Maximum of tauB')
    parser.add_argument('--tmin', type=float, default=0, help='Minimum of tauB')
    parser.add_argument('--ntaus', type=int, default=125, help='Number of tausB')
    parser.add_argument('--plot', type=int, default=0, help='Flag to plot')

    args = parser.parse_args()
    print(args)

    n_spins, n_envs, max_energy, beta, alpha, bcoef, init_rho = args.spins, args.envs, args.max_energy, args.beta, args.alpha, args.bcoef, args.rho
    tmin, tmax, ntaus = args.tmin, args.tmax, args.ntaus
    flagplot = args.plot

    dynamic = args.dynamic
    train_len, val_len, buffer, delay = args.trainlen, args.vallen, args.buffer, args.delay
    ntrials, basename, savedir, taskname, order = args.ntrials, args.basename, args.savedir, args.taskname, args.order
    virtuals = [int(x) for x in args.virtuals.split(',')]

    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    if os.path.isdir(savedir) == True:
        logdir = os.path.join(savedir, 'log')
    else:
        logdir = os.path.join(os.path.dirname(__file__), 'log')
    if os.path.isdir(logdir) == False:
        os.mkdir(logdir)
    
    basename = '{}_{}_od_{}_{}_nspins_{}_{}_a_{}_bc_{}_tmax_{}_tmin_{}_ntaus_{}_Vs_{}_len_{}_{}_{}_d_{}_trials_{}'.format(\
        basename, taskname, order, dynamic, n_spins, n_envs, alpha, bcoef, tmax, tmin, ntaus, \
        '_'.join([str(v) for v in virtuals]), buffer, train_len, val_len, delay, ntrials)
    
    log_filename = os.path.join(logdir, '{}.log'.format(basename))
    logger = get_module_logger(__name__, log_filename)
    logger.info(log_filename)

    B = max_energy / bcoef

    txBs = list(np.linspace(tmin, tmax, ntaus + 1))
    txBs = txBs[1:]
    nproc = min(len(txBs), args.nproc)
    lst = np.array_split(txBs, nproc)

    if os.path.isfile(savedir) == False:
        for V in virtuals:
            jobs, pipels = [], []
            for pid in range(nproc):
                tBs = lst[pid]
                posfix = 'V_{}_{}'.format(V, basename)
                log_filename = os.path.join(logdir, 'tauB_{:.3f}_{:.3f}_V_{}_{}.log'.format(tBs[0], tBs[-1], V, basename))
                
                # check file
                # degfile = os.path.join(savedir, 'degree_{}.txt'.format(posfix))
                # if os.path.isfile(degfile) == True:
                #     continue
                qparams = QRCParams(n_units=n_spins-n_envs, n_envs=n_envs, max_energy=max_energy, non_diag=bcoef,alpha=alpha,\
                            beta=beta, virtual_nodes=V, tau=0.0, init_rho=init_rho, dynamic=dynamic)
                p = multiprocessing.Process(target=fidelity_compute, \
                    args=(qparams, train_len, val_len, buffer, ntrials, log_filename, B, tBs, delay, taskname, order, flagplot))
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
