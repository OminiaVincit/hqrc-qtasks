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

def minmax_norm(arr):
    minval = np.min(arr)
    maxval = np.max(arr)
    arr = arr - minval
    if maxval > minval:
        arr = arr / (maxval - minval)
    return arr

def generate_qtasks_delay(n_envs, ranseed, length, delay, taskname, \
    order, Nreps=1, buffer_train=0, dat_label=None, noise_level=0.3):
    #input_data = generate_one_qubit_states(ranseed=ranseed, Nitems=length)
    np.random.seed(seed=ranseed + 1987)
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
    Nitems = int(length / Nreps)
    if n_envs == 1:
        if dat_label != 'rand':
            input_data = generate_random_states(ranseed=ranseed, Nbase=n_envs, Nitems=length, add=dat_label)
        else:
            input_data = generate_one_qubit_states(ranseed=ranseed, Nitems=Nitems, Nreps=Nreps)
    else:
        input_data = generate_random_states(ranseed=ranseed, Nbase=n_envs, Nitems=length, add=dat_label)
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
    elif (taskname == 'sma-dephase' or taskname == 'wma-dephase' or taskname == 'delay-dephase') and n_envs == 1:
        print('Make NARMA data for Z-dephase (phase-flip) task {}'.format(taskname))
        Z = [[1,0],[0,-1]]
        _, data_noise = make_data_for_narma(length=length, orders=[order])
        pnoise = data_noise[:, 0].ravel()
        idmat = np.eye(D)
        for n in range(delay, length):
            outstate = None
            for d in range(delay+1):
                rho = input_data[n - d] @ Z
                rho = Z @ rho
                mstate = pnoise[n-d] * rho + (1.0 - pnoise[n-d]) * input_data[n - d]
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
    elif 'noise' in taskname:
        output_data[delay:] = input_data[:(length-delay)].copy()

        _, noise1 = make_data_for_narma(length=length, orders=[order])
        _, noise2 = make_data_for_narma(length=length, orders=[20])
        #print(np.min(noise1), np.min(noise2), np.max(noise1), np.max(noise2))
        pnoise1 = noise1.ravel()
        pnoise2 = noise2.ravel()
        pnoise1 = minmax_norm(pnoise1) * noise_level
        pnoise2 = minmax_norm(pnoise2) * noise_level
        # Note that 'Spin-flip probability per qubit greater than 0.5 is unphysical.'

        idmat = np.eye(D)
        Z = [[1,0],[0,-1]]
        # Add noise1 to input_data and noise2 to output data in the training part
        if buffer_train == 0:
            buffer_train = length

        for n in range(length):
            if taskname == 'denoise-depolar':
                #input_data[n] = pnoise1[n] * idmat / D + (1.0 - pnoise1[n]) * input_data[n]
                input_data[n] = depolar_channel(input_data[n], Nspins=n_envs, p=pnoise1[n])
            elif taskname == 'denoise-dephase-per':
                input_data[n] = Pauli_channel(input_data[n], Nspins=n_envs, p=pnoise1[n], Pauli='Z')
            elif taskname == 'denoise-dephase-col':
                input_data[n] = Pauli_collective(input_data[n], Nspins=n_envs, p=pnoise1[n], Pauli='Z')
            elif taskname == 'denoise-flip':
                input_data[n] = Pauli_channel(input_data[n], Nspins=n_envs, p=pnoise1[n], Pauli='X')
            elif taskname == 'denoise-unitary':
                input_data[n] = unitary_noise(input_data[n], t = pnoise1[n], num_gates=20, ranseed=ranseed+2022)
            #output_data[n] = pnoise1[n] * idmat / D + (1.0 - pnoise1[n]) * output_data[n]
        if False:
            for n in range(buffer_train):
                if taskname == 'denoise-depolar':
                    #input_data[n] = pnoise1[n] * idmat / D + (1.0 - pnoise2[n]) * input_data[n]
                    output_data[n] = depolar_channel(output_data[n], Nspins=n_envs, p=pnoise2[n])
                elif taskname == 'denoise-dephase-per':
                    output_data[n] = Pauli_channel(output_data[n], Nspins=n_envs, p=pnoise2[n], Pauli='Z')
                    #output_data[n] = unitary_noise(output_data[n], t = pnoise2[n], num_gates=20, ranseed=ranseed+2022)
                elif taskname == 'denoise-dephase-col':
                    output_data[n] = Pauli_collective(output_data[n], Nspins=n_envs, p=pnoise2[n], Pauli='Z')
                elif taskname == 'denoise-flip':
                    output_data[n] = Pauli_channel(output_data[n], Nspins=n_envs, p=pnoise2[n], Pauli='X')
                    #output_data[n] = unitary_noise(output_data[n], t = pnoise2[n], num_gates=20, ranseed=ranseed+2022)
                elif taskname == 'denoise-unitary':
                    output_data[n] = unitary_noise(output_data[n], t = pnoise2[n], num_gates=20, ranseed=ranseed+2022)
    else:
        output_data = input_data
    
    return input_data, output_data

def convert_seq(input_seq):
    ma = np.real(input_seq).reshape((input_seq.shape[0], -1))
    mb = np.imag(input_seq).reshape((input_seq.shape[0], -1))
    return np.hstack((ma, mb)).transpose()

def plot_result(fig_path, res_title, train_input_seq, train_output_seq, val_input_seq, val_output_seq, \
    train_pred_seq, val_pred_seq, train_fidls, val_fidls, pred_state_list):
    # input_seq = np.vstack((train_input_seq, val_input_seq))
    # output_seq = np.vstack((train_output_seq, val_output_seq))
    # pred_seq = np.vstack((train_pred_seq, val_pred_seq))

    input_seq = val_input_seq
    output_seq = val_output_seq
    pred_seq = val_pred_seq
    
    #print('shape1', input_seq.shape, output_seq.shape)
    input_seq = convert_seq(input_seq)
    output_seq = convert_seq(output_seq)
    pred_seq = convert_seq(pred_seq)
    err_seq = np.abs(output_seq - pred_seq)
    #print('shape2', input_seq.shape, output_seq.shape)

    # fidls = np.array([train_fidls, val_fidls]).ravel()
    fidls = np.array(val_fidls).ravel()
    #matplotlib.style.use('seaborn')
    #cmap = plt.get_cmap("RdBu")
    cmap = plt.get_cmap("twilight_shifted")
    #cmap = plt.get_cmap("tab20c_r")
    
    ecmap = plt.get_cmap("summer_r")
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=10
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #print('\n'.join(color for color in colors))  
    fig, axs = plt.subplots(5, 1, figsize=(20, 15), sharex=True)
    fig.subplots_adjust(hspace=0.4, wspace = 0.4)
    # Plotting the contour plot
    fsize = 14
    vmin, vmax = input_seq.min(), input_seq.max()
    vmin_error, vmax_error = err_seq.min(), err_seq.max()
    vmin = min(vmin, output_seq.min())
    vmin = min(vmin, pred_seq.min())
    vmax = max(vmax, output_seq.max())
    vmax = max(vmax, pred_seq.max())
    
    vmin, vmax = -0.3, 0.9
    vmin_error, vmax_error = 0.0, 0.4
    #print(vmin, vmax)
    axs = axs.ravel()
    for ax in axs:
        ax.set_ylabel('Index', fontsize=fsize)
        ax.set_xlabel('Time', fontsize=fsize)
        
    mp0 = plotContour(fig, axs[0], input_seq, "Input", fsize, vmin, vmax, cmap)
    mp1 = plotContour(fig, axs[1], output_seq, "Target", fsize, vmin, vmax, cmap)
    mp2 = plotContour(fig, axs[2], pred_seq, "Prediction", fsize, vmin, vmax, cmap)
    mp3 = plotContour(fig, axs[3], err_seq, "Diff. and {}".format(res_title), fsize, vmin_error, vmax_error, ecmap)
    bx = axs[3].twinx()
    nicered = (0.769, 0.306, 0.322)
    bx.plot(fidls, linestyle ='-', linewidth=2, marker='o', color=nicered, alpha=0.8)
    bx.set_ylabel('Fidelity', color=nicered, fontsize=fsize)
    bx.set_ylim([0.8, 1.01])

    if pred_state_list is not None:
        ax = axs[4]
        for i in range(pred_state_list.shape[1]):
            ax.plot(pred_state_list[:, i], alpha=0.8)

    for ftype in ['png']:
        transparent = True
        if ftype == 'png':
            transparent = False
        plt.savefig('{}.{}'.format(fig_path, ftype), bbox_inches='tight', transparent=transparent, dpi=600)
    plt.show()

def fidelity_two_seqs(in_mats, out_mats):
    # Calculate the fidelity
    fidls = []
    Nmats = in_mats.shape[0]
    for n in range(Nmats):
        fidval = cal_fidelity_two_mats(in_mats[n], out_mats[n])
        fidls.append(fidval)
    return fidls

def fidelity_compute(qparams, train_len, val_len, buffer, ntrials, log_filename, \
    B, tBs, delay, taskname, order, flagplot, use_corr, \
    Nreps, reservoir, postprocess, test_lastrho, dat_label, noise_level):
    logger = get_module_logger(__name__, log_filename)
    logger.info(log_filename)
    length = buffer + train_len + val_len
    basename = log_filename.replace('.log', '')
    for tauB in tBs:
        qparams.tau = tauB / B
        train_rmean_ls, val_rmean_ls = [], []
        train_intar_ls, val_intar_ls = [], []
        for n in range(ntrials):
            input_data, output_data = generate_qtasks_delay(qparams.n_envs, ranseed=n, length=length, \
                delay=delay, taskname=taskname, order=order, Nreps=Nreps, buffer_train=buffer + train_len, dat_label=dat_label, noise_level=noise_level)
            train_input_seq = np.array(input_data[  : (buffer + train_len)])
            train_output_seq = np.array(output_data[  : (buffer + train_len)])
            
            train_intar_fidls = fidelity_two_seqs(np.array(output_data[  buffer : (buffer + train_len)]), \
                np.array(input_data[ (buffer-delay) : (buffer + train_len - delay)]))

            val_input_seq   = np.array(input_data[(buffer + train_len) : length])
            val_output_seq = np.array(output_data[(buffer + train_len) : length])
            
            val_intar_fidls = fidelity_two_seqs(np.array(output_data[(buffer + train_len) : length]), \
                np.array(input_data[(buffer + train_len - delay) : (length - delay)]))

            fig_path = '{}_tauB_{:.3f}_{}'.format(basename, tauB, n)

            train_pred_seq, train_fidls, val_pred_seq, val_fidls, pred_state_list = \
                qrc.get_fidelity(qparams, buffer, train_input_seq, train_output_seq, \
                    val_input_seq, val_output_seq, ranseed=n, \
                    use_corr=use_corr, reservoir=reservoir, postprocess=postprocess, test_lastrho=test_lastrho)
            
            #train_fid_avg, train_fid_std = np.mean(train_fidls), np.std(train_fidls)
            #val_fid_avg, val_fid_std = np.mean(val_fidls), np.std(val_fidls)
            train_rmean_square_fid = np.sqrt(np.mean(np.array(train_fidls)**2))
            val_rmean_square_fid = np.sqrt(np.mean(np.array(val_fidls)**2))
            
            train_rs_intar_fid = np.sqrt(np.mean(np.array(train_intar_fidls)**2))
            val_rs_intar_fid = np.sqrt(np.mean(np.array(val_intar_fidls)**2))
            
            res_title = 'Root mean square Fidelity at n={}, tauB={:.3f}, train={:.6f}, val={:.6f}, tr-intar={:.6f}, val-intar={:6f}'.format(\
                n, tauB, train_rmean_square_fid, val_rmean_square_fid, train_rs_intar_fid, val_rs_intar_fid)
            logger.debug(res_title)
            
            train_rmean_ls.append(train_rmean_square_fid)
            val_rmean_ls.append(val_rmean_square_fid)
            train_intar_ls.append(train_rs_intar_fid)
            val_intar_ls.append(val_rs_intar_fid)
            
            if flagplot > 0 and n == 0:
                plot_result(fig_path, res_title, train_input_seq[buffer:], train_output_seq[buffer:], \
                    val_input_seq, val_output_seq, train_pred_seq, val_pred_seq,\
                    train_fidls, val_fidls, pred_state_list)
        
        avg_train, avg_val = np.mean(train_rmean_ls), np.mean(val_rmean_ls)
        std_train, std_val = np.std(train_rmean_ls), np.std(val_rmean_ls)

        avg_intar_train, avg_intar_val = np.mean(train_intar_ls), np.mean(val_intar_ls)
        std_intar_train, std_intar_val = np.std(train_intar_ls), np.std(val_intar_ls)

        logger.info('Average RMSF with ntrials={}, tauB={:.3f}, avg-train={:.6f}, avg-val={:.6f}, std-train={:.6f}, std-val={:.6f}'.format(\
            ntrials, tauB, avg_train, avg_val, std_train, std_val))
        logger.debug('intar RMSF with ntrials={}, tauB={:.3f},  avg-train={:.6f}, avg-val={:.6f}, std-train={:.6f}, std-val={:.6f}'.format(\
            ntrials, tauB, avg_intar_train, avg_intar_val, std_intar_train, std_intar_val))

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
    parser.add_argument('--nreps', type=int, default=1, help='The input signal jumps into new random states every nreps')

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
    parser.add_argument('--usecorr', type=int, default=0, help='Use correlator operators')
    parser.add_argument('--reservoir', type=int, default=1, help='Use quantum reservoir to predict')
    parser.add_argument('--postprocess', type=int, default=1, help='Use post processing')
    parser.add_argument('--lastrho', type=int, default=1, help='Use last rho in test phase')

    parser.add_argument('--data', type=str, default='rand')
    parser.add_argument('--noise_level', type=float, default=0.3, help='Noise level')
    args = parser.parse_args()
    print(args)

    n_spins, n_envs, max_energy, beta, alpha, bcoef, init_rho = args.spins, args.envs, args.max_energy, args.beta, args.alpha, args.bcoef, args.rho
    tmin, tmax, ntaus, nreps = args.tmin, args.tmax, args.ntaus, args.nreps
    flagplot, usecorr = args.plot, args.usecorr
    use_reservoir, use_postprocess, test_lastrho = True, True, True
    if args.reservoir == 0:
        use_reservoir = False
    if args.postprocess == 0:
        use_postprocess = False
    if args.lastrho == 0:
        test_lastrho = False
    noise_level = args.noise_level

    dynamic = args.dynamic
    train_len, val_len, buffer, delay = args.trainlen, args.vallen, args.buffer, args.delay
    ntrials, basename, savedir, taskname, order = args.ntrials, args.basename, args.savedir, args.taskname, args.order
    virtuals = [int(x) for x in args.virtuals.split(',')]

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
    
    basename = 'qrc_{}_{}_post_{}_lastr_{}_{}_{}_od_{}_noise_{}_{}_corr_{}_nspins_{}_{}_a_{}_bc_{}_tmax_{}_tmin_{}_ntaus_{}_Vs_{}_len_{}_{}_{}_d_{}_trials_{}_reps_{}'.format(\
        args.reservoir, args.data, args.postprocess, args.lastrho,\
        basename, taskname, order, noise_level, dynamic, usecorr, n_spins, n_envs, alpha, bcoef, tmax, tmin, ntaus, \
        '_'.join([str(v) for v in virtuals]), buffer, train_len, val_len, delay, ntrials, nreps)
    
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
                    args=(qparams, train_len, val_len, buffer, ntrials, log_filename, B, tBs, delay, \
                        taskname, order, flagplot, usecorr, nreps, use_reservoir, use_postprocess, test_lastrho, dat_label, noise_level))
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
