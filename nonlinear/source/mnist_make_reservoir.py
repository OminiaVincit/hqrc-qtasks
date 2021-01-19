#!/usr/bin/env python
"""
Separate spiral data in nonlinear map
"""

import sys
import numpy as np
import os
import scipy
import argparse
import multiprocessing
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import ticker
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from scipy.sparse import linalg as splinalg
from scipy.linalg import pinv2 as scipypinv2

import gzip
import _pickle as cPickle
import pickle

import time
import datetime
import hqrc as hqrc
from loginit import get_module_logger

import utils
from utils import *

MNIST_DIR = "../data/mnist"
TAU_MNIST_DIR = "../data/mnist/taudata"
MNIST_SIZE = '10x10'

def get_reservoir_states(Xs, n_qrs, qparams, model, init_rs, ranseed, savefile, flip):
    rstates = []
    for s in Xs:
        if flip == True:
            input_signals = np.array(np.flip(s))
        else:
            input_signals = np.array(s)
        input_signals = np.tile(input_signals, (n_qrs, 1))
        output_signals = model.init_forward(qparams, input_signals, init_rs=init_rs, ranseed = ranseed)
        rstates.append(output_signals.ravel())
    rstates = np.array(rstates)
    with open(savefile, 'wb') as wrs:
            pickle.dump(rstates, wrs)


if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--nqrs', type=int, default=1, help='Number of QRs')
    parser.add_argument('--units', type=str, default='6', help='Number of the hidden units')
    parser.add_argument('--rho', type=int, default=0, help='Flag for initializing the density matrix')
    parser.add_argument('--coupling', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1e-14)

    parser.add_argument('--nproc', type=int, default=50)
    parser.add_argument('--virtual', type=int, default=1)
    parser.add_argument('--taus', type=str, default='14.0', help='Input interval')

    parser.add_argument('--nondiag', type=float, default=0.42, help='Nonlinear term (non-diagonal term)')
    parser.add_argument('--dynamic', type=str, default='ion_trap', help='full_random,full_const_trans,full_const_coeff,ion_trap')
    parser.add_argument('--basename', type=str, default='qrc')
    parser.add_argument('--savedir', type=str, default=TAU_MNIST_DIR)

    args = parser.parse_args()
    print(args)

    n_qrs, max_energy, beta, g, init_rho = args.nqrs, args.coupling, args.beta, args.nondiag, args.rho
    dynamic = args.dynamic
    V = args.virtual

    basename, savedir = args.basename, args.savedir
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    if os.path.isdir(savedir) == True:
        logdir = os.path.join(savedir, 'log')
    else:
        logdir = os.path.join(os.path.dirname(__file__), 'log')

    if os.path.isdir(logdir) == False:
        os.mkdir(logdir)

    units = [int(x) for x in args.units.split(',')]
    taudeltas = [float(x) for x in args.taus.split(',')]
    if len(taudeltas) == 0:
        taudeltas = np.linspace(1.0, 50.0, 99)
    
    Xs_train, Ys_train, Xs_test, Ys_test, Xs_val, Ys_val = gen_mnist_dataset(MNIST_DIR, MNIST_SIZE)
    
    for n_units in units:
        for tau_delta in taudeltas:
            basename = '{}_{}_{}_units_{}_tau_{:.3f}_V_{}'.format(MNIST_SIZE, args.basename, dynamic, n_units, tau_delta, V)
            log_filename = os.path.join(logdir, '{}.log'.format(basename))
            logger = get_module_logger(__name__, log_filename)
            logger.info(log_filename)
            logger.info('tau_delta={}, mnist shape train = {}, test = {}, val={}'.format(tau_delta, Xs_train.shape, Xs_test.shape, Xs_val.shape))

            if os.path.isfile(savedir) == False:
                qparams = QRCParams(n_units=n_units, max_energy=max_energy, non_diag=g,\
                    beta=beta, virtual_nodes=V, tau=tau_delta, init_rho=init_rho, dynamic=dynamic)
                model = hqrc.HQRC(nqrc=n_qrs, alpha=0.0, sparsity=1.0, sigma_input=1.0, type_input=0.0, use_corr=0)
                model.init_reservoir(qparams, ranseed=0)

                for lb in ['train', 'test', 'val']:
                    if lb == 'train':
                        X = Xs_train
                    elif lb == 'test':
                        X = Xs_test
                    else:
                        X = Xs_val
                    
                    # get reservoir states
                    tx = list(range(X.shape[0]))
                    #tx = list(range(10))
                    
                    nproc = min(len(tx), args.nproc)
                    lst = np.array_split(tx, nproc)

                    for flip in [False, True]:
                        jobs, pipels = [], []
                        tmp_files = []
                        for pid in range(nproc):
                            xs = lst[pid]
                            init_rs = False
                            ranseed = 0
                            savefile = os.path.join(savedir, 'temp_{}_{}_{}.binaryfile'.format(lb, basename, pid))
                            tmp_files.append(savefile)

                            p = multiprocessing.Process(target=get_reservoir_states, args=(X[xs], n_qrs, qparams, model, init_rs, ranseed, savefile, flip))
                            jobs.append(p)

                        # Start the process
                        for p in jobs:
                            p.start()

                        # Ensure all processes have finished execution
                        for p in jobs:
                            p.join()

                        # Sleep 5s
                        time.sleep(5)

                        # Joint dumped temp data file
                        zarr = []
                        for filename in tmp_files:
                            with open(filename, 'rb') as rrs:
                                tmp = pickle.load(rrs)
                                for arr in tmp:
                                    zarr.append(arr)
                            
                            # Delete file
                            os.remove(filename)
                        zarr = np.array(zarr)
                        logger.info('{}_{}'.format(lb, zarr.shape))

                        filename = os.path.join(savedir, '{}_{}_flip_{}.binaryfile'.format(lb, basename, flip))
                        with open(filename, 'wb') as wrs:
                            pickle.dump(zarr, wrs)
