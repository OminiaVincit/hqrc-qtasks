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

import time
import datetime
import pickle

import hqrc as hqrc
from loginit import get_module_logger

import utils
from utils import *

MNIST_DIR = "/data/zoro/qrep/mnist"
TAU_MNIST_DIR = "/data/zoro/qrep/mnist/taudata"
RES_MNIST_DIR = "results"
MNIST_SIZE="10x10"

def get_acc(predict, out_lb):
    pred_lb = np.argmax(predict, axis=1)
    acc = np.sum(pred_lb == out_lb) / pred_lb.shape[0]
    return acc

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--nqrs', type=int, default=1, help='Number of QRs')
    parser.add_argument('--spins', type=int, default=5, help='Number of spins')
    parser.add_argument('--ntrials', type=int, default=1)
    parser.add_argument('--virtuals', type=str, default='1')
    parser.add_argument('--beta', type=float, default=1e-14)
    parser.add_argument('--tau', type=float, default=22.0, help='Input interval')

    parser.add_argument('--linear_reg', type=int, default=0)
    parser.add_argument('--use_corr', type=int, default=0)
    parser.add_argument('--full', type=int, default=1)
    parser.add_argument('--label1', type=int, default=3)
    parser.add_argument('--label2', type=int, default=6)
    
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha of coupled strength, 0 for random coupling')
    parser.add_argument('--bcoef', type=float, default=1.0, help='bcoeff nonlinear term (non-diagonal term)')

    parser.add_argument('--dynamic', type=str, default='ion_trap', help='full_random,full_const_trans,full_const_coeff,ion_trap')
    parser.add_argument('--basename', type=str, default='qrc')
    parser.add_argument('--savedir', type=str, default=RES_MNIST_DIR)
    parser.add_argument('--tau_mnist_dir', type=str, default=TAU_MNIST_DIR)
    parser.add_argument('--mnist_dir', type=str, default=MNIST_DIR)
    parser.add_argument('--mnist_size', type=str, default=MNIST_SIZE)

    args = parser.parse_args()
    print(args)

    n_qrs, n_spins, beta, alpha, bcoef = args.nqrs, args.spins, args.beta, args.alpha, args.bcoef
    dynamic = args.dynamic
    tau_delta = args.tau
    linear_reg, use_corr, full_mnist = args.linear_reg, args.use_corr, args.full
    label1, label2 = args.label1, args.label2
    ntrials, basename, savedir, mnist_dir, tau_mnist_dir, mnist_size = args.ntrials, args.basename, args.savedir, args.mnist_dir, args.tau_mnist_dir, args.mnist_size

    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    if os.path.isdir(savedir) == True:
        logdir = os.path.join(savedir, 'log')
        figdir = os.path.join(savedir, 'figs')
    else:
        logdir = os.path.join(os.path.dirname(__file__), 'log')
        figdir = os.path.join(os.path.dirname(__file__), 'figs')

    if os.path.isdir(logdir) == False:
        os.mkdir(logdir)
    
    if os.path.isdir(figdir) == False:
        os.mkdir(figdir)

    basename = '{}_{}_{}_nspins_{}_a_{}_bc_{}'.format(mnist_size, basename, dynamic, n_spins, alpha, bcoef)
    log_filename = os.path.join(logdir, '{}_V_{}.log'.format(basename, args.virtuals))
    logger = get_module_logger(__name__, log_filename)
    logger.info(log_filename)

    virtuals = [int(x) for x in args.virtuals.split(',')]
    taudeltas = [tau_delta]
    
    x_train, y_train_lb, x_test, y_test_lb, x_val, y_val = gen_mnist_dataset(mnist_dir, mnist_size)
    
    if full_mnist > 0:
        Y_train = np.identity(10)[y_train_lb]
        Y_test  = np.identity(10)[y_test_lb]
    else:
        train_ids = (y_train_lb == label1) | (y_train_lb == label2)
        x_train = x_train[train_ids, :]
        y_train_lb = y_train_lb[train_ids]
        y_train_lb[y_train_lb == label1] = 0
        y_train_lb[y_train_lb == label2] = 1

        Y_train = np.identity(2)[y_train_lb]

        test_ids = (y_test_lb == label1) | (y_test_lb == label2)
        x_test = x_test[test_ids, :]
        y_test_lb = y_test_lb[test_ids]
        y_test_lb[y_test_lb == label1] = 0
        y_test_lb[y_test_lb == label2] = 1

        Y_test  = np.identity(2)[y_test_lb]
    
    logger.info('shape y_train={},  y_test={}'.format(Y_train.shape, Y_test.shape))

    #fig = plt.figure(figsize=(20, 10), dpi=600)
    #jobs, pipels = [], []
    for V in virtuals:
        for tau_delta in taudeltas:
            for flip in ['True', 'False']:
                # get data
                pfix = '{}_tauB_{:.3f}_V_{}_buf_100_flip_{}'.format(basename, tau_delta, V, flip)
                train_file = os.path.join(tau_mnist_dir, 'train_{}.binaryfile'.format(pfix))
                test_file = os.path.join(tau_mnist_dir, 'test_{}.binaryfile'.format(pfix))
                
                if os.path.isfile(train_file) == False or os.path.isfile(test_file) == False:
                    continue
                
                if linear_reg > 0:
                    X_train = np.array(x_train)
                else:
                    # Training
                    with open(train_file, 'rb') as rrs:
                        X_train = pickle.load(rrs)
                        if use_corr == 0:
                            ids = np.array(range(X_train.shape[1]))
                            ids = ids[ids % 15 < 5]
                            X_train = X_train[:, ids]
                        if full_mnist == 0:
                            X_train = X_train[train_ids, :]

                X_train = np.hstack( [X_train, np.ones([X_train.shape[0], 1]) ] )
                logger.info('V={}, tauB={}, flip={}, X_train shape={}'.format(V, tau_delta, flip, X_train.shape))

                XTX = X_train.T @ X_train
                XTY = X_train.T @ Y_train
                I = np.identity(np.shape(XTX)[1])	
                pinv_ = scipypinv2(XTX + beta * I)
                W_out = pinv_ @ XTY
                logger.info('Wout shape={}'.format(W_out.shape))
                train_acc = get_acc(X_train @ W_out, y_train_lb)
                logger.info('Train acc={}'.format(train_acc))

                # Testing
                if linear_reg > 0:
                    X_test = np.array(x_test)
                else:
                    with open(test_file, 'rb') as rrs:
                        X_test = pickle.load(rrs)
                        if use_corr == 0:
                            ids = np.array(range(X_test.shape[1]))
                            ids = ids[ids % 15 < 5]
                            X_test = X_test[:, ids]
                        if full_mnist == 0:
                            X_test = X_test[test_ids, :]

                X_test = np.hstack( [X_test, np.ones([X_test.shape[0], 1]) ] )
                logger.info('V={}, tau={}, X_test shape = {}'.format(V, tau_delta, X_test.shape))
                test_acc = get_acc(X_test @ W_out, y_test_lb)
                logger.info('Test acc={}'.format(test_acc))


    # # Start the process
    # for p in jobs:
    #     p.start()

    # # Ensure all processes have finished execution
    # for p in jobs:
    #     p.join()

    # # Sleep 5s
    # time.sleep(5)
