#!/usr/bin/env python
"""
Separate spiral data in nonlinear map
"""

import sys
import numpy as np
import os
import scipy
from scipy.sparse import linalg as splinalg
from scipy.linalg import pinv2 as scipypinv2

import argparse
import multiprocessing
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import ticker
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler

import time
import datetime
import hqrc as hqrc
from loginit import get_module_logger

import utils
from utils import *

def gen_spiral_dataset(N):
    theta = np.sqrt(np.random.rand(N))*2*np.pi # np.linspace(0,2*pi,100)

    r_a = 2*theta + np.pi
    data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
    x_a = (data_a + np.random.randn(N,2)) / 20.0

    r_b = -2*theta - np.pi
    data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
    x_b = (data_b + np.random.randn(N,2)) / 20.0

    res_a = np.append(x_a, np.zeros((N,1)), axis=1)
    res_b = np.append(x_b, np.ones((N,1)), axis=1)

    res = np.append(res_a, res_b, axis=0)
    np.random.shuffle(res)

    # plt.scatter(x_a[:,0],x_a[:,1])
    # plt.scatter(x_b[:,0],x_b[:,1])
    # plt.show()
    return res

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--nqrs', type=int, default=1, help='Number of QRs')
    parser.add_argument('--units', type=int, default=6, help='Number of the hidden units')
    parser.add_argument('--rho', type=int, default=0, help='Flag for initializing the density matrix')
    parser.add_argument('--coupling', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1e-14)

    parser.add_argument('--nproc', type=int, default=50)
    parser.add_argument('--ntrials', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--virtuals', type=str, default='1')
    parser.add_argument('--tau', type=float, default=14.0, help='Input interval')

    parser.add_argument('--nondiag', type=float, default=0.42, help='Nonlinear term (non-diagonal term)')
    parser.add_argument('--dynamic', type=str, default='ion_trap', help='full_random,full_const_trans,full_const_coeff,ion_trap')
    parser.add_argument('--basename', type=str, default='qrc')
    parser.add_argument('--savedir', type=str, default='res_spiral')
    
    parser.add_argument('--ndata', type=int, default=100)
    parser.add_argument('--usecorr', type=int, default=1)
    
    args = parser.parse_args()
    print(args)

    n_qrs, n_units, max_energy, beta, g, init_rho = args.nqrs, args.units, args.coupling, args.beta, args.nondiag, args.rho
    dynamic = args.dynamic
    ranseed = args.seed
    Ndata = args.ndata
    tau_delta = args.tau
    use_corr = args.usecorr

    ntrials, basename, savedir = args.ntrials, args.basename, args.savedir
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

    basename = '{}_{}_seed_{}'.format(basename, dynamic, ranseed)
    log_filename = os.path.join(logdir, '{}.log'.format(basename))
    logger = get_module_logger(__name__, log_filename)
    logger.info(log_filename)

    virtuals = [int(x) for x in args.virtuals.split(',')]

    if os.path.isfile(savedir) == False:
        spiral = gen_spiral_dataset(Ndata)
        print('spiral', spiral.shape)
        fig = plt.figure(figsize=(20, 10), dpi=600)

        ax1 = plt.subplot2grid((1, 2), (0,0), colspan=1, rowspan=1)
        id_a = spiral[:, 2] == 0
        id_b = spiral[:, 2] == 1
        x_a = spiral[id_a]
        x_b = spiral[id_b]
        ax1.scatter(x_a[:,0], x_a[:,1])
        ax1.scatter(x_b[:,0], x_b[:,1])
        
        jobs, pipels = [], []
        for V in virtuals:
            #for tau_delta in taudeltas:
            log_filename = os.path.join(logdir, 'tau_{:.3f}_V_{}_{}.log'.format(tau_delta, V, basename))
            outbase = os.path.join(figdir, '{}_tau_{:.3f}_V_{}'.format(basename, tau_delta, V))

            qparams = QRCParams(n_units=n_units, max_energy=max_energy, non_diag=g,\
                        beta=beta, virtual_nodes=V, tau=tau_delta, init_rho=init_rho, dynamic=dynamic)
            model = hqrc.HQRC(nqrc=n_qrs, alpha=0.0, sparsity=1.0, sigma_input=1.0, type_input=1.0, use_corr=use_corr)
            X_train, Y_train = [], []
            init_rs = True
            for s in spiral:
                input_signals = np.array(s[:2])
                input_signals = np.tile(input_signals, (n_qrs, 1))
                output_signals = model.init_forward(qparams, input_signals, init_rs=init_rs, ranseed = ranseed)
                #print(input_signals, output_signals)
                X_train.append(output_signals.ravel())
                lb = [1, -1]
                if s[2] > 0:
                    lb = [-1, 1]
                Y_train.append(lb)

                init_rs = False
            X_train = np.array(X_train)
            X_train = np.hstack( [X_train, np.ones([X_train.shape[0], 1]) ] )

            Y_train = np.array(Y_train)
            print(X_train.shape, Y_train.shape)

            # Training
            XTX = X_train.T @ X_train
            XTY = X_train.T @ Y_train
            I = np.identity(np.shape(XTX)[1])	
            pinv_ = scipypinv2(XTX + beta * I)
            W_out = pinv_ @ XTY
            predict = X_train @ W_out
            print(W_out.shape, predict.shape)
            pred_lb = []
            for p in predict:
                if p[0] > p[1]:
                    pred_lb.append(0)
                else:
                    pred_lb.append(1)
            pred_lb = np.array(pred_lb)
            out_lb = spiral[:, 2]
            err = np.sum((pred_lb - out_lb)**2) / pred_lb.shape[0]
            print(err)            

            # cov_mat = np.cov(en_states.T)
            # eig_vals, eig_vecs = np.linalg.eig(cov_mat)
            # sc = StandardScaler()
            # X_train = sc.fit_transform(en_states)

            # pca = PCA(n_components = 2)
            # X_pca = pca.fit_transform(X_train)
            # print(X_pca.shape)
            # #ax2 = plt.subplot2grid((1, 2), (0, 1), colspan=1, rowspan=1, projection='3d')
            # # ax2.scatter3D(X_pca[id_a, 0], X_pca[id_a, 1], X_pca[id_a, 2])
            # # ax2.scatter3D(X_pca[id_b, 0], X_pca[id_b, 1], X_pca[id_b, 2])
            # ax2 = plt.subplot2grid((1, 2), (0, 1), colspan=1, rowspan=1)
            # ax2.scatter(X_pca[id_a, 0], X_pca[id_a, 1])
            # ax2.scatter(X_pca[id_b, 0], X_pca[id_b, 1])

            for ftype in ['png']:
                plt.savefig('{}_v1.{}'.format(outbase, ftype), bbox_inches='tight', dpi=600)
            plt.show()
        # # Start the process
        # for p in jobs:
        #     p.start()

        # # Ensure all processes have finished execution
        # for p in jobs:
        #     p.join()

        # # Sleep 5s
        # time.sleep(5)
