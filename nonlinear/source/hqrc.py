#!/usr/bin/env python
"""
    Higher-order reservoir class
"""

import sys
import numpy as np
import scipy as sp
from scipy import sparse as sparse
from scipy.sparse import linalg as splinalg
from scipy.linalg import pinv2 as scipypinv2
from IPC import *
from utils import *

import time
from datetime import timedelta

class HQRC(object):
    def __init__(self, nqrc, alpha, sparsity, sigma_input, type_input=0):
        self.nqrc = nqrc
        self.alpha = alpha
        self.sparsity = sparsity
        self.sigma_input = sigma_input
        self.type_input = type_input

    def __init_reservoir(self, qparams, ranseed):
        if ranseed >= 0:
            np.random.seed(seed=ranseed)
        
        I = [[1,0],[0,1]]
        Z = [[1,0],[0,-1]]
        X = [[0,1],[1,0]]
        P0 = [[1,0],[0,0]]
        P1 = [[0,0],[0,1]]
        self.n_units = qparams.n_units
        self.virtual_nodes = qparams.virtual_nodes
        self.tau = qparams.tau
        self.max_energy = qparams.max_energy
        self.non_diag = qparams.non_diag
        self.solver = qparams.solver

        self.n_qubits = self.n_units
        self.dim = 2**self.n_qubits
        self.Zop = [1]*self.n_qubits
        self.Xop = [1]*self.n_qubits
        self.P0op = [1]
        self.P1op = [1]
        
        nqrc = self.nqrc
        Nspins = self.n_qubits
        dynamic = qparams.dynamic

        # generate feedback matrix
        n_nodes = self.__get_comput_nodes()
        W_feed = np.zeros((n_nodes, nqrc))
        for i in range(0, self.nqrc):
            smat = sparse.random(n_nodes, 1, density = self.sparsity).data
            smat *= (self.sigma_input / n_nodes) * (smat * 2.0 - 1.0)
            W_feed[:, i] = smat.ravel()
        self.W_feed = W_feed

        # create operators from tensor product
        for cindex in range(self.n_qubits):
            for qindex in range(self.n_qubits):
                if cindex == qindex:
                    self.Xop[qindex] = np.kron(self.Xop[qindex],X)
                    self.Zop[qindex] = np.kron(self.Zop[qindex],Z)
                else:
                    self.Xop[qindex] = np.kron(self.Xop[qindex],I)
                    self.Zop[qindex] = np.kron(self.Zop[qindex],I)

            if cindex == 0:
                self.P0op = np.kron(self.P0op, P0)
                self.P1op = np.kron(self.P1op, P1)
            else:
                self.P0op = np.kron(self.P0op, I)
                self.P1op = np.kron(self.P1op, I)

        # initialize current states
        self.cur_states = [None] * nqrc

        # Intialize evolution operators
        tmp_uops = []
        tmp_rhos = []
        for i in range(nqrc):
            # initialize density matrix
            rho = np.zeros( [self.dim, self.dim] )
            rho[0, 0] = 1
            if qparams.init_rho != 0:
                # initialize random density matrix
                rho = random_density_matrix(self.dim)
            tmp_rhos.append(rho)

            # generate hamiltonian
            hamiltonian = np.zeros( (self.dim,self.dim) )

            # create coupling strength for ion trap
            a = 0.2
            bc = self.non_diag # bc = 0.42
            J = 0
            for qindex1 in range(Nspins):
                for qindex2 in range(qindex1+1, Nspins):
                    Jij = np.abs(qindex2-qindex1)**(-a)
                    J += Jij / (Nspins-1)
            B = J/bc # Magnetic field

            for qindex in range(Nspins):
                if dynamic == DYNAMIC_FULL_RANDOM:
                    coef = (np.random.rand()-0.5) * 2 * self.max_energy
                elif dynamic == DYNAMIC_ION_TRAP:
                    coef = - B * self.max_energy
                else:
                    coef = - self.non_diag * self.max_energy
                hamiltonian += coef * self.Zop[qindex]

            for qindex1 in range(Nspins):
                for qindex2 in range(qindex1+1, Nspins):
                    if dynamic == DYNAMIC_FULL_CONST_COEFF:
                        coef =  - self.max_energy
                    elif dynamic == DYNAMIC_ION_TRAP:
                        coef =  - np.abs(qindex2 - qindex1)**(-a) / J
                        coef = 2 * self.max_energy * coef
                    else:
                        coef = (np.random.rand()-0.5) * 2 * self.max_energy
                    hamiltonian += coef * self.Xop[qindex1] @ self.Xop[qindex2]
                    
            ratio = float(self.tau) / float(self.virtual_nodes)        
            Uop = sp.linalg.expm(-1.j * hamiltonian * ratio)
            tmp_uops.append(Uop)
        
        self.init_rhos = tmp_rhos.copy()
        self.last_rhos = tmp_rhos.copy()
        self.Uops = tmp_uops.copy()

    def __get_comput_nodes(self):
        return self.n_units * self.virtual_nodes * self.nqrc
    
    def __reset_states(self):
        self.cur_states = [None] * self.nqrc

    def gen_rand_rhos(self, ranseed):
        if ranseed >= 0:
            np.random.seed(seed=ranseed)
        tmp_rhos = []
        for i in range(self.nqrc):
            rho = random_density_matrix(self.dim)
            tmp_rhos.append(rho)
        self.init_rhos = tmp_rhos.copy()

    def step_forward(self, local_rhos, input_val):
        nqrc = self.nqrc
        update_input = input_val.copy().ravel()

        if self.alpha > 0 and self.cur_states[0] is not None:
            tmp_states = np.array(self.cur_states, dtype=np.float64).reshape(1, -1)
            tmp_states = tmp_states @ self.W_feed
            tmp_states = tmp_states.ravel()
            update_input = self.alpha * tmp_states + (1.0 - self.alpha) * update_input
            
        for i in range(nqrc):
            Uop = self.Uops[i]
            rho = local_rhos[i]
            # Obtain value from the input
            value = update_input[i]

            # Replace the density matrix
            rho = self.P0op @ rho @ self.P0op + self.Xop[0] @ self.P1op @ rho @ self.P1op @ self.Xop[0]
            # (1 + u Z)/2 = (1+u)/2 |0><0| + (1-u)/2 |1><1|
            # inv1 = (self.affine[1] + self.value) / self.affine[0]
            # inv2 = (self.affine[1] - self.value) / self.affine[0]

            if self.type_input == 0:
                # for input in [0, 1]
                rho = (1 - value) * rho + value * self.Xop[0] @ rho @ self.Xop[0]
            else:
                # for input in [-1, 1]
                rho = ((1+value)/2) * rho + ((1-value)/2) *self.Xop[0] @ rho @ self.Xop[0]
            
            current_state = []
            for v in range(self.virtual_nodes):
                # Time evolution of density matrix
                rho = Uop @ rho @ Uop.T.conj()
                for qindex in range(1, self.n_qubits):
                    expectation_value = np.real(np.trace(self.Zop[qindex] @ rho))
                    current_state.append(expectation_value)
            # Size of current_state is Nqubits x Nvirtuals)
            self.cur_states[i] = np.array(current_state, dtype=np.float64)
            local_rhos[i] = rho
        return local_rhos

    def __feed_forward(self, input_seq, predict, use_lastrho):
        input_dim, input_length = input_seq.shape
        nqrc = self.nqrc
        assert(input_dim == nqrc)
        
        predict_seq = None
        local_rhos = self.init_rhos.copy()
        if use_lastrho == True :
            local_rhos = self.last_rhos.copy()
        
        state_list = []
        for time_step in range(0, input_length):
            input_val = input_seq[:, time_step].ravel()
            local_rhos = self.step_forward(local_rhos, input_val)

            state = np.array(self.cur_states.copy(), dtype=np.float64)
            state_list.append(state.flatten())

        state_list = np.array(state_list)
        self.last_rhos = local_rhos.copy()

        if predict:
            stacked_state = np.hstack( [state_list, np.ones([input_length, 1])])
            predict_seq = stacked_state @ self.W_out
        
        return predict_seq, state_list


    def __train(self, input_seq, output_seq, buffer, beta):
        assert(input_seq.shape[1] == output_seq.shape[0])
        Nout = output_seq.shape[1]
        self.W_out = np.random.rand(self.__get_comput_nodes() + 1, Nout)

        _, state_list = self.__feed_forward(input_seq, predict=False, use_lastrho=False)

        state_list = np.array(state_list)
        state_list = state_list[buffer:, :]

        # discard the transitient state for training
        X = np.reshape(state_list, [-1, self.__get_comput_nodes()])
        X = np.hstack( [state_list, np.ones([X.shape[0], 1]) ] )

        discard_output = output_seq[buffer:, :]
        Y = np.reshape(discard_output, [discard_output.shape[0], -1])
        
        if self.solver == LINEAR_PINV:
            self.W_out = np.linalg.pinv(X, rcond = beta) @ Y
        else:
            XTX = X.T @ X
            XTY = X.T @ Y
            if self.solver == RIDGE_PINV:
                I = np.identity(np.shape(XTX)[1])	
                pinv_ = scipypinv2(XTX + beta * I)
                self.W_out = pinv_ @ XTY
            elif self.solver in ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']:
                ridge = Ridge(alpha=beta, fit_intercept=False, normalize=False, copy_X=True, solver=self.solver)
                ridge.fit(XTX, XTY)
                self.W_out = np.array(ridge.coef_).reshape((-1, Nout))
            else:
                raise ValueError('Undefined solver')

    def train_to_predict(self, input_seq, output_seq, buffer, qparams, ranseed):
        self.__init_reservoir(qparams, ranseed)
        self.__train(input_seq, output_seq, buffer, qparams.beta)

    def predict(self, input_seq, output_seq, buffer, use_lastrho):
        prediction_seq, _ = self.__feed_forward(input_seq, predict=True, use_lastrho=use_lastrho)
        pred = prediction_seq[buffer:, :]
        out  = output_seq[buffer:, :]
        loss = np.sum((pred - out)**2) / np.sum(pred**2)
        return prediction_seq, loss

    def init_forward(self, qparams, input_seq, init_rs, ranseed):
        self.__reset_states()
        if init_rs == True:
            self.__init_reservoir(qparams, ranseed)
        _, state_list =  self.__feed_forward(input_seq, predict=False, use_lastrho=False)
        return state_list

def get_loss(qparams, buffer, train_input_seq, train_output_seq, \
        val_input_seq, val_output_seq, nqrc, alpha, sparsity, sigma_input, ranseed, type_input=0):
    model = HQRC(nqrc, alpha, sparsity, sigma_input, type_input)

    train_input_seq = np.array(train_input_seq)
    train_output_seq = np.array(train_output_seq)
    model.train_to_predict(train_input_seq, train_output_seq, buffer, qparams, ranseed)

    train_pred_seq, train_loss = model.predict(train_input_seq, train_output_seq, buffer=buffer, use_lastrho=False)
    #print("train_loss={}, shape".format(train_loss), train_pred_seq_ls.shape)
    
    # Test phase
    val_input_seq = np.array(val_input_seq)
    val_output_seq = np.array(val_output_seq)
    val_pred_seq, val_loss = model.predict(val_input_seq, val_output_seq, buffer=0, use_lastrho=True)
    #print("val_loss={}, shape".format(val_loss), val_pred_seq_ls.shape)

    return train_pred_seq, train_loss, val_pred_seq, val_loss

def get_IPC(qparams, ipcparams, length, logger, ranseed=-1, Ntrials=1, savedir=None, \
    posfix='capa', type_input=0, label=''):
    start_time = time.monotonic()
    fname = '{}_{}'.format(label, sys._getframe().f_code.co_name)
    nqrc = 1
    transient = length // 2

    if ranseed >= 0:
        np.random.seed(seed=ranseed)
    for n in range(Ntrials):
        if type_input == 0:
            input_signals = np.random.uniform(0, 1, length) 
        else:
            input_signals = np.random.uniform(-1, 1, length)
        input_signals = np.array(input_signals)
        input_signals = np.tile(input_signals, (nqrc, 1))

        ipc = IPC(ipcparams, log=logger, savedir=savedir, label=label)
        model = HQRC(nqrc=nqrc, alpha=0.0, sparsity=1.0, sigma_input=1.0, type_input=type_input)
        output_signals = model.init_forward(qparams, input_signals, init_rs=True, ranseed = n + ranseed)
        logger.debug('{}: n={} per {} trials, input shape = {}, output shape={}'.format(fname, n+1, Ntrials, input_signals.shape, output_signals.shape))
        
        ipc.run(input_signals[0, transient:], output_signals[transient:])
        ipc.write_results(posfix=posfix)
    end_time = time.monotonic()
    logger.info('{}: Executed time {}'.format(fname, timedelta(seconds=end_time - start_time)))

def memory_function(taskname, qparams, train_len, val_len, buffer, dlist, \
        nqrc, alpha, sparsity, sigma_input, ranseed=-1, Ntrials=1, type_input=0):    
    MFlist = []
    MFstds = []
    train_list, val_list = [], []
    length = buffer + train_len + val_len
    # generate data
    if '_stm' not in taskname and '_pc' not in taskname:
        raise ValueError('Not found taskname ={} to generate data'.format(taskname))

    if ranseed >= 0:
        np.random.seed(seed=ranseed)
    
    if '_pc' in taskname:
        print('Generate parity check data')
        data = np.random.randint(0, 2, length)
    else:
        print('Generate STM task data')
        if type_input == 0:
            data = np.random.rand(length)
        else:
            data = 2.0*np.random.rand(length) - 1.0

    for d in dlist:
        train_input_seq = np.array(data[  : buffer + train_len])
        train_input_seq = np.tile(train_input_seq, (nqrc, 1))
        
        val_input_seq = np.array(data[buffer + train_len : length])
        val_input_seq = np.tile(val_input_seq, (nqrc, 1))
            
        train_out, val_out = [], []
        if '_pc' in taskname:
            for k in range(length):
                yk = 0
                if k >= d:
                    yk = np.sum(data[k-d : k+1]) % 2
                if k >= buffer + train_len:
                    val_out.append(yk)
                else:
                    train_out.append(yk)
        else:
            for k in range(length):
                yk = 0
                if k >= d:
                    yk = data[k-d]
                if k >= buffer + train_len:
                    val_out.append(yk)
                else:
                    train_out.append(yk)
        
        train_output_seq = np.array(train_out).reshape(len(train_out), 1)
        val_output_seq = np.array(val_out).reshape(len(val_out), 1)
        
        train_loss_ls, val_loss_ls, mfs = [], [], []
        for n in range(Ntrials):
            ranseed_net = ranseed
            if ranseed >= 0:
                ranseed_net = (ranseed + 10000) * (n + 1)
            # Use the same ranseed the same trial
            train_pred_seq, train_loss, val_pred_seq, val_loss = \
                get_loss(qparams, buffer, train_input_seq, train_output_seq, \
                    val_input_seq, val_output_seq, nqrc, alpha, sparsity, sigma_input, ranseed_net, type_input)

            # Compute memory function
            val_out_seq, val_pred_seq = val_output_seq.flatten(), val_pred_seq.flatten()
            #print('cov', val_output_seq.shape, val_pred_seq.shape)
            cov_matrix = np.cov(np.array([val_out_seq, val_pred_seq]))
            MF_d = cov_matrix[0][1] ** 2
            MF_d = MF_d / (np.var(val_out_seq) * np.var(val_pred_seq))
            # print('d={}, n={}, MF={}'.format(d, n, MF_d))
            train_loss_ls.append(train_loss)
            val_loss_ls.append(val_loss)
            mfs.append(MF_d)

        avg_train, avg_val, avg_MFd, std_MFd = np.mean(train_loss_ls), np.mean(val_loss_ls), np.mean(mfs), np.std(mfs)
        MFlist.append(avg_MFd)
        MFstds.append(std_MFd)
        train_list.append(avg_train)
        val_list.append(avg_val)
    
    return np.array(list(zip(dlist, MFlist, MFstds, train_list, val_list)))

def effective_dim(qparams, buffer, length, nqrc, alpha, sparsity, sigma_input, ranseed, Ntrials):
    # Calculate effective dimension for reservoir
    from numpy import linalg as LA
    
    if ranseed >= 0:
        np.random.seed(seed=ranseed)

    data = np.random.rand(length)
    input_seq = np.array(data)
    input_seq = np.tile(input_seq, (nqrc, 1))

    model = HQRC(nqrc, alpha, sparsity, sigma_input)

    effdims = []
    for n in range(Ntrials):
        ranseed_net = ranseed
        if ranseed >= 0:
            ranseed_net = (ranseed + 11000) * (n + 1)
        
        state_list = model.init_forward(qparams, input_seq, init_rs=True, ranseed=ranseed_net)
        L, D = state_list.shape
        # L = Length of time series
        # D = Number of virtual nodes x Number of qubits
        locls = []
        for i in range(D):
            for j in range(D):
                ri = state_list[buffer:, i]
                rj = state_list[buffer:, j]
                locls.append(np.mean(ri*rj))
        locls = np.array(locls).reshape(D, D)
        w, v = LA.eig(locls)
        #print(w)
        w = np.abs(w) / np.abs(w).sum()
        effdims.append(1.0 / np.power(w, 2).sum())
    return np.mean(effdims), np.std(effdims)

def esp_index(qparams, buffer, length, nqrc, alpha, sparsity, sigma_input, ranseed, state_trials):
    if ranseed >= 0:
        np.random.seed(seed=ranseed)

    data = np.random.rand(length)
    input_seq = np.array(data)
    input_seq = np.tile(input_seq, (nqrc, 1))

    # Initialize the reservoir to zero state - density matrix
    model = HQRC(nqrc, alpha, sparsity, sigma_input)
    x0_state_list = model.init_forward(qparams, input_seq, init_rs = True, ranseed = ranseed)
    # Compute esp index and esp_lambda
    dP = []
    for i in range(state_trials):
        # Initialzie the reservoir to a random initial state
        # Keep same coupling configuration
        model.gen_rand_rhos(ranseed = i + 300000)
        z0_state_list = model.init_forward(qparams, input_seq, init_rs = False, ranseed = i + 200000)
        L, D = z0_state_list.shape
        # L = Length of time series
        # D = Number of layers x Number of virtual nodes x Number of qubits
        # print('i={}, State shape'.format(i), z0_state_list.shape)
        local_diff = 0
        # prev, current = None, None
        for t in range(buffer, L):
            diff_state = x0_state_list[t, :] - z0_state_list[t, :]
            diff = np.sqrt(np.power(diff_state, 2).sum())
            local_diff += diff
        local_diff = local_diff / (L-buffer)
        dP.append(local_diff)
    return np.mean(dP)

def lyapunov_exp(qparams, buffer, length, nqrc, alpha, sparsity, sigma_input, ranseed, initial_distance):
    if ranseed >= 0:
        np.random.seed(seed=ranseed)

    data = np.random.rand(length)
    input_seq = np.array(data)
    input_seq = np.tile(input_seq, (nqrc, 1))

    # Initialize the reservoir to zero state - density matrix
    model = HQRC(nqrc, alpha, sparsity, sigma_input)
    states1 = model.init_forward(qparams, input_seq, init_rs = True, ranseed = -1)
    L, D = states1.shape
    # L = Length of time series
    # D = Number of layers x Number of virtual nodes x Number of qubits
    lyps = []
    for n in range(int(D / nqrc)):
        if n % qparams.n_units == 0:
            # Skip the input qubits
            continue
        model.init_forward(qparams, input_seq[:buffer], init_rs = False, ranseed = -1)
        states2 = np.zeros((L, D))
        states2[buffer-1, :] = states1[buffer-1, :]
        states2[buffer-1, n] = states1[buffer-1, n] + initial_distance
        gamma_k_list = []
        local_rhos = model.last_rhos.copy()
        for k in range(buffer, L):
            input_val = input_seq[:, k].ravel()
            local_rhos = model.step_forward(local_rhos, input_val)
            states2[k, :] = np.array(model.cur_states, dtype=np.float64).flatten()
            # Add to gamma list and update states
            gamma_k = np.linalg.norm(states2[k, :] - states1[k, :])
            gamma_k_list.append(gamma_k / initial_distance)
            states2[k, :] = states1[k, :] + (initial_distance / gamma_k) * (states2[k, :] - states1[k, :])

        lyps.append(np.mean(np.log(gamma_k_list)))
    lyapunov_exp = np.mean(lyps)
    return lyapunov_exp