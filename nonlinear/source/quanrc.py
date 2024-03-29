#!/usr/bin/env python
"""
    Higher-order reservoir class
"""

import sys
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg as splinalg
from scipy.linalg import pinv2 as scipypinv2
from utils import *
from qutils import *

import time
from datetime import timedelta

class QRC(object):
    def __init__(self, use_corr=0):
        self.use_corr = use_corr

    def __init_reservoir(self, qparams, ranseed):
        if ranseed >= 0:
            np.random.seed(seed=ranseed)
        
        I = [[1,0],[0,1]]
        Z = [[1,0],[0,-1]]
        X = [[0,1],[1,0]]
        P0 = [[1,0],[0,0]]
        P1 = [[0,0],[0,1]]

        self.n_units = qparams.n_units
        self.n_envs = qparams.n_envs
        self.virtual_nodes = qparams.virtual_nodes
        self.tau = qparams.tau
        self.max_energy = qparams.max_energy
        self.non_diag = qparams.non_diag
        self.alpha = qparams.alpha
        self.solver = qparams.solver
        self.dynamic = qparams.dynamic

        self.n_qubits = self.n_units + self.n_envs
        self.dim = 2**self.n_qubits
        self.Zop = [1]*self.n_qubits

        self.Zop_corr = dict()
        for q1 in range(self.n_qubits):
            for q2 in range(q1+1, self.n_qubits):
                self.Zop_corr[(q1, q2)] = [1]
        
        self.Xop = [1]*self.n_qubits
        self.P0op = [1]
        self.P1op = [1]
        
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

        # generate correlatior operators
        if self.use_corr > 0:
            for q1 in range(self.n_qubits):
                for q2 in range(q1+1, self.n_qubits):
                    cindex = (q1, q2)
                    for qindex in range(self.n_qubits):
                        if qindex == q1 or qindex == q2:
                            self.Zop_corr[cindex] = np.kron(self.Zop_corr[cindex], Z)
                        else:
                            self.Zop_corr[cindex] = np.kron(self.Zop_corr[cindex], I)
        
        # initialize current states
        self.cur_states = [None]

        # create coupling strength for ion trap
        a = self.alpha
        bc = self.non_diag
        Nalpha = 0
        for qindex1 in range(self.n_qubits):
            for qindex2 in range(qindex1+1, self.n_qubits):
                Jij = np.abs(qindex2-qindex1)**(-a)
                Nalpha += Jij / (self.n_qubits-1)
        B = self.max_energy / bc # Magnetic field

        # generate hamiltonian
        hamiltonian = np.zeros( (self.dim, self.dim) )

        for qindex in range(self.n_qubits):
            if self.dynamic == DYNAMIC_FULL_RANDOM:
                coef = (np.random.rand()-0.5) * self.max_energy
            else:
                coef = B
            hamiltonian -= coef * self.Zop[qindex]

        for qindex1 in range(self.n_qubits):
            for qindex2 in range(qindex1+1, self.n_qubits):
                if self.dynamic == DYNAMIC_FULL_CONST_COEFF:
                    coef =  self.max_energy
                elif self.dynamic == DYNAMIC_ION_TRAP:
                    coef = np.abs(qindex2 - qindex1)**(-a) / Nalpha
                    coef = self.max_energy * coef
                else:
                    coef = (np.random.rand()-0.5) * self.max_energy
                hamiltonian -= coef * self.Xop[qindex1] @ self.Xop[qindex2]
                    
        ratio = float(self.tau) / float(self.virtual_nodes)

        # unitary operator        
        self.Uop = sp.linalg.expm(-1.j * hamiltonian * ratio)
        
        # initialize density matrix
        rho = np.zeros( [self.dim, self.dim] )
        rho[0, 0] = 1
        if qparams.init_rho != 0:
            # initialize random density matrix
            rho = random_density_matrix(self.dim)
        
        self.init_rho = rho.copy()
        self.last_rho = rho.copy()

    def init_reservoir(self, qparams, ranseed):
        self.__init_reservoir(qparams, ranseed)

    def __get_comput_nodes(self):
        N_obs = self.n_units
        if self.use_corr > 0:
            N_obs += int( (self.n_units * (self.n_units - 1)) / 2 )
        return N_obs * self.virtual_nodes
        
    
    def __reset_states(self):
        self.cur_states = [None]

    def gen_rand_rho(self, ranseed):
        if ranseed >= 0:
            np.random.seed(seed=ranseed)
        self.init_rho = random_density_matrix(self.dim)

    def step_forward(self, local_rho, input_state):

        Uop = self.Uop
        rho = local_rho
        
        # Replace the density matrix
        par_rho = partial_trace(rho, keep=[1], dims=[2**self.n_envs, 2**self.n_units], optimize=False)
        rho = np.kron(input_state, par_rho)

        current_state = []
        for v in range(self.virtual_nodes):
            # Time evolution of density matrix
            rho = Uop @ rho @ Uop.T.conj()
            for qindex in range(self.n_envs, self.n_qubits):
                expectation_value = np.real(np.trace(self.Zop[qindex] @ rho))
                current_state.append(expectation_value)
            
            if self.use_corr > 0:
                for q1 in range(self.n_envs, self.n_qubits):
                    for q2 in range(q1+1, self.n_qubits):
                        cindex = (q1, q2)
                        expectation_value = np.real(np.trace(self.Zop_corr[cindex] @ rho))
                        current_state.append(expectation_value)

        # Size of current_state is Nqubits x Nvirtuals)
        self.cur_states = np.array(current_state, dtype=np.float64)
        return rho

    def feed_forward(self, input_seq, predict, use_lastrho):
        input_length = input_seq.shape[0]
        predict_seq = None
        local_rho = self.init_rho.copy()
        if use_lastrho == True :
            local_rho = self.last_rho.copy()
        
        state_list = []
        for time_step in range(0, input_length):
            input_state = input_seq[time_step]
            local_rho = self.step_forward(local_rho, input_state)

            state = np.array(self.cur_states.copy(), dtype=np.float64)
            state_list.append(state.flatten())

        state_list = np.array(state_list)
        self.last_rho = local_rho.copy()

        if predict:
            stacked_state = np.hstack( [state_list, np.ones([input_length, 1])])
            predict_seq = stacked_state @ self.W_out
        
        return predict_seq, state_list


    def __train(self, input_seq, output_seq, buffer, beta, reservoir=True):
        assert(input_seq.shape[0] == output_seq.shape[0])
        
        if reservoir == True:
            _, state_list = self.feed_forward(input_seq, predict=False, use_lastrho=False)

            state_list = np.array(state_list)
            state_list = state_list[buffer:, :]

            # Discard the transitient state for training
            X = np.reshape(state_list, [-1, self.__get_comput_nodes()])
        else:
            X = convert_density_to_features(input_seq[buffer:, :])
        #print('shape', X.shape, state_list.shape)
        X = np.hstack( [X, np.ones([X.shape[0], 1]) ] )

        discard_output = output_seq[buffer:, :]
        # Create vector from density matrix
        Y = convert_density_to_features(discard_output)
        #print('shape X Y', X.shape, Y.shape)
        Nout = len(Y)
        self.W_out = np.random.rand(X.shape[1], Nout)

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

    def train_to_predict(self, input_seq, output_seq, buffer, qparams, ranseed, reservoir=True):
        self.__init_reservoir(qparams, ranseed)
        self.__train(input_seq, output_seq, buffer, qparams.beta, reservoir)

    def predict(self, input_seq, output_seq, buffer, use_lastrho, reservoir=True, postprocess=True):
        out_mats  = output_seq[buffer:, :]
        Nmats = out_mats.shape[0]
        state_list = None
        if reservoir == True:
            prediction_seq, state_list = self.feed_forward(input_seq, predict=True, use_lastrho=use_lastrho)
        else:
            X = convert_density_to_features(input_seq)
            X = np.hstack( [X, np.ones([X.shape[0], 1]) ] )
            prediction_seq = X @ self.W_out

        pred_vec = prediction_seq[buffer:, :]

        # Convert vector to density matrix
        pred_mats = convert_features_to_density(pred_vec, postprocess)

        # Calculate the fidelity
        fidls = []
        for n in range(Nmats):
            fidval = cal_fidelity_two_mats(out_mats[n], pred_mats[n])
            #print(n, fidval, out_mats[n].shape, pred_mats[n].shape)
            fidls.append(fidval)
        return pred_mats, fidls, state_list
    
    def init_forward(self, qparams, input_seq, init_rs, ranseed):
        self.__reset_states()
        if init_rs == True:
            self.__init_reservoir(qparams, ranseed)
        _, state_list =  self.feed_forward(input_seq, predict=False, use_lastrho=False)
        return state_list

def get_fidelity(qparams, buffer, train_input_seq, train_output_seq, \
    val_input_seq, val_output_seq, ranseed, use_corr, \
    reservoir=True, postprocess=True, test_lastrho=True, self_loop=False):
    model = QRC(use_corr)

    train_input_seq = np.array(train_input_seq)
    train_output_seq = np.array(train_output_seq)
    
    model.train_to_predict(train_input_seq, train_output_seq, buffer, qparams, ranseed, reservoir=reservoir)

    train_pred_seq, train_fidls, _ = model.predict(train_input_seq, train_output_seq, buffer=buffer, use_lastrho=False, reservoir=reservoir, postprocess=postprocess)
    #print("train_loss={}, shape".format(train_loss), train_pred_seq_ls.shape)
    
    val_pred_seq, val_fidls, state_list = [], [], []
    # Test phase
    if self_loop == True:
        length = len(val_output_seq)
        current_input = train_pred_seq[-1]
        for n in range(length):
            local_rho = model.last_rho.copy()
            model.last_rho = model.step_forward(local_rho, current_input)
            state = np.array(model.cur_states, dtype=np.float64)
            state_list.append(state)
            stacked_state = np.hstack( [state, np.ones([1, 1])])
            out_state = convert_vec_to_density(stacked_state @ model.W_out, postprocess=postprocess)
            fidval = cal_fidelity_two_mats(out_state, val_output_seq[n])
            val_pred_seq.append(out_state)
            val_fidls.append(fidval)
            current_input = out_state.copy()
    else:
        val_input_seq = np.array(val_input_seq)
        val_output_seq = np.array(val_output_seq)
        val_pred_seq, val_fidls, state_list = model.predict(val_input_seq, val_output_seq, buffer=0, use_lastrho=test_lastrho, reservoir=reservoir, postprocess=postprocess)
    #print("val_loss={}, shape".format(val_loss), val_pred_seq_ls.shape)

    return train_pred_seq, train_fidls, val_pred_seq, val_fidls, state_list


def memory_function(qparams, train_len, val_len, buffer, dlist, ranseed, Ntrials, usecorr):

    MFavgs, MFstds, train_avgs, val_avgs = [], [], [], []
    length = buffer + train_len + val_len
    n_envs = qparams.n_envs
    # generate data
    if qparams.n_envs == 1:
        data = generate_one_qubit_states(ranseed=ranseed, Nitems=length)
    else:
        data = generate_random_states(ranseed=ranseed, Nbase=n_envs, Nitems=length)
    
    idrho = np.zeros((2**n_envs, 2**n_envs))
    idrho[0, 0] = 1

    for d in dlist:
        train_input_seq = np.array(data[  : buffer + train_len])
        val_input_seq = np.array(data[buffer + train_len : length])
    
        train_out, val_out = [], []
        for k in range(length):
            yk = idrho
            if k >= d:
                yk = data[k-d]
            if k >= buffer + train_len:
                val_out.append(yk)
            else:
                train_out.append(yk)
        
        train_output_seq = np.array(train_out)
        val_output_seq = np.array(val_out)
        
        train_fid_ls, val_fid_ls, mfs = [], [], []
        for n in range(Ntrials):
            ranseed_net = ranseed
            if ranseed >= 0:
                ranseed_net = (ranseed + 10000) * (n + 1)
            # Use the same ranseed the same trial
            train_pred_seq, train_fidls, val_pred_seq, val_fidls, _ = \
                get_fidelity(qparams, buffer, train_input_seq, train_output_seq, val_input_seq, val_output_seq, ranseed_net, usecorr)
            
            train_rmean_square_fid = np.sqrt(np.mean(np.array(train_fidls)**2))
            val_rmean_square_fid = np.sqrt(np.mean(np.array(val_fidls)**2))
            #print('d={}, n={}, Fid train={} val={}'.format(\
            #   d, n, train_rmean_square_fid, val_rmean_square_fid))
            # Compute memory function
            # print('Val shape', val_pred_seq.shape)
            
            MF_d = square_distance_correlation(val_pred_seq, val_output_seq)
            #print('d={}, n={}, MF={}'.format(d, n, MF_d))

            train_fid_ls.append(train_rmean_square_fid)
            val_fid_ls.append(val_rmean_square_fid)
            mfs.append(MF_d)

        avg_train, avg_val, avg_MFd, std_MFd = np.mean(train_fid_ls), np.mean(val_fid_ls), np.mean(mfs), np.std(mfs)

        MFavgs.append(avg_MFd)
        MFstds.append(std_MFd)
        train_avgs.append(avg_train)
        val_avgs.append(avg_val)

    return np.array(list(zip(dlist, MFavgs, MFstds, train_avgs, val_avgs)))

def esp_states(qparams, length, ranseed, state_trials, use_corr):
    if ranseed >= 0:
        np.random.seed(seed=ranseed)

    n_envs = qparams.n_envs
    # generate data
    if n_envs == 1:
        data = generate_one_qubit_states(ranseed=ranseed, Nitems=length)
    else:
        data = generate_random_states(ranseed=ranseed, Nbase=n_envs, Nitems=length)
    input_seq = np.array(data)

    # Initialize the reservoir to zero state - density matrix
    model = QRC(use_corr)
    model.init_forward(qparams, input_seq, init_rs = True, ranseed = ranseed)
    rho1_init = model.init_rho.copy()
    rho1_last = model.last_rho.copy()

    # Compute the diff states
    dP = []
    for i in range(state_trials):
        # Initialzie the reservoir to a random initial state
        # Keep same coupling configuration
        for n in range(100):
            model.gen_rand_rho(ranseed = i + n*1000)
            rho2_init = model.init_rho.copy()
            dist_init = cal_distance_two_mats(rho1_init, rho2_init, distype='trace')
            if dist_init > 0.5:
                break
        model.init_forward(qparams, input_seq, init_rs = False, ranseed = i + 200000)
        rho2_last = model.last_rho.copy()
        
        dist_last = cal_distance_two_mats(rho1_last, rho2_last, distype='trace')
        #print('distlast', dist_last)
        dist_rate = dist_last / dist_init + 1e-10
        #dist_rate = dist_last
        dP.append(dist_rate)
    dP = np.array(dP)
    return dP