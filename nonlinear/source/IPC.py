import numpy as np
import scipy
from scipy.special import legendre
from scipy import sparse
from scipy.sparse import lil_matrix
import itertools
import argparse
import time
from datetime import timedelta
import sys

import os
from loginit import get_module_logger

class IPCParams:
    def __init__(self, max_delay, max_deg, max_num_var, thres, chunk):
        self.max_delay = max_delay
        self.max_deg = max_deg
        self.max_num_var = max_num_var
        self.thres = thres
        self.chunk = chunk

class IPC:
    def __init__(self, ipcparams, log, savedir=None):
        self.max_delay = ipcparams.max_delay
        self.max_deg = ipcparams.max_deg
        self.max_num_var = ipcparams.max_num_var
        self.thres = ipcparams.thres
        self.chunk = ipcparams.chunk
        
        self.log = log
        if savedir == None:
            tmpdir = os.path.join(os.path.dirname(__file__), 'results')
        else:
            tmpdir = savedir
        self.savedir = tmpdir
    
    def __prepare_signals(self, input_signals):
        start_time = time.monotonic()
        fname = sys._getframe().f_code.co_name

        self.log.info('{}: Prepare signals...'.format(fname))
        self.num_data = input_signals.shape[0]
        input_legendre_arr = []
        data = input_signals.reshape(-1)
        for d in range(self.max_deg + 1):
            input_legendre_arr.append( legendre(d)(data) )
        self.input_legendre_arr = np.array(input_legendre_arr)
        self.log.debug('{}: Created legendre input {}'.format(fname, self.input_legendre_arr.shape))

        delays = range(self.max_delay + 1)

        #target_cb = np.empty((self.max_delay + 1 + 1, 0), dtype='uint8')
        target_cb = lil_matrix((self.max_delay + 1, 0), dtype='uint8')
        for i in range(1, self.max_deg + 1):
            degs_delay = []
            
            #     IMPLEMENT 1: Naive implementation: consump of large memory
            #     delay_cb = list(itertools.combinations_with_replacement(delays, i))
            #     self.log.debug('{}: Making index deg i={}, num combines={}'.format(fname, i, len(delay_cb)))
            #     #degs_delay = np.zeros((delay_cb.shape[0], self.max_delay + 1 + 1))
            #     degs_delay = lil_matrix((len(delay_cb), self.max_delay + 1 + 1))
                
            #     for j in range(len(delay_cb)):
            #         for dindex in delay_cb[j]:
            #             degs_delay[j, dindex] += 1
            #     #degs_delay = degs_delay[np.count_nonzero(degs_delay, axis=1) <= self.max_num_var]
            #     self.log.debug('{}: deg i={}, Before filter degs_delay shape={}'.format(fname, i, degs_delay.shape))
            #     degs_delay = degs_delay[np.diff(degs_delay.tocsr().indptr) <= self.max_num_var]  # SLOW but can reduce memory
            
            # #    IMPLEMENT 2: efficient implementation (but general)
            # it_reps = itertools.combinations_with_replacement(delays, i)
            # for it in it_reps:
            #     if len(set(it)) > self.max_num_var:
            #         continue
            #     tmp_delay = np.zeros(self.max_delay + 1, dtype='uint8')
            #     for d in it:
            #         tmp_delay[d] += 1
            #     degs_delay.append(tmp_delay)

            #     IMPLEMENT 3: efficient implementation for the limitation of number of variables
            for nuniq in range(1, 1 + min(i, self.max_num_var)):
                it_dis = itertools.combinations(delays, nuniq)
                nrm = i - nuniq
                for it1 in it_dis:
                    if nrm > 0:
                        it_sub = itertools.combinations_with_replacement(it1, nrm)
                        its = [it1 + it2 for it2 in it_sub]
                    else:
                        its = [it1]
                    for it in its:
                        tmp_delay = np.zeros(self.max_delay + 1, dtype='uint8')
                        for d in it:
                            tmp_delay[d] += 1
                        degs_delay.append(tmp_delay)
            
            degs_delay = lil_matrix(degs_delay, dtype='uint8')
            self.log.debug('{}: deg i={}, After filter degs_delay shape={}'.format(fname, i, degs_delay.shape))

            #target_cb = np.concatenate([target_cb, degs_delay.T], axis=1)
            target_cb = sparse.hstack((target_cb, degs_delay.T)) # SLOW
            self.log.debug('{}: Making index deg i={}, shape={}'.format(fname, i, target_cb.shape))
            
        self.ipc_arr = lil_matrix(target_cb, dtype='uint8')
        self.ipc_rs = np.zeros(self.ipc_arr.shape[1])
        self.log.debug('{}: ipc_arr shape: {}'.format(fname, self.ipc_arr.shape))

        end_time = time.monotonic()
        self.log.info('{}: Executed time {}'.format(fname, timedelta(seconds=end_time - start_time)))
    
    def __calc_IPC(self, output_signals):
        start_time = time.monotonic()
        fname = sys._getframe().f_code.co_name

        self.log.info('{}: Calc ipc...'.format(fname))
        out = output_signals[self.max_delay:]
        avg = np.average(out, axis=0).reshape((-1, out.shape[1]))
        #std = np.std(out, axis=0).reshape((-1, out.shape[1]))
        #print(avg, avg.shape, std)
        #out = (out - avg) / std
        out -= avg
        N = out.shape[0]
        # calculate correlation
        inv_corr_mat_data = np.linalg.pinv((out.T@out)/float(N))
        #ot = out.T

        # Make target signals
        target_cb = self.ipc_arr.T
        Nc = target_cb.shape[0]

        broke = False
        for p in range(0, Nc, self.chunk):
            if broke:
                break
            pb = p
            pe = min(p + self.chunk, Nc)
            target_signals = []
        
            for i in range(pb, pe):
                lg_val = np.ones(self.num_data - self.max_delay)
                for delay in range(self.max_delay + 1):
                    tidx = int(target_cb[i, delay])
                    if tidx > 0:
                        bg = int(self.max_delay - delay)
                        ed = int(self.num_data - delay)
                        lg_val = np.multiply(lg_val, self.input_legendre_arr[ tidx, bg:ed ])
                        #lg_val *= self.input_legendre_arr[ tidx, bg:ed ]
                target_signals.append(lg_val) 
            target_signals = np.array(target_signals)
            self.log.debug('{}: Process {}/{} ({:.2f})%: Created target_signals shape: {}, ipc shape: {}'.format(\
                fname, p, Nc, (p/Nc) * 100, target_signals.shape, self.ipc_arr.shape))
        
            corr_mat_data_target = target_signals@out/float(N)
            self.log.debug('{}: out={}, inv={}, corr={}'.format(fname, out.shape, inv_corr_mat_data.shape, corr_mat_data_target.shape))
            variance = np.var(target_signals, axis=1)
            #savg = np.average(self.target_signals**2, axis=1)
            if p == 0:
                variance[0] = 1

            # ipc_arr.shape = (max_delay + 1 + 1, num_combins)
            for k in range(corr_mat_data_target.shape[0]):
                tmp = corr_mat_data_target[k]@inv_corr_mat_data@corr_mat_data_target[k].T/variance[k]

                # tmp = 0
                # z = self.target_signals[k]
                # for i in range(ot.shape[0]):
                #     zxi = np.average(z*ot[i])
                #     for j in range(ot.shape[0]):
                #         zxj = np.average(z*ot[j])
                #         xij = inv_corr_mat_data[i, j]
                #         tmp += zxi * xij * zxj
                # tmp = tmp/variance[k]
                
                if tmp < self.thres:
                    broke = True
                    break
                # the last row store ipc component
                self.ipc_rs[k + pb] = tmp
        end_time = time.monotonic()
        self.log.info('{}: Executed time {}'.format(fname, timedelta(seconds=end_time - start_time)))

    def write_results(self, posfix='capa', writedelay=False):
        start_time = time.monotonic()
        fname = sys._getframe().f_code.co_name

        # write IPC by degree and delay
        ipc_by_delay = np.zeros((2, self.max_delay + 1))
        ipc_by_delay[0] = np.arange(0, self.max_delay + 1)

        ipc_by_deg = np.zeros((2, self.max_deg + 1))
        ipc_by_deg[0] = np.arange(0, self.max_deg + 1)

        (ndelay, ncb) = self.ipc_arr.shape
        ipc = self.ipc_rs
        delays_arr = self.ipc_arr.tocsc()

        if writedelay:
            # extract ipc by delay
            # THIS IS SLOW (but equal to np.count_nonzero in each column )
            # or tocsr().indptr to count nonzero elements in each row
            ndeg = np.diff(delays_arr.indptr)
            self.log.debug('{}: Making delays arr'.format(fname))                
            for delay in range(self.max_delay + 1):
                tmp = 0
                for i in range(ncb):
                    #ndeg = np.count_nonzero(delays_arr[:, i])
                    if delays_arr[delay, i] > 0:
                        tmp += ipc[i] / ndeg[i]
                ipc_by_delay[1, delay] = tmp

        # extract ipc by degree
        self.log.debug('{}: Making degree arr'.format(fname))    
        for i in range(ncb):
            sdeg = int(np.sum(delays_arr[:, i]))
            if sdeg >= 1 and sdeg < self.max_deg + 1:
                ipc_by_deg[1, sdeg] += ipc[i]
        self.log.debug('{}: Writing to files'.format(fname))  
        
        if os.path.isdir(self.savedir) == False:
            os.mkdir(self.savedir)
        
        if writedelay:
            np.savetxt(os.path.join(self.savedir, 'delay_{}.txt'.format(posfix)), ipc_by_delay.T)

        np.savetxt(os.path.join(self.savedir, 'degree_{}.txt'.format(posfix)), ipc_by_deg.T)
        sparse.save_npz(os.path.join(self.savedir, 'ipc_arr_{}.txt'.format(posfix)), delays_arr)
        np.savetxt(os.path.join(self.savedir, 'ipc_{}.txt'.format(posfix)), ipc)
        
        end_time = time.monotonic()
        self.log.info('{}: Finish write to results. Executed time {}'.format(fname, timedelta(seconds=end_time - start_time)))

    def run(self, input_signals, output_signals):
        self.__prepare_signals(input_signals)
        self.__calc_IPC(output_signals)
