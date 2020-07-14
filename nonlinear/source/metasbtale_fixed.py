# Run export OMP_NUM_THREADS=1
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from loginit import get_module_logger
import multiprocessing
from collections import defaultdict
import os

def getSci(sc, i, Nspins):
    iop = identity(2)
    sci = iop
    if i == 0:
        sci = sc
    for j in range(1, Nspins):
        tmp = iop
        if j == i:
            tmp = sc
        sci = tensor(sci, tmp)
    return sci

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--nspins', type=int, default=5, help='Number of spins')
    parser.add_argument('--pstate', type=float, default=0, help='Mixed coefficient in the swap state, in [0, 1], -1 is random')
    parser.add_argument('--tau', type=float, default=10.0, help='Time between the input')
    parser.add_argument('--Tstep', type=int, default=1000, help='Number of time steps')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha of coupled strength')
    parser.add_argument('--bcoef', type=float, default=0.42)

    parser.add_argument('--savedir', type=str, default='metastable')
    parser.add_argument('--basename', type=str, default='meta')
    parser.add_argument('--plot', type=int, default=0, help='Flag to plot')
    args = parser.parse_args()
    print(args)

    Nspins, alpha, bc = args.nspins, args.alpha, args.bcoef
    tau, T, pstate = args.tau, args.Tstep, args.pstate
    basename = '{}_nspins_{}_state_{:.2f}_a_{}_bc_{}_tau_{}_T_{}'.format(args.basename, Nspins, pstate, alpha, bc, tau, T)
    savedir = args.savedir
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    if os.path.isfile(savedir) == False:
        logdir = os.path.join(savedir, 'log')
        if os.path.isdir(logdir) == False:
            os.mkdir(logdir)
        log_filename = os.path.join(logdir, '{}.log'.format(basename))
        logger = get_module_logger(__name__, log_filename)
        logger.info(log_filename)
        logger.info('Nspins={},pstate={},alpha={},bcoef={},tau={},Tsteps={}'.format(Nspins, pstate, alpha, bc, tau, T))

        # Create coupling strength
        J = 0
        for j in range(Nspins):
            for i in range(j+1, Nspins):
                Jij = np.abs(i-j)**(-alpha)
                J += Jij / (Nspins-1)
        B = J/bc # Magnetic field

        X = sigmax()
        Z = sigmaz()
        I = identity(2)

        # Create Hamiltonian
        H0 = getSci(I, 0, Nspins) * 0.0
        H1 = getSci(I, 0, Nspins) * 0.0
        for i in range(Nspins):
            Szi = getSci(Z, i, Nspins)
            H0 = H0 - B * Szi # Hamiltonian for the magnetic field
            for j in range(Nspins):
                if i != j:
                    Sxi = getSci(X, i, Nspins)
                    Sxj = getSci(X, j, Nspins)
                    hij = np.abs(i-j)**(-alpha) / J
                    H1 = H1 - hij * Sxi * Sxj # Interaction Hamiltonian
        H = H0 + H1 # Total Hamiltonian
        L = liouvillian(H, [])

        # swap with environment
        q0 = getSci(basis(2, 0), 0, Nspins)
        q1 = getSci(basis(2, 1), 0, Nspins)
        s0 = sprepost(q0, q0.dag())
        s1 = sprepost(q1, q1.dag())
        S = (tau*L).expm()
        # t1 = tensor_contract(S, (0, Nspins)) * s0
        # t2 = tensor_contract(S, (0, Nspins)) * s1
        
        s_prep = pstate * s0 + (1.0-pstate) * s1
        ts = tensor_contract(S, (0, Nspins)) * s_prep
        print(ts.shape, ts.iscp, ts.istp, ts.iscptp)
        ev = ts.eigenstates()[1]
        print(ev.shape)
        # Definite the initial state
        
        #for t in range(Tstep):
