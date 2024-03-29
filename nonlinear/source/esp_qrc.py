"""
    Calculate quantum echo state property for higher-order quantum reservoir
    See run_calculate_esp.sh for an example to run the script
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
import hqrc as hqrc
import utils
from utils import *
from loginit import get_module_logger

def esp_job(qparams, nqrc, layer_strength, buffer, length, state_trials, net_trials, send_end):
    print('Start process layer={}, taudelta={}, virtual={}, Jdelta={}, strength={}'.format(\
        nqrc, qparams.tau, qparams.virtual_nodes, qparams.max_energy, layer_strength))
    btime = int(time.time() * 1000.0)
    dPs = []
    for n in range(net_trials):
        dP = hqrc.esp_index(qparams, buffer, length, nqrc, layer_strength, \
            sparsity=1.0, sigma_input=1.0, ranseed=n, state_trials=state_trials)
        dPs.append(dP)

    mean_dp, std_dp = np.mean(dPs), np.std(dPs)
    
    rstr = '{} {} {} {} {} {} {}'.format(\
        nqrc, qparams.virtual_nodes, qparams.tau, qparams.max_energy, layer_strength, mean_dp, std_dp)
    etime = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))
    print('{} Finish process {} in {}s'.format(datestr, rstr, etime-btime))
    send_end.send(rstr)

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--units', type=int, default=5)
    parser.add_argument('--coupling', type=float, default=1.0)
    parser.add_argument('--rho', type=int, default=0)
    parser.add_argument('--beta', type=float, default=1e-14)
    
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha of coupled strength, 0 for random coupling')
    parser.add_argument('--bcoef', type=float, default=0.42, help='bcoeff nonlinear term (non-diagonal term)')
    parser.add_argument('--dynamic', type=str, default='ion_trap')

    parser.add_argument('--buffer', type=int, default=100)
    
    parser.add_argument('--ntrials', type=int, default=1)
    parser.add_argument('--strials', type=int, default=2)

    parser.add_argument('--layers', type=str, default='1')
    parser.add_argument('--strengths', type=str, default='0.0')
    parser.add_argument('--virtuals', type=str, default='1')

    parser.add_argument('--tmax', type=float, default=25.0, help='Maximum of tauB')
    parser.add_argument('--ntaus', type=int, default=125, help='Number of tausB')
    parser.add_argument('--nproc', type=int, default=50)

    parser.add_argument('--basename', type=str, default='qrc_echo')
    parser.add_argument('--savedir', type=str, default='echo_repeated')
    args = parser.parse_args()
    print(args)

    n_units, max_energy, beta, alpha, bcoef = args.units, args.coupling, args.beta, args.alpha, args.bcoef
    tmax, ntaus = args.tmax, args.ntaus

    dynamic = args.dynamic
    buffer = args.buffer
    length = buffer + 1000

    init_rho = args.rho
    net_trials, state_trials = args.ntrials, args.strials

    basename, savedir = args.basename, args.savedir
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    bindir = os.path.join(savedir, 'binary')
    if os.path.isdir(bindir) == False:
            os.mkdir(bindir)

    B = max_energy / bcoef
    taudeltas = list(np.linspace(0.0, tmax, ntaus + 1) / B)
    taudeltas = taudeltas[1:]

    virtuals = [int(x) for x in args.virtuals.split(',')]
    layers = [int(x) for x in args.layers.split(',')]
    strengths = [float(x) for x in args.strengths.split(',')]

    # Evaluation
    timestamp = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))

    outbase = os.path.join(bindir, '{}_{}_nspins_{}_a_{}_bc_{}_tmax_{}_ntaus_{}_J_{}_strength_{}_V_{}_layers_{}_T_{}_{}_esp_trials_{}_{}'.format(\
        basename, dynamic, n_units, alpha, bcoef, tmax, ntaus, max_energy,\
            '_'.join([str(s) for s in strengths]), \
            '_'.join([str(v) for v in virtuals]), \
            '_'.join([str(l) for l in layers]), \
            buffer, length, net_trials, state_trials))
    
    if os.path.isfile(savedir) == False:
        jobs, pipels = [], []
        for nqrc in layers:
            for layer_strength in strengths:
                for V in virtuals:
                    for tau_delta in taudeltas:
                        recv_end, send_end = multiprocessing.Pipe(False)
                        qparams = QRCParams(n_units=n_units, max_energy=max_energy, non_diag=bcoef, alpha=alpha,\
                            beta=beta, virtual_nodes=V, tau=tau_delta, init_rho=init_rho, dynamic=dynamic)
                        p = multiprocessing.Process(target=esp_job, \
                            args=(qparams, nqrc, layer_strength, buffer, length, net_trials, state_trials, send_end))
                        jobs.append(p)
                        pipels.append(recv_end)

        # Start the process
        for p in jobs:
            p.start()

        # Ensure all processes have finished execution
        for p in jobs:
            p.join()

        # Sleep 5s
        time.sleep(5)

        result_list = [np.array( [float(y) for y in x.recv().split(' ')]  ) for x in pipels]
        rsarr = np.array(result_list)
        # save the result
        np.savetxt('{}_esp.txt'.format(outbase), rsarr, delimiter=' ')

        # save experiments setting
        with open('{}_setting.txt'.format(outbase), 'w') as sfile:
            sfile.write('length={}, buffer={}\n'.format(length, buffer))
            sfile.write('n_units={}\n'.format(n_units))
            sfile.write('max_energy={},bcoef={},alpha={}\n'.format(max_energy, bcoef, alpha))
            sfile.write('beta={}\n'.format(beta))
            sfile.write('dynamic={}\n'.format(dynamic))
            sfile.write('layers={}\n'.format(' '.join([str(l) for l in layers])))
            sfile.write('Vs={}\n'.format(' '.join([str(v) for v in virtuals])))
            sfile.write('strengths={}\n'.format(' '.join([str(s) for s in strengths])))
            sfile.write('net_trials={}, state_trials={}\n'.format(net_trials, state_trials))
            sfile.write('tmax={}, ntaus={}\n'.format(tmax, ntaus))
