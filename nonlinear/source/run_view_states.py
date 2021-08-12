#!/usr/bin/env python
"""
    Check reservoir states for varying parameter: tau, J/B
    See run_view_states.sh for an example to run the script
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
import pickle
import quanrc as qrc
import glob
from utils import *
from qutils import *
from qutip import *

colors = [
'#e41a1c',
'#377eb8',
'#4daf4a',
'#984ea3',
'#ff7f00',
'#ffff33'
]

def dynamics_job(qparams, length, bCs, tBs, bindir, ranseed):
    for bcoef in bCs:
        for tauB in tBs:
            qparams.tau = tauB * bcoef / qparams.max_energy
            qparams.non_diag = bcoef
            outfile = '{}_nspins_{}_{}_V_{}_a_{:.3f}_bc_{:.3f}_tauB_{:.3f}_seed_{}_len_{}.binary'.format(\
                qparams.dynamic, qparams.n_units + qparams.n_envs, qparams.n_envs, qparams.virtual_nodes, qparams.alpha, bcoef, tauB, ranseed, length)
            outfile = os.path.join(bindir, outfile)
            if os.path.isfile(outfile) == False:
                input_seq = generate_random_states(ranseed=ranseed, Nbase=qparams.n_envs, Nitems=length)
                input_seq = np.array(input_seq)
                model = qrc.QRC()
                state_list = model.init_forward(qparams, input_seq, init_rs=True, ranseed=ranseed)

                #dump statelist to file
                with open(outfile, 'wb') as wrs:
                    pickle.dump(state_list, wrs)
            
if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--spins', type=int, default=6, help='Number of the spins in the total system')
    parser.add_argument('--envs', type=int, default=2, help='Number of the spins in the environmental system')
    parser.add_argument('--rho', type=int, default=0, help='Flag for initializing the density matrix')
    parser.add_argument('--max_energy', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1e-14)

    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha of coupled strength, 0 for random coupling')
    parser.add_argument('--bcoef', type=float, default=1.0, help='bcoeff nonlinear term (non-diagonal term)')
    parser.add_argument('--tauB', type=float, default=10.0, help='tau in unitary exp(-i*tau*H)')
    parser.add_argument('--virtuals', type=int, default=1)

    parser.add_argument('--dynamic', type=str, default='ion_trap')
    parser.add_argument('--solver', type=str, default=RIDGE_PINV, \
        help='regression solver by linear_pinv,ridge_pinv,auto,svd,cholesky,lsqr,sparse_cg,sag')

    parser.add_argument('--length', type=int, default=1000)
    parser.add_argument('--nproc', type=int, default=125)
    parser.add_argument('--savedir', type=str, default='dynamics')

    parser.add_argument('--valmax', type=float, default=25, help='Maximum of val')
    parser.add_argument('--valmin', type=float, default=0, help='Minimum of val')
    parser.add_argument('--nvals', type=int, default=125, help='Number of val')

    parser.add_argument('--param_type', type=str, default='tauB', help='tauB,bc')
    parser.add_argument('--ranseed', type=int, default=0)

    parser.add_argument('--plot', type=int, default=0)
    parser.add_argument('--buffer', type=int, default=100)
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--plot_vals', type=str, default='0.1,0.25,0.5,1.0,2.5,4.0')

    args = parser.parse_args()
    print(args)

    n_spins, n_envs, max_energy, V, beta = args.spins, args.envs, args.max_energy, args.virtuals, args.beta
    alpha, bcoef, tauB, init_rho = args.alpha, args.bcoef, args.tauB, args.rho
    valmin, valmax, nvals = args.valmin, args.valmax, args.nvals

    dynamic, solver, param_type, ranseed = args.dynamic, args.solver, args.param_type, args.ranseed
    length, nproc, savedir = args.length, args.nproc, args.savedir
    
    os.makedirs(savedir, exist_ok=True)
    bindir = os.path.join(savedir, 'bins')
    figdir = os.path.join(savedir, 'figs')

    os.makedirs(bindir, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)

    vals = list(np.linspace(valmin, valmax, nvals + 1))
    vals = vals[1:]
    nproc = min(len(vals), args.nproc)
    lst = np.array_split(vals, nproc)

    bCs, tauBs = [bcoef], [tauB]
    
    if args.plot == 0:
        jobs, pipels = [], []
        for pid in range(nproc):
            sub_vals = lst[pid]
            if param_type == 'tauB':
                tauBs = sub_vals
            elif param_type == 'bc':
                bCs = sub_vals
            else:
                print('Not found param {}'.format(param_type))
                exit(1)

            qparams = QRCParams(n_units=n_spins-n_envs, n_envs=n_envs, max_energy=max_energy, non_diag=bcoef,alpha=alpha,\
                        beta=beta, virtual_nodes=V, tau=0.0, init_rho=init_rho, dynamic=dynamic)
            p = multiprocessing.Process(target=dynamics_job, args=(qparams, length, bCs, tauBs, bindir, ranseed))
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
    else:
        ntitle = ''
        bg = args.buffer
        ed = bg + args.T
        plot_vals = [float(x) for x in args.plot_vals.split(',')]
        plot_vals = ['{:.3f}'.format(x) for x in plot_vals]
        
        ys, xs = [], []
        val_ys = dict()
        for val in vals:
            statelist = []
            val1, val2 = bcoef, tauB
            if param_type == 'tauB':
                val2 = val
            elif param_type == 'bc':
                val1 = val
            else:
                print('Not found param {}'.format(param_type))
                exit(1)
            for rfile in glob.glob('{}/{}_nspins_{}_{}_V_{}_a_{:.3f}_bc_{:.3f}_tauB_{:.3f}*seed_{}_len_{}.binary'.format(\
                bindir, dynamic, n_spins, n_envs, V, alpha, val1, val2, ranseed, length)):
                print(rfile)
                with open(rfile, 'rb') as rrs:
                    z = pickle.load(rrs)
                    ntitle = os.path.basename(rfile)
                    statelist.append(z[bg:ed, :])
                    print(z.shape)
            if len(statelist) > 0:
                statelist = np.concatenate(statelist, axis=0)
            else:
                continue
                
            rs = statelist.ravel()
            ys.extend(rs)
            xs.extend([val] * len(rs))
            print(len(xs), len(ys))
            val_str = '{:.3f}'.format(val) 
            if val_str in plot_vals:
                val_ys[val_str] = statelist
        

        # Plot file
        if ntitle != '':
            ntitle = '{}_bg_{}_ed_{}'.format(ntitle, bg, ed)
            plt.rc('font', family='serif')
            plt.rc('mathtext', fontset='cm')
            plt.rcParams["font.size"] = 10 
            plt.rcParams['xtick.labelsize'] = 10 
            plt.rcParams['ytick.labelsize'] = 10

            fig = plt.figure(figsize=(16, 20), dpi=600)
            ncols = len(plot_vals)
            ax1 = plt.subplot2grid((3, ncols), (0,0), colspan=ncols, rowspan=1)
            
            ax1.scatter(xs, ys, s=(12*72./fig.dpi)**2, marker='o', lw=0, rasterized=True)
            ax1.set_yscale("symlog", base=10, linthresh=1e-5)
            ax1.set_title(ntitle)
            ax1.set_xticks(np.linspace(valmin, valmax, 21))
            ax1.set_xlim([valmin, valmax])
            ax1.grid(alpha=0.8, axis='x')
            ax1.set_xlabel(param_type)

            xs = list(range(bg, ed))
            for i in range(ncols):
                ax2 = plt.subplot2grid((3, ncols), (1,i), colspan=1, rowspan=2)
                statelist = val_ys[plot_vals[i]]
                for j in range(statelist.shape[1]):
                    ys = statelist[:, j]
                    ax2.plot(ys, xs, c=colors[j], label='q-{}'.format(j+n_envs+1), linewidth=1)
                    ax2.set_title('{}={}'.format(param_type, plot_vals[i]))
                    ax2.legend()
            

            for ftype in ['png']:
                figfile = os.path.join(figdir, '{}.{}'.format(ntitle, ftype))
                plt.savefig(figfile, bbox_inches='tight')
