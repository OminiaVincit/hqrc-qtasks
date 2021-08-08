
import numpy as np
import matplotlib.pyplot as plt

import plotutils as putils

fig, axs = plt.subplots(1, 1, figsize=(6, 12), squeeze=False)
axs = np.array(axs).ravel()
ax = axs[0]
#plt.style.use('seaborn-colorblind')
plt.rc('font', family='serif')
plt.rc('mathtext', fontset='cm')
plt.rcParams['font.size']=20

xs = [1, 2, 3, 4]


avg_qrc_ys, std_qrc_ys = dict(), dict()

avg_qrc_ys['N_m=4,K=4'] = np.array([0.990502, 0.952178, 0.919598, 0.911393])
std_qrc_ys['N_m=4,K=4'] = np.array([0.000451, 0.001100, 0.000436, 0.000700])

avg_qrc_ys['N_m=4,K=10'] = np.array([0.990577, 0.963902, 0.935524, 0.916531])
std_qrc_ys['N_m=4,K=10'] = np.array([0.000460, 0.000714, 0.000475, 0.000565])

avg_qrc_ys['N_m=5,K=5'] = np.array([0.990535, 0.959179, 0.924365, 0.913575])
std_qrc_ys['N_m=5,K=5'] = np.array([0.000453, 0.000664, 0.000371, 0.000596])

avg_qrc_ys['N_m=5,K=15'] = np.array([0.990500, 0.967026, 0.944348, 0.923069])
std_qrc_ys['N_m=5,K=15'] = np.array([0.000435, 0.000527, 0.000343, 0.000458])

avg_qrc_ys['Baseline'] = np.array([0.968466, 0.908038, 0.900101, 0.897647])
std_qrc_ys['Baseline'] = np.array([0.001054, 0.001275, 0.000698, 0.000756])


for ftype in ['png', 'svg']:
    transparent = True
    if ftype == 'png':
        transparent = False
    dcl = 0
    for label in avg_qrc_ys.keys():
        ys = 1.0-avg_qrc_ys[label]
        es = std_qrc_ys[label]
        
        if label == 'Baseline':
            color = 'gray'
            alpha = 0.9
        else:
            color = putils.cycle[dcl]
            alpha = 0.8
        ax.plot(xs, ys, label='${}$'.format(label), alpha = alpha, marker='s', markeredgecolor='k', \
                            linewidth=3, markersize=10, c=color)
        #ax.fill_between(xs, ys - es, ys + es, facecolor=color, alpha=0.2)
        dcl += 1
    ax.set_yscale('log')
    ax.legend()
    ax.set_xlabel('$N_e$', fontsize=24)
    ax.set_ylabel('Error', fontsize=24)
    ax.tick_params('both', length=10, width=1, which='both', labelsize=20)
    ax.grid(True, which="both", ls="-", color='0.65')
    plt.tight_layout()
    #ax.set_ylim([0.89, 1.0])
    plt.savefig('../results/figs/tau_10.0_a_1.0_bc_1.0_1000_3000_1000_error.{}'.format(ftype), bbox_inches='tight', transparent=transparent, dpi=600)
    