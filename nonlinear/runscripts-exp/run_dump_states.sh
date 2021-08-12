#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1
SAVE=/data/zoro/qrep/dynamics
EXE=../source/run_view_states.py
DYNAMIC=ion_trap

NSPINS=6
NEV=2

# For tau
PARTYPE='tauB'
VALMIN=0.0
VALMAX=5.0
NVALS=500
PLOTVALS='0.1,0.25,0.5,1.0,2.5,4.0'

# # For bcoef
# PARTYPE='bc'
# VALMIN=0.0
# VALMAX=2.0
# NVALS=500
# PLOTVALS='0.02,0.1,0.2,0.5,1.0,1.5'

LENGTH=10000
BUFFER=1000

PLOT=1
SEED=0
for BUFFER in 100 500 1000 2000 5000 9000
#for BUFFER in 1000
do
python $EXE --plot_vals $PLOTVALS --ranseed $SEED --buffer $BUFFER --plot $PLOT --length $LENGTH --savedir $SAVE --dynamic $DYNAMIC --spins $NSPINS --envs $NEV --param_type $PARTYPE --valmin $VALMIN --valmax $VALMAX --nvals $NVALS
done
