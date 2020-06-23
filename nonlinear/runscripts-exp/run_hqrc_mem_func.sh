#!/usr/bin/bash
# Script to calculate memory function
export OMP_NUM_THREADS=1

BIN=../source/mf_hqrc.py
N=10
J=1.0
TASK=qrc_stm
SPAR=1.0
SAVE=../test_mf_spar_$SPAR
QR=1
PROC=126
SOLVER='ridge_pinv'
#SOLVER='linear_pinv'

TAUS=\'-1,0,1,2,4\' # The parameters for tau is 2**x for x in TAUS
MIND=0
MAXD=250
INT=1

for a in 0.0 0.1 0.5 0.9
do
for V in 1 5
do
for g in 0.125 0.5 2.0 8.0 32.0
do
    python $BIN --solver $SOLVER --sparsity $SPAR --coupling $J --nondiag $g --taudeltas $TAUS --taskname $TASK --nqrc $QR --strength $a --virtuals $V --ntrials $N --nproc $PROC --mind $MIND --maxd $MAXD --interval $INT --savedir $SAVE
done
done
done
