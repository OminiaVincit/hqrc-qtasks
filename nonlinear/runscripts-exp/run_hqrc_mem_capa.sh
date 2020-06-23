#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

BIN=../source/mc_hqrc.py
N=2
TASK=qrc_stm
SPAR=1.0
SAVE=../test_mc_spar_$SPAR
PROC=61
SOLVER='ridge_pinv'

TAUS=\'-3,-2,-1,0,1,2,3,4,5,6,7\' # The parameters for tau is 2**x for x in TAUS
STRENGTHS='0.0,0.1,0.5,0.9'
VS='1,5'
QRS='1'
Js='1.0'
Gs='0.125,0.5,2.0,8.0,32.0'

MIND=0
MAXD=240
INT=1

python $BIN --solver $SOLVER --sparsity $SPAR --couplings $Js --nondiags $Gs --taudeltas $TAUS --taskname $TASK --layers $QRS --strengths $STRENGTHS --virtuals $VS --ntrials $N --nproc $PROC --mind $MIND --maxd $MAXD --interval $INT --savedir $SAVE
