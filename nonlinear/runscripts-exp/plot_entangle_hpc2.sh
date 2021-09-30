#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

EXE=../postprocess/plot_delay_entangle.py
DYNAMIC=ion_trap

NPROC=125
NTRIALS=10
TAU=10.0

BUFFER=500
TRAINLEN=500
VALEN=200

NEV=1
DAT='Bell'
fvmin=0.5
fvmax=1.0
nvmin=0.0
nvmax=0.5
for DELAY in 10
do
for NSPINS in 6
do
for CORR in 1
do
for V in 10
do
for alpha in 1.0
do
for bc in 1.0
do
SAVE=../../../data/hqrc-qtasks/Bell_a_$alpha\_bc_$bc\_$NSPINS\_$NEV\_dl\_$DELAY

python $EXE --data $DAT --fvmin $fvmin --fvmax $fvmax --nvmin $nvmin --nvmax $nvmax --usecorr $CORR --alpha $alpha --bcoef $bc --rho 1 --delay $DELAY --ntrials $NTRIALS --folder $SAVE --spins $NSPINS --envs $NEV --nproc $NPROC --tauB $TAU --virtuals $V --dynamic $DYNAMIC --buffer $BUFFER --trainlen $TRAINLEN --vallen $VALEN 
done
done
done
done
done
done