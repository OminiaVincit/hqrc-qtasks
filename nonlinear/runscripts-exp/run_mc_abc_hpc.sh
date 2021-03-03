#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1
SAVE=/data/zoro/qrep/quan_capa_abc
EXE=../source/mc_quanrc.py
DYNAMIC=ion_trap

NEV=2
NPROC=101
NTRIALS=5

TMIN=9.0
TMAX=10.0
NTAUS=1

MIND=0
MAXD=100
BUFFER=1000
TRAINLEN=3000
VALEN=100

vals=$(seq 0.02 0.02 2.2)
#vals=$(seq 0.05 0.05 2.0)
for NSPINS in 6 5 4 3 7
do
for V in 1
do
for alpha in 1.0
do
for bc in $vals
do
python $EXE --alpha $alpha --bcoef $bc --rho 1 --ntrials $NTRIALS --savedir $SAVE --spins $NSPINS --envs $NEV --nproc $NPROC --mind $MIND --maxd $MAXD --tmin $TMIN --tmax $TMAX --ntaus $NTAUS --virtuals $V --dynamic $DYNAMIC --buffer $BUFFER --trainlen $TRAINLEN --vallen $VALEN 
done
done
done
done

