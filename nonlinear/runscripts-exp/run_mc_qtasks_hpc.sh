#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

EXE=../source/mc_quanrc.py
SAVE=/data/zoro/qrep/quan_capacity
DYNAMIC=ion_trap

NEV=1
NPROC=101
NTRIALS=5

TMIN=0.0
TMAX=25.0
NTAUS=25

MIND=0
MAXD=100
BUFFER=1000
TRAINLEN=3000
VALEN=1000

for NSPINS in 6 4 3 2
do
for V in 1
do
for alpha in 1.0
do
for bc in 2.0
do
python $EXE --alpha $alpha --bcoef $bc --rho 1 --ntrials $NTRIALS --savedir $SAVE --spins $NSPINS --envs $NEV --nproc $NPROC --mind $MIND --maxd $MAXD --tmin $TMIN --tmax $TMAX --ntaus $NTAUS --virtuals $V --dynamic $DYNAMIC --buffer $BUFFER --trainlen $TRAINLEN --vallen $VALEN 
done
done
done
done

