#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

EXE=../source/mc_quanrc.py
SAVE=/home/zoro/Workspace/data/hqrc-qtasks/quan_capacity
DYNAMIC=ion_trap

NSPINS=5
NEV=1
NPROC=126
NTRIALS=10

TMIN=0.0
TMAX=25.0
NTAUS=25

MIND=0
MAXD=250
BUFFER=1000
TRAINLEN=3000
VALEN=1000

for V in 1
do
for alpha in 0.2
do
for bc in 1.0
do
python $EXE --savedir $SAVE --spins $NSPINS --envs $NEV --nproc $NPROC --mind $MIND --maxd $MAXD --tmin $TMIN --tmax $TMAX --ntaus $NTAUS --virtuals $V --dynamic $DYNAMIC --buffer $BUFFER --trainlen $TRAINLEN --vallen $VALEN 
done
done
done

