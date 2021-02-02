#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

EXE=../source/runQTask.py
TASKNAME=depolar
DYNAMIC=ion_trap

NSPINS=5
NEV=1

SAVE=/data/zoro/qrep/deleme_$TASKNAME

NPROC=125
NTRIALS=1

TMIN=0.0
TMAX=25.0
NTAUS=125

BUFFER=1000
TRAINLEN=500
VALEN=500

DELAY=2

for V in 1 5
do
for alpha in 0.2
do
for bc in 1.0
do
python $EXE --taskname $TASKNAME --delay $DELAY --ntrials $NTRIALS --savedir $SAVE --spins $NSPINS --envs $NEV --nproc $NPROC --tmin $TMIN --tmax $TMAX --ntaus $NTAUS --virtuals $V --dynamic $DYNAMIC --buffer $BUFFER --trainlen $TRAINLEN --vallen $VALEN 
done
done
done
