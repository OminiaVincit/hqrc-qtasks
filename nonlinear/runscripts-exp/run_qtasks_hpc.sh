#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

EXE=../source/runQTask.py
TASKNAME=delay-depolar
DYNAMIC=ion_trap

NPROC=100
NTRIALS=10

TMIN=0.0
TMAX=10.0
NTAUS=100

BUFFER=1000
TRAINLEN=3000
VALEN=1000

NEV=1
ORDER=10
PLOT=0
CORR=1

for DELAY in 1
do
for V in 1 5
do
for NSPINS in 6 5 4 3 2
do
for alpha in 1.0
do
for bc in 1.0
do
SAVE=/data/zoro/qrep/delay_tasks_corr/eig_a_$alpha\_bc_$bc\_$NSPINS\_$NEV\_od_$ORDER\_dl\_$DELAY\_$TASKNAME

python $EXE --usecorr $CORR --alpha $alpha --bcoef $bc --plot $PLOT --rho 1 --order $ORDER --taskname $TASKNAME --delay $DELAY --ntrials $NTRIALS --savedir $SAVE --spins $NSPINS --envs $NEV --nproc $NPROC --tmin $TMIN --tmax $TMAX --ntaus $NTAUS --virtuals $V --dynamic $DYNAMIC --buffer $BUFFER --trainlen $TRAINLEN --vallen $VALEN 
done
done
done
done
done
