#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

EXE=../source/runQTask.py
TASKNAME=delay-depolar
DYNAMIC=ion_trap

NPROC=125
NTRIALS=10

TMIN=0.0
TMAX=25.0
NTAUS=125

BUFFER=500
TRAINLEN=500
VALEN=500

NEV=1
ORDER=10
PLOT=0

for DELAY in 1
do
for NSPINS in 2 3 4 5 6
do
for V in 1 5
do
for alpha in 0.2 0.5 2.0
do
for bc in 0.1 0.2 0.5 1.0 2.0 5.0
do
SAVE=/data/zoro/qrep/delay_tasks/eig_a_$alpha\_bc_$bc\_$NSPINS\_$NEV\_od_$ORDER\_dl\_$DELAY\_$TASKNAME

python $EXE --alpha $alpha --bcoef $bc --plot $PLOT --rho 1 --order $ORDER --taskname $TASKNAME --delay $DELAY --ntrials $NTRIALS --savedir $SAVE --spins $NSPINS --envs $NEV --nproc $NPROC --tmin $TMIN --tmax $TMAX --ntaus $NTAUS --virtuals $V --dynamic $DYNAMIC --buffer $BUFFER --trainlen $TRAINLEN --vallen $VALEN 
done
done
done
done
done
