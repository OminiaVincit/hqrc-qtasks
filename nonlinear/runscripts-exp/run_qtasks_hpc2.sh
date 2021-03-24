#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

EXE=../source/runQTask.py
TASKNAME=delay-depolar
DYNAMIC=ion_trap

NPROC=125
NTRIALS=10

TMIN=0.0
TMAX=12.5
NTAUS=125

BUFFER=1000
TRAINLEN=3000
VALEN=1000

NEV=2
ORDER=10
PLOT=0
CORR=0

avals=$(seq 0.5 0.5 3.0)
bcvals=$(seq 0.05 0.05 2.0)

for DELAY in 1
do
for V in 1 5
do
for NSPINS in 6
do
for alpha in $avals
do
for bc in $bcvals
do
SAVE=/data/zoro/qrep/delay_tasks3/eig_a_$alpha\_bc_$bc\_$NSPINS\_$NEV\_od_$ORDER\_dl\_$DELAY\_$TASKNAME

python $EXE --usecorr $CORR --alpha $alpha --bcoef $bc --plot $PLOT --rho 1 --order $ORDER --taskname $TASKNAME --delay $DELAY --ntrials $NTRIALS --savedir $SAVE --spins $NSPINS --envs $NEV --nproc $NPROC --tmin $TMIN --tmax $TMAX --ntaus $NTAUS --virtuals $V --dynamic $DYNAMIC --buffer $BUFFER --trainlen $TRAINLEN --vallen $VALEN 
done
done
done
done
done
