#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

EXE=../source/runQTask.py
TASKNAME=delay-depolar
DYNAMIC=ion_trap

NPROC=125
NTRIALS=10

TMIN=0.0
TMAX=0.0
NTAUS=1
RESERVOIR=0
POSTPROCESS=1

BUFFER=1000
TRAINLEN=3000
VALEN=1000

ORDER=10
PLOT=0
NREP=1
NSPINS=5

for DELAY in 1
do
for NEV in 1 5 6
do
SAVE=/data/zoro/qrep/linear\_$NSPINS\_$NEV\_od_$ORDER\_dl\_$DELAY\_$TASKNAME

python $EXE --reservoir $RESERVOIR --postprocess $POSTPROCESS --nreps $NREP --plot $PLOT --rho 1 --order $ORDER --taskname $TASKNAME --delay $DELAY --ntrials $NTRIALS --savedir $SAVE --spins $NSPINS --envs $NEV --nproc $NPROC --tmin $TMIN --tmax $TMAX --ntaus $NTAUS --dynamic $DYNAMIC --buffer $BUFFER --trainlen $TRAINLEN --vallen $VALEN 
done
done
