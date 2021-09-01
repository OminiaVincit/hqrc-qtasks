#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

EXE=../source/runQTask.py
#TASKNAME=delay-depolar
TASKNAME=denoise-dephase
DYNAMIC=ion_trap

NPROC=125
NTRIALS=1

TMIN=0.0
TMAX=25.0
NTAUS=125

BUFFER=500
TRAINLEN=500
VALEN=200

NEV=1
ORDER=10
PLOT=0
NREP=1
CORR=1
LASTRHO=1
DAT='GHZ'

for DELAY in 5
do
for NSPINS in 6
do
for V in 10
do
for alpha in 1.0
do
for bc in 1.0
do
SAVE=../../../data/hqrc-qtasks/draw_a_$alpha\_bc_$bc\_$NSPINS\_$NEV\_od_$ORDER\_dl\_$DELAY\_$TASKNAME

python $EXE --data $DAT --lastrho $LASTRHO --usecorr $CORR --nreps $NREP --alpha $alpha --bcoef $bc --plot $PLOT --rho 1 --order $ORDER --taskname $TASKNAME --delay $DELAY --ntrials $NTRIALS --savedir $SAVE --spins $NSPINS --envs $NEV --nproc $NPROC --tmin $TMIN --tmax $TMAX --ntaus $NTAUS --virtuals $V --dynamic $DYNAMIC --buffer $BUFFER --trainlen $TRAINLEN --vallen $VALEN 
done
done
done
done
done
