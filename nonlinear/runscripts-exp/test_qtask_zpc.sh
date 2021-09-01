#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

EXE=../source/runQTask.py
TASKNAME=delay-depolar
DYNAMIC=ion_trap

NPROC=125
NTRIALS=10

TMIN=9.0
TMAX=10.0
NTAUS=1

BUFFER=1000
TRAINLEN=3000
VALEN=1000

ORDER=10
PLOT=0
NREP=1
CORR=1
NSPINS=5
for DELAY in 1
do
for NEV in 1 2 3 4 5 6
do
NSPINS=$((NSPINS+1))
for V in 5
do
for alpha in 1.0
do
for bc in 1.0
do
SAVE=../../../data/hqrc-qtasks/qubits_a_$alpha\_bc_$bc\_$NSPINS\_$NEV\_od_$ORDER\_dl\_$DELAY\_$TASKNAME

python $EXE --usecorr $CORR --nreps $NREP --alpha $alpha --bcoef $bc --plot $PLOT --rho 1 --order $ORDER --taskname $TASKNAME --delay $DELAY --ntrials $NTRIALS --savedir $SAVE --spins $NSPINS --envs $NEV --nproc $NPROC --tmin $TMIN --tmax $TMAX --ntaus $NTAUS --virtuals $V --dynamic $DYNAMIC --buffer $BUFFER --trainlen $TRAINLEN --vallen $VALEN 
done
done
done
done
done
