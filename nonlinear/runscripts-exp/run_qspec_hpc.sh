#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

EXE=../source/superop_spectral_ion_qstate.py

NPROC=125
BG=0
ED=10

TMIN=0.0
TMAX=12.5
NTAUS=125


NEV=2
PLOT=0

for NSPINS in 6
do
for alpha in 1.0
do
for bc in 1.0
do
SAVE=/data/zoro/qrep/spectral2/eig_a_$alpha\_bc_$bc\_$NSPINS\_$NEV

python $EXE --alpha $alpha --bcoef $bc --plot $PLOT --bgidx $BG --edidx $ED --savedir $SAVE --nspins $NSPINS --nenvs $NEV --nproc $NPROC --tmin $TMIN --tmax $TMAX --ntaus $NTAUS
done
done
done
