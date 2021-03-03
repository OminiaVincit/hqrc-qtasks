#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

EXE=../source/superop_spectral_ion_qstate_varbc.py

NPROC=125
BG=0
ED=10

BCMIN=0.0
BCMAX=2.5
NBCS=125

NEV=2
PLOT=0

for NSPINS in 6
do
for alpha in 1.0
do
for tauB in 10.0
do
SAVE=/home/zoro/Workspace/data/hqrc-qtasks/spectral/eig_a_$alpha\_tauB_$tauB\_$NSPINS\_$NEV

python $EXE --alpha $alpha --tauB $tauB --plot $PLOT --bgidx $BG --edidx $ED --savedir $SAVE --nspins $NSPINS --nenvs $NEV --nproc $NPROC --bcmin $BCMIN --bcmax $BCMAX --nbcs $NBCS
done
done
done
