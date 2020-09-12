#!/usr/bin/bash
# Script to view dynamics of the HQRC
export OMP_NUM_THREADS=1

BIN=../source/superop_spectral_random.py
NSPINS=5
TMAX=0.0
NTAUS=0
NPROC=125
SAVE=../spectral_random2
vals=$(seq 0.00 0.01 1.00)

for p in $vals
do
python $BIN --savedir $SAVE --nspins $NSPINS --tmax $TMAX --ntaus $NTAUS --nproc $NPROC --pstate $p 
done
