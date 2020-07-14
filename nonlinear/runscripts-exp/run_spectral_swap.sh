#!/usr/bin/bash
# Script to view dynamics of the HQRC
export OMP_NUM_THREADS=1

BIN=../source/superop_spectral.py
NSPINS=5
TMAX=50.0
NTAUS=501
NPROC=101

for p in 0.51 0.52 0.53 0.54 0.56 0.57 0.58 0.59
do
python $BIN --nspins $NSPINS --tmax $TMAX --ntaus $NTAUS --nproc $NPROC --pstate $p 
done
