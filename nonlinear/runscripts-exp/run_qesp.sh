#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

EXE=../source/esp_quanrc_vals.py
SAVE=/data/zoro/qrep/esp_states

LEN=100
N=100
for LEN in 15 25 35 45 55 65 75 85 95
do
for NSPINS in 6
do
for NENVS in 2
do
python $EXE --savedir $SAVE --spins $NSPINS --envs $NENVS --length $LEN --ntrials $N
done
done
done
