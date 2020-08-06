#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

BIN=../source/esp_qrc.py
S=10
J=1.0
SAVE=../qesp_ion_trap2
DYN=ion_trap
BC=0.42

for T in 10 20 50 100 200 500 1000 2000 5000 10000
do
python $BIN --buffer $T --dynamic $DYN --nondiag $BC --coupling $J --strials $S --savedir $SAVE
done
