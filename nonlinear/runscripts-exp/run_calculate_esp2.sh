#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

BIN=../source/esp_qrc.py
PLOTBIN=../postprocess/plot_ESP_buffer.py
S=10
J=1.0
SAVE=../results/echo_repeated
DYN=ion_trap

# For eigs
NSPINS=5
TMAX=25.0
NTAUS=250

# For plot
POSFIX=esp_trials_1_10_esp
PREFIX=qrc_echo_ion_trap_nspins_$NSPINS


for alpha in 0.2 0.5 1.0 2.0
do
for bc in 0.1 0.2 0.5 1.0 2.0 5.0
do

for T in 10 20 50 100 200 500 1000 2000 5000 10000
do
python $BIN --alpha $alpha --bcoef $bc --units $NSPINS --tmax $TMAX --ntaus $NTAUS --buffer $T --dynamic $DYN --coupling $J --strials $S --savedir $SAVE
done

# plot
python $PLOTBIN --prefix $PREFIX --posfix $POSFIX --alpha $alpha --bcoef $bc --tmax $TMAX --ntaus $NTAUS --folder $SAVE

done
done