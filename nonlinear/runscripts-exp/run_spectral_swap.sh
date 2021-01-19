#!/usr/bin/bash
# Script to view dynamics of the HQRC
export OMP_NUM_THREADS=1

BIN=../source/superop_spectral_ion.py
PLOTBIN=../postprocess/plot_spectral_mixu.py

SAVE=../results/spec_repeated

# For eigs
NSPINS=5
TMAX=25.0
NTAUS=250
NPROC=125

vals=$(seq 0.00 0.01 1.00)

# For plot
POSFIX=tmax_25.0_ntaus_250_eig_id_124_tot.binaryfile
PREFIX=spec_nspins_5

for alpha in 0.2 0.5 1.0 2.0
do
for bc in 0.5 1.0 2.0 5.0 0.2 0.1 0.05 0.42
do

# Run for eigs
for p in $vals
do
python $BIN --savedir $SAVE --nspins $NSPINS --tmax $TMAX --ntaus $NTAUS --nproc $NPROC --pstate $p --alpha $alpha --bcoef $bc
done

# Plot
python $PLOTBIN --folder $SAVE --prefix $PREFIX --posfix $POSFIX --alpha $alpha --bcoef $bc

done
done
