#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

EXE=../source/mnist_make_reservoir.py
SAVE=/data/zoro/qrep/mnist/taudata
MNIST=/data/zoro/qrep/mnist
SIZE=28x28
DYNAMIC='ion_trap'

T=100 #buffer
TMIN=0.0
TMAX=14.0
NTAUS=14
NPROC=101
TAUS='22.0'

for NSPINS in 5
do
for V in 1
do
for alpha in 0.2
do
for bc in 1.0
do
python3 $EXE --taus $TAUS --savedir $SAVE --mnist_dir $MNIST --mnist_size $SIZE --nproc $NPROC --spins $NSPINS --dynamic $DYNAMIC --tmin $TMIN --tmax $TMAX --ntaus $NTAUS --alpha $alpha --bcoef $bc --virtual $V --buffer $T
done
done
done
done

