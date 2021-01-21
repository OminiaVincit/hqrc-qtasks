#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

EXE=../source/mnist_make_reservoir.py
SAVE=/home/zoro/Workspace/data/hqrc-qtasks/mnist/taudata
MNIST=/home/zoro/Workspace/data/hqrc-qtasks/mnist
SIZE=10x10
DYNAMIC='full_random'

T=100 #buffer
NSPINS=5
TMIN=0.0
TMAX=25.0
NTAUS=25
NPROC=126
TAUS='22.0'
RHO=0 #init rho or not
NQR=5

for V in 1
do
for alpha in 0.2
do
for bc in 1.0
do
python $EXE --nqrs $NQR --rho $RHO --taus $TAUS --savedir $SAVE --mnist_dir $MNIST --mnist_size $SIZE --nproc $NPROC --spins $NSPINS --dynamic $DYNAMIC --tmin $TMIN --tmax $TMAX --ntaus $NTAUS --alpha $alpha --bcoef $bc --virtual $V --buffer $T
done
done
done


