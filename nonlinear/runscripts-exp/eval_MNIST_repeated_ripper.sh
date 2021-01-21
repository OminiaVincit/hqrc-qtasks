#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

EXE=../source/mnist_eval.py
TAU_MNIST=/home/zoro/Workspace/data/hqrc-qtasks/mnist/taudata
MNIST=/home/zoro/Workspace/data/hqrc-qtasks/mnist
SIZE=10x10
DYNAMIC='full_random'
#DYNAMIC='ion_trap'

T=100 #buffer
NSPINS=5
NQRS=5
TMIN=0.0
TMAX=25.0
NTAUS=25
NPROC=126
TAU=22.0
RHO=0 #init rho or not
LB1=6
LB2=3
LINEAR=0
FULL=1

for V in 1
do
for alpha in 0.2
do
for bc in 1.0
do
python $EXE --nqrs $NQRS --full $FULL --virtual $V --tau $TAU --mnist_dir $MNIST --tau_mnist_dir $TAU_MNIST --mnist_size $SIZE --label1 $LB1 --label2 $LB2 --linear_reg $LINEAR --spins $NSPINS --dynamic $DYNAMIC 
done
done
done

