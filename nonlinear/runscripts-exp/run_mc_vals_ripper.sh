#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1
SAVE=/home/zoro/Workspace/data/hqrc-qtasks/quan_capa
EXE=../source/mc_quanrc_vals.py
DYNAMIC=ion_trap

NEV=2
BG=0
ED=10

TMIN=0.0
TMAX=10.0
NTAUS=100

MIND=0
MAXD=20
BUFFER=1000
TRAINLEN=3000
VALEN=100

vals=$(seq 0.02 0.02 2.2)
#vals=$(seq 0.05 0.05 2.0)
for NSPINS in 6 5 4 3
do
for V in 5
do
for als in 1.0
do
for bcls in 1.0 2.0
do
python $EXE --als $als --bcls $bcls --rho 0 --bgidx $BG --edidx $ED --savedir $SAVE --spins $NSPINS --envs $NEV --mind $MIND --maxd $MAXD --tmin $TMIN --tmax $TMAX --ntaus $NTAUS --virtuals $V --dynamic $DYNAMIC --buffer $BUFFER --trainlen $TRAINLEN --vallen $VALEN 
done
done
done
done

