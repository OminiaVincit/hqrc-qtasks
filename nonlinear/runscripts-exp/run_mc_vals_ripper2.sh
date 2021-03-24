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
TMAX=25.0
NTAUS=125

BCMIN=0.0
BCMAX=2.0
NBCS=40

MIND=0
MAXD=10
BUFFER=1000
TRAINLEN=3000
VALEN=1000

vals=$(seq 0.1 0.1 2.0)
#vals=$(seq 0.05 0.05 2.0)
for NSPINS in 6
do
for V in '1'
do
for als in '0.2,0.5,1.0' '1.5,2.0,3.0' '0.1,0.3,0.4' '0.6,0.7,0.8' '0.9,1.1,1.2'
do
for tauls in '10.0'
do
python $EXE --als $als --tauls $tauls --rho 0 --bgidx $BG --edidx $ED --savedir $SAVE --spins $NSPINS --envs $NEV --mind $MIND --maxd $MAXD --bcmin $BCMIN --bcmax $BCMAX --nbcs $NBCS --virtuals $V --dynamic $DYNAMIC --buffer $BUFFER --trainlen $TRAINLEN --vallen $VALEN 
done
done
done
done
