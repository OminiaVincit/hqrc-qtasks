#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

BIN=../source/mc_qrc.py
N=10
TASK=qrc_stm
SAVE=../results/capa_repeated

SOLVER='ridge_pinv'
DYNAMIC='ion_trap'

VS='1'
NSPINS=5
TMIN=0.0
TMAX=25.0
NTAUS=250
NPROC=126

MIND=0
MAXD=250
INT=1

TRAIN=2000
VAL=1000
BUFF=10000

for alpha in 0.2
do
for bc in 1.0
do
python $BIN --tmin $TMIN --tmax $TMAX --ntaus $NTAUS --trainlen $TRAIN --vallen $VAL --buffer $BUFF  --solver $SOLVER --spins $NSPINS --taskname $TASK --virtuals $VS --ntrials $N --nproc $NPROC --mind $MIND --maxd $MAXD --interval $INT --savedir $SAVE
done
done