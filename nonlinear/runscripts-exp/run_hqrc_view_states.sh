#!/usr/bin/bash
# Script to view dynamics of the HQRC
export OMP_NUM_THREADS=1

BIN=../source/view_states.py
LENGTH=10000
BG=4000
ED=5000
SAVE=../rs_dynamics_spar_1.0
QR=1
ALPHA=0.0
PROC=100
CONST=0
BASE=qrc_varj
DYN=full_const_coeff
J=1.0

for g in 1.0
do
python $BIN --dynamic $DYN --basename $BASE --const $CONST --coupling $J --nondiag $g --savedir $SAVE --length $LENGTH --bg $BG --ed $ED --nqrc $QR --strength $ALPHA --nproc $PROC
done