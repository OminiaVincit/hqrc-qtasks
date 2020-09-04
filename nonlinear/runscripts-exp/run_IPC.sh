#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

EXE=../source/IPC_qrc.py
SAVE=../rs3_IPC_seed1

T=2000000
TS=_T_$T
WD=50
VAR=7
DEG=7
DELAYS='0,100,50,50,20,20,10,10'
#DELAYS='0,20,20,20,20,20,10,10'

V='5'
BG=-7.0
ED=7.1
INTV=0.2
THRES=0.0
DYNAMIC='full_random'
SEED=1
#python $EXE --seed $SEED --dynamic $DYNAMIC --deg_delays $DELAYS --thres $THRES --tbg $BG --ted $ED --interval $INTV --virtuals $V --length $T --max_deg $DEG --max_window $WD --max_num_var $VAR --savedir $SAVE


BIN=../postprocess/plot_IPC.py
P=V_5_qrc_IPC_full_random_seed_1_mdeg_7_mvar_7_thres_0.0_delays_$DELAYS$TS
#P=V_1_qrc_IPC_ion_trap_mdeg_7_mvar_7_thres_0.0_delays_$DELAYS$TS
#python $BIN --folder $SAVE --posfix $P

BINT=../postprocess/plot_IPC_thres.py
CAPA=20
python $BINT --folder $SAVE --posfix $P --thres 1e-4 --max_capa $CAPA
python $BINT --folder $SAVE --posfix $P --thres 1e-5 --max_capa $CAPA
python $BINT --folder $SAVE --posfix $P --thres 1e-6 --max_capa $CAPA
python $BINT --folder $SAVE --posfix $P --thres 0.0 --max_capa $CAPA
