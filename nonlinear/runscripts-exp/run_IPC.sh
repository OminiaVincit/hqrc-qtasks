#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

EXE=../source/IPC_qrc.py
SAVE=../thres1_IPC

T=20000
D=50
VAR=5
DEG=5
V='5'
BG=-7.0
ED=0.1
INTV=0.2
THRES=1e-4
C=2005
python $EXE --thres $THRES --tbg $BG --ted $ED --interval $INTV --virtuals $V --length $T --max_deg $DEG --chunk $C --max_delay $D --max_num_var $VAR --savedir $SAVE


BIN=../postprocess/plot_IPC.py
P=V_5_qrc_IPC_ion_trap_mdelay_50_mdeg_5_mvar_5_thres_0.0001_T_$T
python $BIN --folder $SAVE --posfix $P
