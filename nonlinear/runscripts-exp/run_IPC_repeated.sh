#!/usr/bin/bash
# Script to calculate memory capacity
export OMP_NUM_THREADS=1

EXE=../source/runIPC.py
BINPLOT=../postprocess/plot_IPC_thres.py
SAVE=../results/IPC_repeated_sed2

T=2000000
#T=2000
TS=_T_$T
WD=50
VAR=7
DEG=7
DELAYS='0,100,50,50,20,20,10,10'
#DELAYS='0,20,20,20,20,20,10,10'

V='1'
NSPINS=5
TMIN=0.0
TMAX=25.0
NTAUS=250
NPROC=125

THRES=0.0
DYNAMIC='ion_trap'
SEED=2
CAPA=4
WIDTH=0.1
P=seed_2_mdeg_7_mvar_7_thres_0.0_delays_$DELAYS$TS

for alpha in 0.2
do
for bc in 0.1 0.2 0.5 2.0 5.0 0.05
do
python $EXE --nproc $NPROC --spins $NSPINS --seed $SEED --dynamic $DYNAMIC --deg_delays $DELAYS --thres $THRES --tmin $TMIN --tmax $TMAX --ntaus $NTAUS --alpha $alpha --bcoef $bc --virtuals $V --length $T --max_deg $DEG --max_window $WD --max_num_var $VAR --savedir $SAVE

python $BINPLOT --dynamic $DYNAMIC --virtuals $V --nspins $NSPINS --folder $SAVE --posfix $P --thres 1e-4 --max_capa $CAPA --alpha $alpha --bcoef $bc  --tmin $TMIN --tmax $TMAX --ntaus $NTAUS --width $WIDTH
python $BINPLOT --dynamic $DYNAMIC --virtuals $V --nspins $NSPINS --folder $SAVE --posfix $P --thres 1e-5 --max_capa $CAPA --alpha $alpha --bcoef $bc  --tmin $TMIN --tmax $TMAX --ntaus $NTAUS --width $WIDTH
python $BINPLOT --dynamic $DYNAMIC --virtuals $V --nspins $NSPINS --folder $SAVE --posfix $P --thres 2e-5 --max_capa $CAPA --alpha $alpha --bcoef $bc  --tmin $TMIN --tmax $TMAX --ntaus $NTAUS --width $WIDTH
python $BINPLOT --dynamic $DYNAMIC --virtuals $V --nspins $NSPINS --folder $SAVE --posfix $P --thres 5e-5 --max_capa $CAPA --alpha $alpha --bcoef $bc  --tmin $TMIN --tmax $TMAX --ntaus $NTAUS --width $WIDTH
python $BINPLOT --dynamic $DYNAMIC --virtuals $V --nspins $NSPINS --folder $SAVE --posfix $P --thres 1e-6 --max_capa $CAPA --alpha $alpha --bcoef $bc  --tmin $TMIN --tmax $TMAX --ntaus $NTAUS --width $WIDTH
python $BINPLOT --dynamic $DYNAMIC --virtuals $V --nspins $NSPINS --folder $SAVE --posfix $P --thres 2e-6 --max_capa $CAPA --alpha $alpha --bcoef $bc  --tmin $TMIN --tmax $TMAX --ntaus $NTAUS --width $WIDTH
python $BINPLOT --dynamic $DYNAMIC --virtuals $V --nspins $NSPINS --folder $SAVE --posfix $P --thres 5e-6 --max_capa $CAPA --alpha $alpha --bcoef $bc  --tmin $TMIN --tmax $TMAX --ntaus $NTAUS --width $WIDTH
python $BINPLOT --dynamic $DYNAMIC --virtuals $V --nspins $NSPINS --folder $SAVE --posfix $P --thres 0.0  --max_capa $CAPA --alpha $alpha --bcoef $bc  --tmin $TMIN --tmax $TMAX --ntaus $NTAUS --width $WIDTH
done
done


