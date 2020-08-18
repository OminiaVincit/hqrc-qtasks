#!/bin/bash
# Script to emulate chaos by using hqrc
# 
    # --N_used 10000 \ # Number of time steps in training
    # --RDIM 1 \ # Dim of the input data
    # --noise_level 1 \ # Noise level added to the traning data
    # --scaler MinMaxZeroOne \
    # --nqrc 5 \ # Number of QRs
    # --alpha $ALPHA \ # Connection strength
    # --max_energy 2.0 \ # Max coupling energy
    # --virtual_nodes $V \ # Number of virtual nodes
    # --tau 4.0 \ # Interval between inputs
    # --n_units 6 \ # Number of hidden units =qubits in our setting
    # --reg $BETA \ # Ridge parameter
    # --dynamics_length 2000 \ # Transient time steps
    # --it_pred_length 1000 \ # Predicted length
    # --n_tests 2 \ # Number of tests
    # --solver pinv \ # Ridge by pseudo inverse
    # --augment 0 \ # Augment the hidden states
# End of the script to emulate chaos by using hqrc
# 
cd ../../../Methods
export OMP_NUM_THREADS=48

for V in 15
do
for ALPHA in 0.0
do
for BETA in 1e-7
do
for TAU in 32.0 16.0 8.0 2.0 1.0 0.25 0.125
do
python3 RUN.py hqrc \
    --mode all \
    --display_output 1 \
    --system_name Lorenz3D \
    --write_to_log 1 \
    --N 100000 \
    --N_used 10000 \
    --RDIM 1 \
    --noise_level 1 \
    --scaler MinMaxZeroOne \
    --nqrc 1 \
    --alpha $ALPHA \
    --max_energy 1.0 \
    --virtual_nodes $V \
    --tau $TAU \
    --n_units 6 \
    --reg $BETA \
    --dynamics_length 2000 \
    --it_pred_length 1000 \
    --n_tests 2 \
    --record_mag 1\
    --solver pinv \
    --augment 0
done
done
done
done