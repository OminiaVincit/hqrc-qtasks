# Quantum Reservoir Computing for Temporal Quantum Tomography

This repository is the official implementation of "Learning Temporal Quantum Tomography" paper. 

Learning Temporal Quantum Tomography
Quoc Hoan Tran and Kohei Nakajima
Phys. Rev. Lett. 127, 260401 – Published 22 December 2021
https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.127.260401

Press release:
Japanese: https://www.i.u-tokyo.ac.jp/news/press/2021/202112231947.shtml
(details): https://www.i.u-tokyo.ac.jp/news/files/IST_pressrelease_20211223_Nakajima.pdf
English: https://www.u-tokyo.ac.jp/focus/en/press/z0508_00200.html


## Code structures
(To be updated)

## Requirements
The code requires the following libraries:
- python 3.6-3.7
- numpy, matplotlib, sklearn

The packages can be installed as follows. First, we recommend to create a virtual environment in Python3 and use pyenv to manage python version:

```create virtual env
# Create virtual environment
pyenv install 3.7.7
python3 -m venv ~/vqrc 
source ~/vqrc/bin/activate
pyenv local 3.7.7
```

Install the following packages for the basic functions of our implementations:
- Calculate the quantum memory capacities of quantum reservoir (HQR)
- Perform Tomography Tasks

```
# For running scripts in the nonlinear folder
pip3 install numpy matplotlib sklearn 
```

## Demo
(To be updated)
