from qutip import *
import numpy as np
import scipy

def getNormCoef(Nspins, alpha):
    Nalpha = 0
    for i in range(Nspins):
        for j in range(i+1, Nspins):
            Jij = np.abs(i-j)**(-alpha)
            Nalpha += Jij / (Nspins-1)
    return Nalpha

def getSci(sc, i, Nspins):
    iop = identity(2)
    sci = iop
    if i == 0:
        sci = sc
    for j in range(1, Nspins):
        tmp = iop
        if j == i:
            tmp = sc
        sci = tensor(sci, tmp)
    return sci

def eigenstates(a):
    # Sort the abs of the eigenvalue from high to low
    w, vl, vr = scipy.linalg.eig(a, left=True, right=True)
    ids = np.argsort(-abs(w))
    return w[ids], vl[:, ids], vr[:, ids]

def getLiouv_IsingOpen(Nspins, alpha, B, nobs, J=1.0):
    # Create coupling strength
    Nalpha = getNormCoef(Nspins, alpha)

    X = sigmax()
    Z = sigmaz()
    I = identity(2)

    # Create Hamiltonian
    H0 = getSci(I, 0, Nspins) * 0.0
    H1 = getSci(I, 0, Nspins) * 0.0
    Sxs, Szs = [], []
    for i in range(Nspins):
        Sxi = getSci(X, i, Nspins)
        Szi = getSci(Z, i, Nspins)
        Sxs.append(Sxi)
        Szs.append(Szi)
    
    for i in range(Nspins):
        H0 = H0 - B * Szs[i] # Hamiltonian for the magnetic field
        for j in range(i+1, Nspins):
            hij = J * np.abs(i-j)**(-alpha) / Nalpha
            H1 = H1 - hij * Sxs[i] * Sxs[j] # Interaction Hamiltonian

    Mx = getSci(I, 0, nobs) * 0.0
    Mz = getSci(I, 0, nobs) * 0.0
    for i in range(nobs):
        Pxi = getSci(X, i, nobs)
        Pzi = getSci(Z, i, nobs)
        Mx += Pxi / nobs
        Mz += Pzi / nobs
        
    H = H0 + H1 # Total Hamiltonian
    L = liouvillian(H, [])
    return L, Mx, Mz
