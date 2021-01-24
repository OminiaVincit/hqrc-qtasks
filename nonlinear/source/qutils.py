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

def generate_one_qubit_states(ranseed, Nitems):
    np.random.seed(seed=ranseed)

    I = np.array([[1, 0], [0, 1]])
    SigmaX = np.array([[0, 1], [1, 0]])
    SigmaY = np.array([[0, -1j], [1j, 0]])
    SigmaZ = np.array([[1, 0], [0, -1]])
    rhos = []
    for n in range(Nitems):
        r = 2.0 * (np.random.rand() - 0.5)
        x = np.random.rand()
        y = np.random.rand()
        s1 = r * x
        s2 = r * np.sqrt(1-x**2) * y
        s3 = r * np.sqrt(1-x**2) * np.sqrt(1 - y**2)
        rho = 0.5 * (I + s1 * SigmaX + s2 * SigmaY + s3 * SigmaZ)
        #rho = np.array(rand_dm(2, density=0.5))
        rhos.append(rho)
    return rhos

def generate_random_states(ranseed, Nbase, Nitems):
    rhos = []
    for n in range(Nitems):
        rho = np.array(rand_dm(2**Nbase, density=0.5))
        #print(n, rho)
        rhos.append(rho)
    return rhos

def convert_density_to_features(rho_ls):
    fevec = []
    for rho in rho_ls:
        local_vec = []
        local_vec.append(np.real(rho).ravel())
        local_vec.append(np.imag(rho).ravel())
        local_vec = np.array(local_vec).ravel()
        fevec.append(local_vec)
    fevec = np.array(fevec)
    return fevec

def convert_features_to_density(fevec):
    Nbase_sq = int(fevec.shape[1] / 2)
    Nbase    = int(np.sqrt(Nbase_sq))
    rho_ls = []
    for local_vec in fevec:
        real_rho = np.array(local_vec[:Nbase_sq]).reshape(Nbase, Nbase)
        imag_rho = np.array(local_vec[Nbase_sq:]).reshape(Nbase, Nbase)
        full_rho = real_rho + imag_rho * 1j

        rho_ls.append(full_rho)
    rho_ls = np.array(rho_ls)
    return rho_ls

def is_positive_semi(mat, tol=1e-9):
    E = np.linalg.eigvalsh(mat)
    return np.all(np.real(E) > -tol)

def is_hermitian(mat):
    return np.all(mat == mat.conj().T)

def is_trace_one(mat, tol=1e-9):
    err = np.abs(np.real(np.trace(mat)) - 1.0)
    return (err < tol)

def check_density(mat):
    return (is_hermitian(mat) and is_trace_one(mat) and is_positive_semi(mat))

def cal_fidelity_two_mats(matA, matB):
    if check_density(matA) == False or check_density(matB) == False:
        #print('Not density matrix')
        fidval = 0.0
    else:
        stateA = Qobj(matA)
        stateB = Qobj(matB)
        fidval = fidelity(stateA, stateB)
    return fidval

def cal_distance_two_mats(matA, matB, distype='angle'):
    fidval = cal_fidelity_two_mats(matA, matB)
    distance = np.arccos(fidval)
    return distance

def average_qstates(rho_ls):
    avg_rho = np.mean(rho_ls, axis=0)
    return avg_rho

def make_distance_mat(rhos, distype='angle'):
    N = rhos.shape[0]
    distmat = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            distmat[i, j] = cal_distance_two_mats(rhos[i], rhos[j])
    
    for i in range(N):
        for j in range(i):
            distmat[i, j] = distmat[j, i]
    return distmat

def distance_correlation(rhoAs, rhoBs, distype='angle'):
    assert(rhoAs.shape[0] == rhoBs.shape[0])
    Nsq = rhoAs.shape[0] ** 2
    distA = make_distance_mat(rhoAs)
    distB = make_distance_mat(rhoBs)
    A = distA - distA.mean(axis=0)[None, :] - distA.mean(axis=1)[:, None] + distA.mean()
    B = distB - distB.mean(axis=0)[None, :] - distB.mean(axis=1)[:, None] + distB.mean()
    dcov2_xy = (A * B).sum()/float(Nsq)
    dcov2_xx = (A * A).sum()/float(Nsq)
    dcov2_yy = (B * B).sum()/float(Nsq)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor