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

def generate_random_states(ranseed, Nbase, Nitems, distribution='uniform', add=None):
    rhos = []
    density_arrs = np.random.uniform(size=Nitems)
    D = 2**Nbase
    pertur_mat = np.eye(D) / D
    for n in range(Nitems):
        rho = np.array(rand_dm(D, density=density_arrs[n]))
        if add == 'sin':
            beta = np.sin(n)**2
            rho = beta * pertur_mat + (1.0 - beta) * rho
        #print(n, rho)
        rhos.append(rho)
    return rhos

def is_positive_semi(mat, tol=1e-10):
    E = np.linalg.eigvalsh(mat)
    return np.all(np.real(E) > -tol)

def is_hermitian(mat, tol=1e-10):
    return np.all(np.abs(mat - mat.conj().T) < tol)

def is_trace_one(mat, tol=1e-10):
    err = np.abs(np.abs(np.trace(mat)) - 1.0)
    return (err < tol)

def check_density(mat):
    return (is_hermitian(mat) and is_trace_one(mat) and is_positive_semi(mat))

def eps(z):
    """Equivalent to MATLAB eps
    """
    zre = np.real(z)
    zim = np.imag(z)
    return np.spacing(np.max([zre, zim]))

def nearest_psd(A, method):
    # % The nearest (in Frobenius norm) symmetric Positive Semi-Definite matrix to A
    # % Matrix A may be real or complex
    # %
    # % From Higham: "The nearest symmetric positive semidefinite matrix in the
    # % Frobenius norm to an arbitrary real matrix A is shown to be (B + H)/2,
    # % where H is the symmetric polar factor of B = (A + A')/2."
    # %
    # % See for proof of method SVD
    # % Higham NJ. Computing a nearest symmetric positive semidefinite matrix. 
    # % Linear algebra and its applications. 1988 May 1;103:103-18.
    # %  (http://www.sciencedirect.com/science/article/pii/0024379588902236)
    # %
    # % arguments: (input)
    # %  A - square matrix, which will be converted to the nearest Symmetric
    # %    Positive Definite Matrix.
    # %
    # %  method - 'svd' or eig', [Optional, default= 'svd']
    # %             'svd' is the method of Higham using the symmetric polar factor.
    # %             'eig' rectifies the eigvalues and recomposes the matrix.
    # %             While theorically equivalent, method 'svd' is more numerically stable
    # %             especially in cases of high co-linearity, and tends to returns an
    # %             Ahat slightly closer to A than method 'eig'. Therefore, while method 'eig' executes
    # %             faster, it is not recomended.
    # %
    # % Output:
    # %  Ahat - The matrix chosen as the nearest PSD matrix to A.

    # Fist check A is psd
    if is_positive_semi(A):
        return A
    B = (A + A.conj().T) / 2.0
    if method == 'eig':
        eigval, eigvec = np.linalg.eig(B)
        eigval[np.real(eigval) < 0] = 0
        Ahat = np.dot(eigvec, np.dot(np.diag(eigval), eigvec.conj().T))
    else:
        # SVD
        u, s, vh = np.linalg.svd(B, full_matrices=True)
        H = np.dot(vh.conj().T, np.dot(np.diag(s), vh)) # is PSD
        Ahat = (B + H) / 2.0
    
    # Make Ahat Hermitian
    Ahat = (Ahat + Ahat.conj().T) / 2.0

    # Test Ahat is PSD, if not, then modify just a bit
    psd = False
    k = 0
    tol = 1e-10
    while psd == False:
        E = np.linalg.eigvalsh(Ahat)
        #print(np.real(E))
        k += 1
        if np.all(np.real(E) > -tol):
            psd = True
        if psd == False:
            mineig = np.min(np.real(E))
            # adding a tiny multiple of an identity matrix.
            Ahat = Ahat + ( - (mineig*k)**2 + eps(mineig)) * np.eye(A.shape[0])
        if k > 10:
            print('May be it is a bug for taking too long time')
    Ahat = Ahat / np.trace(Ahat)
    return Ahat

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
        full_rho = nearest_psd(full_rho, method='eig')
        rho_ls.append(full_rho)
    rho_ls = np.array(rho_ls)
    return rho_ls

def cal_fidelity_two_mats(matA, matB):
    if check_density(matA) == False or check_density(matB) == False:
        # print('Not density matrix')
        fidval = 0.0
    else:
        stateA = Qobj(matA)
        stateB = Qobj(matB)
        fidval = fidelity(stateA, stateB)
    fidval = max(0.0, fidval)
    fidval = min(1.0, fidval)
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