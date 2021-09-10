from qutip import *
import numpy as np
import scipy

def make_adj_matrix(Nbase):
    A = np.zeros((Nbase, Nbase), dtype=np.int32)
    for i in range(Nbase):
        for j in range(i+1, Nbase):
            A[i, j] = np.random.randint(0, 2)
            A[j, i] = A[i, j]
    return A

def minmax_norm(arr):
    minval = np.min(arr)
    maxval = np.max(arr)
    arr = arr - minval
    if maxval > minval:
        arr = arr / (maxval - minval)
    return arr

#----------------------------------------------------------------------
# Generate the output data from tensor product of two past input data
#----------------------------------------------------------------------
def generate_delay_tensor(n_envs, delay1, delay2, length, ranseed, dat_label):
    n_out_qubits = 2*n_envs
    Dout = 2**n_out_qubits
    idrho = np.zeros((Dout, Dout)).astype(complex)
    idrho[0, 0] = 1
    output_data = [idrho] * length

    max_delay = max(delay1, delay2)
    input_data = generate_random_states(ranseed=ranseed, Nbase=n_envs, Nitems=length, add=dat_label)
    for n in range(max_delay, length):
        output_data[n] = np.kron(input_data[n-delay1], input_data[n-delay2])
    return input_data, output_data

def generate_qtasks_delay(n_envs, ranseed, length, delay, taskname, \
    order, Nreps=1, buffer_train=0, dat_label=None, noise_level=0.3):
    #input_data = generate_one_qubit_states(ranseed=ranseed, Nitems=length)
    np.random.seed(seed=ranseed + 1987)
    D = 2**n_envs
    # Returns a superoperator acting on vectorized dim × dim density operators, 
    # sampled from the BCSZ distribution.
    sup_ops = []
    
    for d in range(delay+1):
        sop = rand_super_bcsz(N=D, enforce_tp=True)
        #sop = rand_unitary(N=2**n_envs)
        sup_ops.append(sop)
    
    # generate coefficients
    coeffs = np.random.rand(delay + 1) 
    #coeffs[-1] = 1.0
    if 'delay' in taskname:
        coeffs = np.zeros(delay+1)
        coeffs[-1] = 1.0
    elif 'wma' in taskname: # weighted moving average filter
        coeffs = np.flip(np.arange(1, delay+2))
    elif 'sma' in taskname:
        coeffs = np.ones(delay+1)
    #print('Coeffs', coeffs)
    coeffs = coeffs / np.sum(coeffs)
    coeffs = coeffs.astype(complex)
    Nitems = int(length / Nreps)
    if n_envs == 1:
        if dat_label != 'rand':
            input_data = generate_random_states(ranseed=ranseed, Nbase=n_envs, Nitems=length, add=dat_label)
        else:
            input_data = generate_one_qubit_states(ranseed=ranseed, Nitems=Nitems, Nreps=Nreps)
    else:
        input_data = generate_random_states(ranseed=ranseed, Nbase=n_envs, Nitems=length, add=dat_label)
    #input_data = generate_random_states(ranseed=ranseed, Nbase=n_envs, Nitems=length)
    
    n_out_qubits = n_envs
    if taskname == 'delay-entangle':
        n_out_qubits = 2*n_envs
        
    Dout = 2**n_out_qubits
    idrho = np.zeros((Dout, Dout)).astype(complex)
    idrho[0, 0] = 1
    output_data = [idrho] * length
    if taskname == 'sma-rand' or taskname == 'wma-rand':
        print('Task {}'.format(taskname))
        for n in range(delay, length):
            outstate = None
            for d in range(delay+1):
                mstate = Qobj(input_data[n - d])
                mstate = operator_to_vector(mstate)
                mstate = sup_ops[d] * mstate
                if outstate is not None:
                    outstate += coeffs[d] * mstate
                else:
                    outstate = coeffs[d] * mstate
            #print('Shape outstate', is_trace_one(outstate), is_hermitian(outstate), is_positive_semi(outstate))
            output_data[n] = np.array(vector_to_operator(outstate))
    elif taskname == 'sma-depolar' or taskname == 'wma-depolar' or taskname == 'delay-depolar':
        print('Make NARMA data for depolar task {}'.format(taskname))
        _, data_noise = make_data_for_narma(length=length, orders=[order])
        pnoise = data_noise[:, 0].ravel()
        idmat = np.eye(D)
        for n in range(delay, length):
            outstate = None
            for d in range(delay+1):
                mstate = pnoise[n-d] * idmat / D + (1.0 - pnoise[n-d]) * input_data[n - d]
                if outstate is not None:
                    outstate += coeffs[d] * mstate
                else:
                    outstate = coeffs[d] * mstate
                output_data[n] = outstate
    elif (taskname == 'sma-dephase' or taskname == 'wma-dephase' or taskname == 'delay-dephase') and n_envs == 1:
        print('Make NARMA data for Z-dephase (phase-flip) task {}'.format(taskname))
        Z = [[1,0],[0,-1]]
        _, data_noise = make_data_for_narma(length=length, orders=[order])
        pnoise = data_noise[:, 0].ravel()
        idmat = np.eye(D)
        for n in range(delay, length):
            outstate = None
            for d in range(delay+1):
                rho = input_data[n - d] @ Z
                rho = Z @ rho
                mstate = pnoise[n-d] * rho + (1.0 - pnoise[n-d]) * input_data[n - d]
                if outstate is not None:
                    outstate += coeffs[d] * mstate
                else:
                    outstate = coeffs[d] * mstate
                output_data[n] = outstate
    elif taskname == 'delay-id':
        print('Task {}'.format(taskname))
        output_data[delay:] = input_data[:(length-delay)]
    elif taskname == 'delay-rand':
        for n in range(delay, length):
            mstate = Qobj(input_data[n - delay])
            mstate = operator_to_vector(mstate)
            mstate = sup_ops[delay] * mstate
            output_data[n] = np.array(vector_to_operator(mstate))
    elif taskname == 'delay-entangle':
        Uop = get_transverse_unitary(Nspins=2*n_envs, B=2, dt=10.0)
        for n in range(delay, length):
            rho = np.kron(input_data[n], input_data[n - delay])
            rho = Uop @ rho @ Uop.T.conj()
            output_data[n] = rho
    elif 'noise' in taskname:
        output_data[delay:] = input_data[:(length-delay)].copy()
        _, noise1 = make_data_for_narma(length=length, orders=[order])
        _, noise2 = make_data_for_narma(length=length, orders=[20])
        #print(np.min(noise1), np.min(noise2), np.max(noise1), np.max(noise2))
        pnoise1 = noise1.ravel()
        pnoise2 = noise2.ravel()
        pnoise1 = minmax_norm(pnoise1) * noise_level
        pnoise2 = minmax_norm(pnoise2) * noise_level
        # Note that 'Spin-flip probability per qubit greater than 0.5 is unphysical.'

        idmat = np.eye(D)
        Z = [[1,0],[0,-1]]
        # Add noise1 to input_data and noise2 to output data in the training part
        if buffer_train == 0:
            buffer_train = length

        for n in range(length):
            if taskname == 'denoise-depolar':
                #input_data[n] = pnoise1[n] * idmat / D + (1.0 - pnoise1[n]) * input_data[n]
                input_data[n] = depolar_channel(input_data[n], Nspins=n_envs, p=pnoise1[n])
            elif taskname == 'denoise-dephase-per':
                input_data[n] = Pauli_channel(input_data[n], Nspins=n_envs, p=pnoise1[n], Pauli='Z')
            elif taskname == 'denoise-dephase-col':
                input_data[n] = Pauli_collective(input_data[n], Nspins=n_envs, p=pnoise1[n], Pauli='Z')
            elif taskname == 'denoise-flip':
                input_data[n] = Pauli_channel(input_data[n], Nspins=n_envs, p=pnoise1[n], Pauli='X')
            elif taskname == 'denoise-unitary':
                input_data[n] = unitary_noise(input_data[n], t = pnoise1[n], num_gates=20, ranseed=ranseed+2022)
            #output_data[n] = pnoise1[n] * idmat / D + (1.0 - pnoise1[n]) * output_data[n]
        if False:
            for n in range(buffer_train):
                if taskname == 'denoise-depolar':
                    #input_data[n] = pnoise1[n] * idmat / D + (1.0 - pnoise2[n]) * input_data[n]
                    output_data[n] = depolar_channel(output_data[n], Nspins=n_envs, p=pnoise2[n])
                elif taskname == 'denoise-dephase-per':
                    output_data[n] = Pauli_channel(output_data[n], Nspins=n_envs, p=pnoise2[n], Pauli='Z')
                    #output_data[n] = unitary_noise(output_data[n], t = pnoise2[n], num_gates=20, ranseed=ranseed+2022)
                elif taskname == 'denoise-dephase-col':
                    output_data[n] = Pauli_collective(output_data[n], Nspins=n_envs, p=pnoise2[n], Pauli='Z')
                elif taskname == 'denoise-flip':
                    output_data[n] = Pauli_channel(output_data[n], Nspins=n_envs, p=pnoise2[n], Pauli='X')
                    #output_data[n] = unitary_noise(output_data[n], t = pnoise2[n], num_gates=20, ranseed=ranseed+2022)
                elif taskname == 'denoise-unitary':
                    output_data[n] = unitary_noise(output_data[n], t = pnoise2[n], num_gates=20, ranseed=ranseed+2022)
    else:
        output_data = input_data
    
    return input_data, output_data

def psi_cluster_state(A):
    # Create cluster state from adjacency matrix
    # psi: m-qubit cluster state with adjacency matrix A \in {0,1}^(m x m),
    # sum_{a \in {0,1}^m} (-1)^(a^T*A*a/2) |a>,
    # 0 in a corresponds to [1,0], 1 to [0,1], 
    # joint eigenstates of K_j with eigenvalue 1,
    # K_j: sigma_x on jth qubit, sigma_z on neighbors, in the tensor product basis
    Nbase = A.shape[0] # number of qubits
    D = 2**Nbase

    # binary basis (rows)
    # 0 ~ [1,0], 1 ~ [0,1] 
    # ordered according to tensor product basis
    bin_rep = []
    for d in range(D):
        bin_rep.extend(list(np.binary_repr(d, width = Nbase)))
    bin_rep = [int(x) for x in bin_rep]
    bin_rep = np.array(bin_rep).reshape((D, Nbase))
    sgn = np.diag(bin_rep @ A @ bin_rep.T) / 2
    psi = (-1)**sgn
    return psi/np.sqrt(D)

def convert_seq(input_seq):
    ma = np.real(input_seq).reshape((input_seq.shape[0], -1))
    mb = np.imag(input_seq).reshape((input_seq.shape[0], -1))
    return np.hstack((ma, mb)).transpose()    

def Brownian_circuit(dim, n, dt):
    # BrownianCircuit(dim,n,dt) creates a random unitary matrix by multiplying
    # exp(i H_1 dt) exp(i H_2 dt) ... exp(i H_n dt)
    # for random normal distributed hermitian dim-dimensional matricies
    #  \{ H_i \}_{i=1}^{n} standard deviation 1/2.
    U = np.eye(dim)
    for i in range(n):
        Re = np.random.normal(size = (dim, dim))
        Im = 1.0j*np.random.normal(size = (dim, dim))
        C = Re + Im
        H = (C + C.T.conj())/4
        U = U @ scipy.linalg.expm(1.j * H * dt)
    return U

def random_unitary1(dim, t, n):
    # Creates a dim-dimensional unitary close to identiry.
    # For t=0 the answer is identity
    # (for practical purposes, t~1) 
    # the answer is a unitary sampled from Haar distribution.
    # The error to Texp(\int_0^t i \pi H(t) dt ) with random H(t) scales as 1/sqrt(n).

    U = Brownian_circuit(dim, n, np.sqrt(1/(n*dim))*2*np.pi*t)
    return U

def unitary_noise(rho, t, num_gates, ranseed=0):
    # rho: noiseless state.
    # t: noise strength, "time" during which noise acts, around t=1 the noise \
    # is so strong the distribution is essentially uniform.
    # num_gates: number of noisy gates for approximation
    np.random.seed(seed=ranseed)
    
    dim = int(rho.shape[0])
    Uop = random_unitary1(dim, t, num_gates)
    rs_rho = Uop @ rho @ Uop.T.conj()
    return rs_rho

def fidelity_two_seqs(in_mats, out_mats):
    # Calculate the fidelity
    fidls = []
    Nmats = in_mats.shape[0]
    for n in range(Nmats):
        fidval = cal_fidelity_two_mats(in_mats[n], out_mats[n])
        fidls.append(fidval)
    return fidls

def negativity_compute(seq_states):
    # Calculate the negativity of the sequence of states
    neg_vals = []
    for state in seq_states:
        rho = Qobj(state)
        Ndim = int(np.log2(rho.shape[0]))
        rho.dims = [[2]*Ndim, [2]*Ndim]
        nval = negativity(rho, subsys=0, logarithmic=False)
        neg_vals.append(nval)
    return neg_vals

def Pauli_collective(rho, Nspins, p, Pauli='Z'):
    if Pauli == 'X':
        single_pauli = sigmax()
    elif Pauli == 'Y':
        single_pauli = sigmay()
    elif Pauli == 'Z':
        single_pauli = sigmaz()
    else:
        single_pauli = identity(2)
    sp_op = single_pauli
    for j in range(1, Nspins):
        sp_op = tensor(sp_op, single_pauli)
    spauli = sp_op.data
    rs_rho = rho.copy()
    tmp = rs_rho @ spauli
    rs_rho = p * (spauli @ tmp) + (1.0 - p) * rs_rho
    return rs_rho

def Pauli_channel(rho, Nspins, p, Pauli='X'):
    if Pauli == 'X':
        single_pauli = sigmax()
    elif Pauli == 'Y':
        single_pauli = sigmay()
    elif Pauli == 'Z':
        single_pauli = sigmaz()
    else:
        single_pauli = identity(2)
    rs_rho = rho.copy()
    for i in range(Nspins):
        spauli = getSci(single_pauli, i, Nspins).data
        tmp = rs_rho @ spauli
        rs_rho = p * (spauli @ tmp) + (1.0-p) * rs_rho
        
    rs_rho = np.array(rs_rho)
    return rs_rho

def depolar_channel(rho, Nspins, p):
    rs_rho = rho.copy()

    for i in range(Nspins):
        spauli_x = getSci(sigmax(), i, Nspins).data
        spauli_y = getSci(sigmay(), i, Nspins).data
        spauli_z = getSci(sigmaz(), i, Nspins).data
        
        rs_rho = 0.25 * p * (spauli_x @ rs_rho @ spauli_x + spauli_y @ rs_rho @ spauli_y + spauli_z @ rs_rho @ spauli_z) \
            + (1.0 - 0.75 * p) * rs_rho
        
    rs_rho = np.array(rs_rho)
    return rs_rho

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
    Y = sigmay()
    Z = sigmaz()
    I = identity(2)

    # Create Hamiltonian
    H0 = getSci(I, 0, Nspins) * 0.0
    H1 = getSci(I, 0, Nspins) * 0.0
    Sxs, Sys, Szs = [], [], []
    for i in range(Nspins):
        Sxi = getSci(X, i, Nspins)
        Syi = getSci(Y, i, Nspins)
        Szi = getSci(Z, i, Nspins)
        Sxs.append(Sxi)
        Sys.append(Syi)
        Szs.append(Szi)
    
    for i in range(Nspins):
        H0 = H0 - B * Szs[i] # Hamiltonian for the magnetic field
        for j in range(i+1, Nspins):
            if alpha > 0:
                hij = J * np.abs(i-j)**(-alpha) / Nalpha
            else:
                hij = J * (np.random.rand() - 0.5)
            H1 = H1 - hij * Sxs[i] * Sxs[j] # Interaction Hamiltonian

    Mx = getSci(I, 0, nobs) * 0.0
    My = getSci(I, 0, nobs) * 0.0
    Mz = getSci(I, 0, nobs) * 0.0
    for i in range(nobs):
        Pxi = getSci(X, i, nobs)
        Pyi = getSci(Y, i, nobs)
        Pzi = getSci(Z, i, nobs)
        Mx += Pxi / nobs
        My += Pyi / nobs
        Mz += Pzi / nobs
        
    H = H0 + H1 # Total Hamiltonian
    L = liouvillian(H, [])
    return L, Mx, My, Mz

def get_transverse_unitary(Nspins, B=2, J=1.0, alpha=0.0, ranseed=0, dt=1.0):
    np.random.seed(seed=ranseed)
    X = sigmax()
    Y = sigmay()
    Z = sigmaz()
    I = identity(2)

    # Create Hamiltonian
    H0 = getSci(I, 0, Nspins) * 0.0
    H1 = getSci(I, 0, Nspins) * 0.0
    Sxs, Sys, Szs = [], [], []
    for i in range(Nspins):
        Sxi = getSci(X, i, Nspins)
        Syi = getSci(Y, i, Nspins)
        Szi = getSci(Z, i, Nspins)
        Sxs.append(Sxi)
        Sys.append(Syi)
        Szs.append(Szi)
    
    for i in range(Nspins):
        gi = J * (np.random.rand() - 0.5)
        H0 = H0 - (B + gi) * Szs[i] # Hamiltonian for the magnetic field
        for j in range(i+1, Nspins):
            if alpha > 0:
                hij = J * np.abs(i-j)**(-alpha) / Nalpha
            else:
                hij = J * (np.random.rand() - 0.5)
            H1 = H1 - hij * Sxs[i] * Sxs[j] # Interaction Hamiltonian

    H = H0 + H1 # Total Hamiltonian
    H = np.array(H)
    U = scipy.linalg.expm(1.j * H * dt)
    return U

def generate_one_qubit_states(ranseed, Nitems, Nreps=1):
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
        for i in range(Nreps):
            rhos.append(rho)
    return rhos

def GHZ_state(N, phase):
    state = (tensor([basis(2) for k in range(N)]) + \
             np.exp(1j*phase) * tensor([basis(2, 1) for k in range(N)]))
    state = state / np.sqrt(2)
    return state

def generate_random_states(ranseed, Nbase, Nitems, distribution='uniform', add=None):
    np.random.seed(seed=ranseed)
    rhos = []
    D = 2**Nbase
    if add == 'GHZ':
        phase_angles = np.random.uniform(low=0.0, high=np.pi, size=Nitems)
        for n in range(Nitems):
            state = ket2dm(GHZ_state(Nbase, phase_angles[n]))
            rho = np.array(state)
            rhos.append(rho)
    elif add == 'Wstate':
        rho = np.array(ket2dm(w_state(Nbase)))
        rhos = [rho] * Nitems
    elif add == 'Cluster':
        for n in range(Nitems):
            # Make adj matrix
            A = make_adj_matrix(Nbase)
            psi = psi_cluster_state(A)
            psi = psi.reshape(len(psi), 1)
            rho = psi @ psi.T.conj()
            
            rhos.append(rho)
    else:
        density_arrs = np.random.uniform(size=Nitems)
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

def proj_spectrahedron(A):
    # To obtain a density matrix, the vector of eigenvalues of the matrix A is projected 
    # onto a standard simplex (non-negative numbers with unit sum)
    # project a matrix onto the spectrahedron
    # returns a positive semidefinite matrix X such that 
    # the trace of X is equal to 1 and the Frobenius norm between X and Hermitian matrix A is minimized
    
    # Fist check A is psd
    if is_positive_semi(A):
        return A
    # to ensure the Hermitian matrix
    B = (A + A.conj().T) / 2.0

    # perform eigenvalue decomposition and remove the imaginary components
    # that arise from numerical precision errors
    eigval, eigvec = np.linalg.eig(B)
    rval = np.real(eigval)

    # project the eigenvalues onto the probability simplex
    u = np.sort(rval)[::-1]
    sv = np.cumsum(u)
    Lu = np.array(np.arange(1, len(u) + 1))
    b = (sv - 1.0) / Lu
    rho = np.argwhere( u > b )[-1][0]
    # if rho == 0:
    #     theta_ = sv[rho] - 1
    # else:
    theta_ = (sv[rho] - 1) / (rho + 1)
    w = rval - theta_
    w[w < 0] = 0
    w = np.sqrt(w).reshape(len(w), 1)
    # reconstitue the matrix while ensuring positive semidefinite
    X = eigvec * w
    X = np.dot(X, X.conj().T)
    return X

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

def convert_vec_to_density(local_vec, postprocess):
    Nbase_sq = int(local_vec.shape[0] / 2)
    Nbase    = int(np.sqrt(Nbase_sq))
    real_rho = np.array(local_vec[:Nbase_sq]).reshape(Nbase, Nbase)
    imag_rho = np.array(local_vec[Nbase_sq:]).reshape(Nbase, Nbase)
    full_rho = real_rho + imag_rho * 1j
    if postprocess == True:
        full_rho = nearest_psd(full_rho, method='eig')
    #full_rho = proj_spectrahedron(full_rho)
    return full_rho

def convert_features_to_density(fevec, postprocess=True):
    rho_ls = []
    for local_vec in fevec:
        rho_ls.append(convert_vec_to_density(local_vec, postprocess))
    rho_ls = np.array(rho_ls)
    return rho_ls

def cal_fidelity_two_mats(matA, matB):
    if check_density(matA) == False or check_density(matB) == False:
        print('Not density matrix')
        fidval = 0.0
    else:
        stateA = Qobj(matA)
        stateB = Qobj(matB)
        fidval = fidelity(stateA, stateB)
    fidval = max(0.0, fidval)
    fidval = min(1.0, fidval)
    return fidval

def cal_trace_dist_two_mats(matA, matB):
    matA = nearest_psd(matA, method='eig')
    matB = nearest_psd(matB, method='eig')
    
    stateA = Qobj(matA)
    stateB = Qobj(matB)
    dtrace = tracedist(stateA, stateB)
    return dtrace

def cal_distance_two_mats(matA, matB, distype='angle'):
    if distype == 'trace':
        distance = cal_trace_dist_two_mats(matA, matB)
    else:
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

def square_distance_correlation(rhoAs, rhoBs, distype='angle'):
    assert(rhoAs.shape[0] == rhoBs.shape[0])
    Nsq = rhoAs.shape[0] ** 2
    distA = make_distance_mat(rhoAs)
    distB = make_distance_mat(rhoBs)
    A = distA - distA.mean(axis=0)[None, :] - distA.mean(axis=1)[:, None] + distA.mean()
    B = distB - distB.mean(axis=0)[None, :] - distB.mean(axis=1)[:, None] + distB.mean()
    dcov2_xy = (A * B).sum()/float(Nsq)
    dcov2_xx = (A * A).sum()/float(Nsq)
    dcov2_yy = (B * B).sum()/float(Nsq)
    dcor = dcov2_xy/np.sqrt(dcov2_xx * dcov2_yy)
    return dcor