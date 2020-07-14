from qutip import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from numpy import linalg as LA
import pickle

def ploteig(eval):
    fig, ax = plt.subplots()
    patches = []
    circle = Circle((0, 0), 1.0)
    patches.append(circle)
    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.1)
    ax.add_collection(p)
    plt.axis('equal')
    
    for xi, yi in zip(np.real(eval), np.imag(eval)):
        plt.plot(xi, yi,'o')
    plt.savefig('eigval.png', bbox_inches='tight')
    plt.show()

def demoGerschgorin(A):

    n = len(A)
    eval, evec = LA.eig(A)

    patches = []
    
    # draw discs
    
    for i in range(n):
        xi = np.real(A[i,i])
        yi = np.imag(A[i,i])
        ri = np.sum(np.abs(A[i,:])) - np.abs(A[i,i]) 
        
        circle = Circle((xi, yi), ri)
        patches.append(circle)

    fig, ax = plt.subplots()

    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.1)
    ax.add_collection(p)
    plt.axis('equal')
    
    for xi, yi in zip(np.real(eval), np.imag(eval)):
        plt.plot(xi, yi,'o')
    plt.savefig('eigenve.png', bbox_inches='tight')
    plt.show()

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

r = np.random.rand(4, 4)
qo = Qobj(r)
#print(qo)

to = tensor(basis(2, 0), basis(2, 0))
#print(to)

psi = basis(2, 0)
rho = ket2dm(psi)

X = sigmax()
S = spre(X) * spost(X.dag()) # Represents conjugation by X.
print(S.shape, S.iscp, S.istp, S.iscptp)
#print(S.eigenstates())
#print(S)
#print(X, spre(X), spost(X.dag()))

Nspins = 4
alpha = 0.2
J = 0
for j in range(Nspins):
    for i in range(j+1, Nspins):
        Jij = np.abs(i-j)**(-alpha)
        J += Jij / (Nspins-1)
B = J/0.42 # Magnetic field

X = sigmax()
Z = sigmaz()
I = identity(2)
H0 = getSci(I, 0, Nspins) * 0.0
H1 = getSci(I, 0, Nspins) * 0.0
for i in range(Nspins):
    Szi = getSci(Z, i, Nspins)
    H0 = H0 - B * Szi # Hamiltonian for the magnetic field
    for j in range(Nspins):
        if i != j:
            Sxi = getSci(X, i, Nspins)
            Sxj = getSci(X, j, Nspins)
            hij = np.abs(i-j)**(-alpha) / J
            H1 = H1 - hij * Sxi * Sxj # Interaction Hamiltonian
H = H0 + H1 # Total Hamiltonian
# H = - tensor(sigmaz(), identity(2), identity(2)) \
#     - tensor(identity(2), sigmaz(), identity(2)) \
#     - tensor(identity(2), identity(2), sigmaz()) \
#     - 0.1 * tensor(sigmax(), sigmax(), identity(2)) \
#     - 0.2 * tensor(sigmax(), identity(2), sigmax()) \
#     - 0.3 * tensor(identity(2), sigmax(), sigmax())
# times = np.linspace(0.0, 10.0, 100)
#H = tensor(sigmaz(), sigmax())
L = liouvillian(H, [])
#print(L)
#print(S.eigenstates()[0])
# result = sesolve(H, psi, times, [sigmaz(), sigmay()])

# fig, ax = plt.subplots()
# ax.plot(result.times, result.expect[0])
# ax.plot(result.times, result.expect[1])
# ax.legend(("Sigma-Z", "Sigma-Y"))
# plt.show()

# rho = tensor(ket2dm((basis(2, 0) + basis(2, 1)).unit()), fock_dm(2, 0))
# print(rho.ptrace(0))

q = getSci(basis(2), 0, Nspins)
#q = tensor(identity(2), basis(2))
# print(q)
s_prep = sprepost(q, q.dag())
print('s_prep', s_prep.shape)

fig, ax = plt.subplots()
tls, vals, first = [], [], []
results = dict()
for tau in np.linspace(0.0, 10.0, 1):
    tls.append(B*tau)
    S = (tau*L).expm()
    print(S.shape, S.iscp, S.istp, S.iscptp)
    #print(to_super(cnot()))
    #ts = tensor_contract(S, (2, 5))
    ts = tensor_contract(S, (0, Nspins))
    print(tau, 'ts', ts.shape)
    ts = ts * s_prep
    print(tau, ts.shape, ts.iscp, ts.istp, ts.iscptp)
    egvals = ts.eigenstates()[0]
    second_eg = egvals[-2]
    vals.append(np.abs(second_eg))
    first.append(np.abs(egvals[-1]))
    results[tau] = egvals
    #ploteig(egvals)
    print(egvals.shape)
# filename = 'egval'
# with open(filename, 'wb') as wrs:
#     pickle.dump(results, wrs)

# with open(filename, 'rb') as rrs:
#     z = pickle.load(rrs)
# print(z.keys())

# ax.plot(tls, vals, 'o-')
# ax.plot(tls, first, 'o-')
# plt.savefig('second_largest.png', bbox_inches='tight')
# plt.show()

# A = ts.full()
# A = np.matrix(A)
# print(A.shape)
# demoGerschgorin(A)
# M = tensor(sigmaz(), sigmax(), sigmay())
# N = tensor(sigmaz(), sigmax())
# P = tensor(N, sigmay())
# print(M==P)


print(basis(2, 0), basis(2, 1))
