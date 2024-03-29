#!/usr/bin/env python
"""
    Utils for higher-order quantum reservoir computing framework
"""
import sys
import numpy as np
import math, os
from scipy.stats import unitary_group
from scipy.special import softmax
from qutip import *

LINEAR_PINV = 'linear_pinv'
RIDGE_PINV  = 'ridge_pinv'
RIDGE_AUTO  = 'auto'
RIDGE_SVD   = 'svd'
RIDGE_CHOLESKY = 'cholesky'
RIDGE_LSQR = 'lsqr'
RIDGE_SPARSE = 'sparse_cg'
RIDGE_SAG = 'sag'

DYNAMIC_FULL_RANDOM = 'full_random'
DYNAMIC_HALF_RANDOM = 'half_random'
DYNAMIC_FULL_CONST_TRANS = 'full_const_trans'
DYNAMIC_FULL_CONST_COEFF = 'full_const_coeff'
DYNAMIC_ION_TRAP = 'ion_trap'

def plotContour(fig, ax, data, title, fontsize, vmin, vmax, cmap, colorbar=True):
    ax.set_title(title, fontsize=fontsize)
    t, s = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
    mp = ax.contourf(s, t, np.transpose(data), 15, cmap=cmap, levels=np.linspace(vmin, vmax, 60), extend="both", zorder=-20)
    #mp = ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
                
    if colorbar == True:
        fig.colorbar(mp, ax=ax)
    ax.set_rasterization_zorder(-10)
    #ax.set_xlabel(r"Time", fontsize=fontsize)
    return mp

class QRCParams():
    def __init__(self, n_units, n_envs, max_energy, non_diag, alpha, beta, virtual_nodes, tau, init_rho, \
        solver=LINEAR_PINV, dynamic=DYNAMIC_FULL_CONST_TRANS):
        self.n_units = n_units
        self.n_envs = n_envs
        self.max_energy = max_energy
        self.non_diag = non_diag
        self.alpha = alpha
        self.beta = beta
        self.virtual_nodes = virtual_nodes
        self.tau = tau
        self.init_rho = init_rho
        self.solver = solver
        self.dynamic = dynamic

    def info(self):
        print('units={},n_envs={},J={},non_diag={},alpha={},V={},t={},init_rho={}'.format(\
            self.n_units, self.n_envs, self.max_energy, self.non_diag, self.alpha,
            self.virtual_nodes, self.tau, self.init_rho))

def solfmax_layer(states):
    states = np.array(states)
    return softmax(states)

def softmax_linear_combine(u, states, coeffs):
    states = solfmax_layer(states)
    return linear_combine(u, states, coeffs)

def linear_combine(u, states, coeffs):
    assert(len(coeffs) == len(states))
    v = 1.0 - np.sum(coeffs)
    assert(v <= 1.00001 and v >= -0.00001)
    v = max(v, 0.0)
    v = min(v, 1.0)
    total = v * u
    total += np.dot(np.array(states).flatten(), np.array(coeffs).flatten())
    return total

def scale_linear_combine(u, states, coeffs, bias=0):
    if bias != 0:
        states = (states + bias) / (2.0 * bias)
    return linear_combine(u, states, coeffs)

def make_data_for_narma(length, orders, ranseed=-1):
    if ranseed >= 0:
        np.random.seed(seed=ranseed)
    xs = np.random.rand(length)
    x = xs * 0.2
    N = len(orders)
    Y = np.zeros((length, N))
    for j in range(N):
        order = orders[j]
        y = np.zeros(length)
        if order == 2:
            for i in range(length):
                y[i] = 0.4 * y[i-1] + 0.4 * y[i-1]*y[i-2] + 0.6 * (x[i]**3) + 0.1
        else:
            for i in range(length):
                if i < order:
                    y[i] = 0.3 * y[i - 1] + 0.05 * y[i - 1] * np.sum(np.hstack((y[i - order:], y[:i]))) + \
                        1.5 * x[i - order + 1] * x[i] + 0.1
                else:
                    y[i] = 0.3 * y[i - 1] + 0.05 * y[i - 1] * np.sum(np.hstack((y[i - order:i]))) + \
                        1.5 * x[i - order + 1] * x[i] + 0.1
        Y[:,j] = y
    return xs, Y

# Generate MNIST dataset with appropriate size
def gen_mnist_dataset(mnist_dir, mnist_size):
    import gzip
    import _pickle as cPickle

    f = gzip.open(os.path.join(mnist_dir, 'mnist_{}.pkl.gz'.format(mnist_size)),'rb')
    data = cPickle.load(f, encoding='latin1')
    f.close()
    train_set, valid_set, test_set = data

    xs_train, ys_train = train_set
    xs_test, ys_test = test_set
    xs_val, ys_val = valid_set

    xs_train = xs_train / 255.0
    xs_test = xs_test / 255.0
    xs_val  = xs_val / 255.0

    return xs_train, ys_train, xs_test, ys_test, xs_val, ys_val

# Generate sparse matrix with row norm <= gamma
# sparse matrix with elements between [-1, 1]
def gen_sparse_mat(sizex, sizey, sparsity, rseed=1):
    W = sparse.random(sizex, sizey, density = sparsity, random_state=rseed)
    W *= 2.0
    W -= 1.0
    return W
    

# Reference from
# https://qiskit.org/documentation/_modules/qiskit/quantum_info/random/utils.html

def random_state(dim, seed=None):
    """
    Return a random quantum state from the uniform (Haar) measure on
    state space.

    Args:
        dim (int): the dim of the state space
        seed (int): Optional. To set a random seed.

    Returns:
        ndarray:  state(2**num) a random quantum state.
    """
    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.RandomState(seed)
    # Random array over interval (0, 1]
    x = rng.rand(dim)
    x += x == 0
    x = -np.log(x)
    sumx = sum(x)
    phases = rng.rand(dim) * 2.0 * np.pi
    return np.sqrt(x / sumx) * np.exp(1j * phases)

def random_density_matrix(length, rank=None, method='Hilbert-Schmidt', seed=None):
    """
    Generate a random density matrix rho.

    Args:
        length (int): the length of the density matrix.
        rank (int or None): the rank of the density matrix. The default
            value is full-rank.
        method (string): the method to use.
            'Hilbert-Schmidt': sample rho from the Hilbert-Schmidt metric.
            'Bures': sample rho from the Bures metric.
        seed (int): Optional. To set a random seed.
    Returns:
        ndarray: rho (length, length) a density matrix.
    Raises:
        QiskitError: if the method is not valid.
    """
    if method == 'Hilbert-Schmidt':
        return __random_density_hs(length, rank, seed)
    elif method == 'Bures':
        return __random_density_bures(length, rank, seed)
    else:
        raise ValueError('Error: unrecognized method {}'.format(method))


def __ginibre_matrix(nrow, ncol=None, seed=None):
    """
    Return a normally distributed complex random matrix.

    Args:
        nrow (int): number of rows in output matrix.
        ncol (int): number of columns in output matrix.
        seed (int): Optional. To set a random seed.
    Returns:
        ndarray: A complex rectangular matrix where each real and imaginary
            entry is sampled from the normal distribution.
    """
    if ncol is None:
        ncol = nrow
    rng = np.random.RandomState(seed)

    ginibre = rng.normal(size=(nrow, ncol)) + rng.normal(size=(nrow, ncol)) * 1j
    return ginibre


def __random_density_hs(length, rank=None, seed=None):
    """
    Generate a random density matrix from the Hilbert-Schmidt metric.

    Args:
        length (int): the length of the density matrix.
        rank (int or None): the rank of the density matrix. The default
            value is full-rank.
        seed (int): Optional. To set a random seed.
    Returns:
        ndarray: rho (N,N  a density matrix.
    """
    ginibre = __ginibre_matrix(length, rank, seed)
    ginibre = ginibre.dot(ginibre.conj().T)
    return ginibre / np.trace(ginibre)


def __random_density_bures(length, rank=None, seed=None):
    """
    Generate a random density matrix from the Bures metric.

    Args:
        length (int): the length of the density matrix.
        rank (int or None): the rank of the density matrix. The default
            value is full-rank.
        seed (int): Optional. To set a random seed.
    Returns:
        ndarray: rho (N,N) a density matrix.
    """
    density = np.eye(length) + random_unitary(length).data
    ginibre = density.dot(__ginibre_matrix(length, rank, seed))
    ginibre = ginibre.dot(ginibre.conj().T)
    return ginibre / np.trace(ginibre)

def partial_trace(rho, keep, dims, optimize=False):
    """
    Calculate the partial trace.
    Consider a joint state ρ on the Hilbert space :math:`H_a \otimes H_b`. We wish to trace out
    :math:`H_b`
    .. math::
        ρ_a = Tr_b(ρ)
    :param rho: 2D array, the matrix to trace.
    :param keep: An array of indices of the spaces to keep after being traced. For instance,
                 if the space is A x B x C x D and we want to trace out B and D, keep = [0, 2].
    :param dims: An array of the dimensions of each space. For example, if the space is
                 A x B x C x D, dims = [dim_A, dim_B, dim_C, dim_D].
    :param optimize: optimize argument in einsum
    :return:  ρ_a, a 2D array i.e. the traced matrix
    """
    # Code from
    # https://scicomp.stackexchange.com/questions/30052/calculate-partial-trace-of-an-outer-product-in-python
    keep = np.asarray(keep)
    dims = np.asarray(dims)
    Ndim = dims.size
    Nkeep = np.prod(dims[keep])

    idx1 = [i for i in range(Ndim)]
    idx2 = [Ndim + i if i in keep else i for i in range(Ndim)]
    rho_a = rho.reshape(np.tile(dims, 2))
    rho_a = np.einsum(rho_a, idx1 + idx2, optimize=optimize)
    return rho_a.reshape(Nkeep, Nkeep)
