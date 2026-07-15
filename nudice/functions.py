#!/usr/bin/env python
# coding: utf-8

"""Numerical Lindblad evolution utilities for neutrino oscillations.

The module works in natural units with energies and masses expressed in eV and
baselines expressed in eV^-1.  It evolves one density matrix per energy bin in
mass space.  Dissipation is supplied by the caller through general Lindblad
operators instead of being hard-coded to a visible-neutrino-decay model.
"""

# this code runs internally with eV so these might be useful
meter = 5.06773093741e6        # [eV^-1/m]
km    = 1.0e3*meter            # [eV^-1/km]
MeV   = 1.0e6                  # [eV/MeV]

# === Third-party imports ===@===@===@===@===@===@===

import numpy as np
from odeintw import odeintw
from scipy.linalg import expm

#############################################################################

# === Array/matrix utilities ===@===@===@===@===@===@===

# matrix/array operations
sh = lambda x: print(np.shape(x))
tr = lambda x: print(np.trace(x))

# take diagonal of each block
def dibloc(x): return np.real(np.array([np.diag(block) for block in x]))

# we are gonna need these
def sq(x): return x * x
def cube(x): return x * x * x
def ht(x): return np.heaviside(x, 0)

# conjugate transpose (works for higher dimensional arrays too)
def dagger(x):
    if np.ndim(x) == 2:
        return np.conj(x).T
    if np.ndim(x) == 3:
        return np.transpose(np.conj(x), axes=(0, 2, 1))
    if np.ndim(x) == 4:
        return np.transpose(np.conj(x), axes=(0, 1, 3, 2))
    if np.ndim(x) == 5:
        return np.transpose(np.conj(x), axes=(0, 1, 2, 4, 3))
    raise ValueError("dagger only supports arrays with 2 to 5 dimensions")

# calc bin centres
def calc_bin_centres(bin_edges):
    return 0.5 * (bin_edges[1:] + bin_edges[:-1])


# === Mixing ===@===@===@===@===@===@===

# generate PMNS with custom mixing parameters + CP phase
def Uall(theta12, theta23, theta13, deltaCP):

    d_ = np.exp(-1j * deltaCP)
    d  = np.exp( 1j * deltaCP)

    s12, c12 = np.sin(theta12), np.cos(theta12)
    s23, c23 = np.sin(theta23), np.cos(theta23)
    s13, c13 = np.sin(theta13), np.cos(theta13)

    U = np.linalg.multi_dot(([[1, 0,    0   ],
                               [0, c23,  s23 ],
                               [0, -s23, c23 ]],
                                                [[c13,      0, s13 * d_],
                                                 [0,        1, 0       ],
                                                 [-s13 * d, 0, c13    ]],
                              [[c12,  s12, 0],
                               [-s12, c12, 0],
                               [0,    0,   1]]))
    return U

# rho_m =  U_dagger * rho_f * U (for neutrinos i_nu == 0)
def flav_to_mass(rho, U, i_nu=0):

    if i_nu == 0:
        return np.linalg.multi_dot((dagger(U), rho, U))
    return np.linalg.multi_dot((U, rho, dagger(U)))

# rho_f = U * rho_m * U_dagger (for neutrinos i_nu == 0)
def mass_to_flav(rho, U, i_nu=0):

    if i_nu == 0:
        return np.linalg.multi_dot((U, rho, dagger(U)))
    return np.linalg.multi_dot((dagger(U), rho, U))


# === Open Quantum System Machinery ===@===@===@===@===@===@===

# Dynamical map -> Choi
def s2c(matrix):
    M = matrix.shape[0]
    L = int(np.sqrt(M))
    return np.einsum('abcd->dbca', matrix.reshape(L,L,L,L)).reshape(M,M)

# Choi -> Kraus
def c2k(matrix):
    M = matrix.shape[0]
    L = int(np.sqrt(M))

    # real eigenvalues, sorted ascending, orthonormal evecs
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # ignore tiny eigenvalues
    tol = max(eigenvalues[-1], 1.0) * 1e-10
    mask = eigenvalues > tol
    eigenvalues = eigenvalues[mask]
    eigenvectors = eigenvectors[:, mask]

    # build all Kraus operators at once
    kraus = (np.sqrt(eigenvalues) * eigenvectors).T.reshape(-1, L, L)
    kraus = np.swapaxes(kraus, 1, 2)

    return list(kraus)


def _hamiltonian(e_edges, masses, hamiltonian=None):
    """Build the vacuum Hamiltonian or validate a user-supplied Hamiltonian."""

    e_centr = calc_bin_centres(e_edges)
    n_bins  = len(e_centr)
    Ndim    = len(masses)

    if hamiltonian is None:
        H = np.zeros((n_bins, Ndim, Ndim), dtype=np.complex128)
        for k, m in enumerate(masses):
            H[:, k, k] = (sq(m) - sq(masses[0])) / (2 * e_centr)
        return H

    H = np.asarray(hamiltonian, dtype=np.complex128)
    if H.shape == (Ndim, Ndim):
        return np.broadcast_to(H, (n_bins, Ndim, Ndim)).copy()
    if H.shape == (n_bins, Ndim, Ndim):
        return H
    raise ValueError(
        "hamiltonian must have shape (Ndim, Ndim) or "
        "(n_bins, Ndim, Ndim)."
    )


def lindblad_operators(e_edges, masses, operators=None):
    """Create and validate general Lindblad operators for each energy bin.

    Parameters
    ----------
    e_edges : array-like, shape (n_bins + 1,)
        Energy bin edges.
    masses : array-like, shape (Ndim,)
        Neutrino masses.  Only the length is used here; masses define the state
        dimension shared by the Hamiltonian and density matrices.
    operators : None, array-like, callable, or dict
        General Lindblad jump/decoherence operators in the mass basis.

        Accepted forms are:

        * ``None``: no dissipative terms, giving unitary oscillation.
        * ``(Ndim, Ndim)``: one operator shared by all energy bins.
        * ``(N_ops, Ndim, Ndim)``: several operators shared by all bins.
        * ``(n_bins, N_ops, Ndim, Ndim)``: energy-dependent operators.
        * callable ``operators(E, bin_index, Ndim)`` returning any of the first
          three non-``None`` array forms for that bin.
        * dict mapping labels to matrices or callables.  Each value follows the
          same rules as above, and all values are concatenated along the
          operator axis.

    Returns
    -------
    L_ops : ndarray, shape (n_bins, N_ops, Ndim, Ndim)
        Lindblad operators for every energy bin.
    L_dag : ndarray, shape (n_bins, N_ops, Ndim, Ndim)
        Hermitian adjoints of ``L_ops``.
    sum_LdagL : ndarray, shape (n_bins, Ndim, Ndim)
        Per-bin sum of ``L_a^† L_a``.
    """

    e_centr = calc_bin_centres(e_edges)
    n_bins  = len(e_centr)
    Ndim    = len(masses)

    def normalize_array(value, *, per_bin=False):
        arr = np.asarray(value, dtype=np.complex128)
        if arr.shape == (Ndim, Ndim):
            return arr.reshape(1, Ndim, Ndim) if per_bin else np.broadcast_to(
                arr, (n_bins, 1, Ndim, Ndim)
            ).copy()
        if arr.ndim == 3 and arr.shape[1:] == (Ndim, Ndim):
            return arr if per_bin else np.broadcast_to(
                arr, (n_bins, *arr.shape)
            ).copy()
        if not per_bin and arr.ndim == 4 and arr.shape[0] == n_bins and arr.shape[2:] == (Ndim, Ndim):
            return arr
        raise ValueError(
            "Lindblad operators must have shape (Ndim, Ndim), "
            "(N_ops, Ndim, Ndim), or (n_bins, N_ops, Ndim, Ndim)."
        )

    if operators is None:
        L_ops = np.zeros((n_bins, 0, Ndim, Ndim), dtype=np.complex128)
    elif callable(operators):
        per_bin_ops = []
        for i, E in enumerate(e_centr):
            per_bin_ops.append(normalize_array(operators(E, i, Ndim), per_bin=True))
        n_ops = {ops.shape[0] for ops in per_bin_ops}
        if len(n_ops) != 1:
            raise ValueError("callable operators must return the same number of operators for each bin.")
        L_ops = np.stack(per_bin_ops, axis=0)
    elif isinstance(operators, dict):
        pieces = [lindblad_operators(e_edges, masses, value)[0] for value in operators.values()]
        L_ops = np.concatenate(pieces, axis=1) if pieces else np.zeros((n_bins, 0, Ndim, Ndim), dtype=np.complex128)
    else:
        L_ops = normalize_array(operators)

    L_dag = dagger(L_ops)
    sum_LdagL = np.einsum('baji,bajk->bik', L_dag, L_ops)

    return L_ops, L_dag, sum_LdagL

# the RHS of the master equation ODE - solved at every step, L
def master_eqn(p_flat, baseline, H, L_ops, L_dag, sum_LdagL):

    n_bins = len(H)
    Ndim   = H.shape[-1]

    p  = p_flat.reshape(n_bins, Ndim, Ndim)
    dp = np.zeros((n_bins, Ndim, Ndim), dtype=np.complex128)

    dp += -1j * (np.matmul(H, p) - np.matmul(p, H))
    if L_ops.shape[1] != 0:
        dp -= 0.5 * (sum_LdagL @ p + p @ sum_LdagL)
        dp += np.einsum('baij,bajk,bakl->bil', L_ops, p[:, None], L_dag)

    return dp.ravel()

# the unravelled master equation - returns the dynamical map
def unravelled_master_eqn(L, H, L_ops, L_dag, sum_LdagL):

    n_bins = len(H)
    Ndim   = H.shape[-1]

    L_super = np.zeros((sq(Ndim) * n_bins, sq(Ndim) * n_bins), dtype=np.complex128)

    I = np.eye(Ndim)

    for n in range(n_bins):
        block  = -1j * (np.kron(np.eye(Ndim), H[n]) - np.kron(H[n].T, I))
        if L_ops.shape[1] != 0:
            block -= 0.5 * (np.kron(I, sum_LdagL[n]) + np.kron(sum_LdagL[n].T, I))
            block += sum(np.kron(L.conj(), L) for L in L_ops[n])
        L_super[
            sq(Ndim) * n : sq(Ndim) * (n + 1),
            sq(Ndim) * n : sq(Ndim) * (n + 1),
        ] += block

    return expm(L_super * L)


# === User Functions ===@===@===@===@===@===@===

def lind(initial_value, L, e_edges, masses, operators=None, hamiltonian=None, **kwargs):
    """Solve the Lindblad master equation by directly solving an ODE.

    Parameters
    ----------
    initial_value : ndarray, shape (n_bins, Ndim, Ndim)
        Initial density matrix in the mass basis for each energy bin.
    L : float
        Propagation distance in eV^-1.
    e_edges : array-like, shape (n_bins + 1,)
        Energy bin edges.
    masses : array-like, shape (Ndim,)
        Neutrino masses.  Used to build the vacuum Hamiltonian when no custom
        Hamiltonian is supplied.
    operators : optional
        General Lindblad operators accepted by ``lindblad_operators``.
    hamiltonian : optional
        Custom Hamiltonian with shape ``(Ndim, Ndim)`` or
        ``(n_bins, Ndim, Ndim)``.  If omitted, the vacuum mass-basis
        Hamiltonian is used.
    **kwargs
        Optional ODE settings: ``n_steps`` (default 100), ``rtol`` (default
        1e-8), ``atol`` (default 1e-8), and ``mxstep`` (default 50_000).

    Returns
    -------
    ndarray, shape (n_bins, Ndim, Ndim)
        Evolved density matrix at ``L``.
    """

    H = _hamiltonian(e_edges, masses, hamiltonian)
    L_ops, L_dag, sum_LdagL = lindblad_operators(e_edges, masses, operators)

    n_steps = kwargs.pop("n_steps", 100)
    rtol    = kwargs.pop("rtol", 1e-8)
    atol    = kwargs.pop("atol", 1e-8)
    mxstep  = kwargs.pop("mxstep", 50_000)
    if kwargs:
        raise TypeError(f"Unexpected keyword argument(s): {', '.join(kwargs)}")

    solution = odeintw(master_eqn, initial_value, np.linspace(0, L, n_steps),
                       args=(H, L_ops, L_dag, sum_LdagL),
                       rtol=rtol, atol=atol, mxstep=mxstep)

    return solution[-1].reshape(H.shape)


def dynam(initial_value, L, e_edges, masses, operators=None, hamiltonian=None):
    """Solve the Lindblad equation via the dynamical map.

    The Liouvillian is exponentiated once and applied to the vectorized initial
    state.  This implementation is for per-energy-bin Lindblad evolution; it
    does not redistribute probability between energy bins.
    """

    H = _hamiltonian(e_edges, masses, hamiltonian)
    L_ops, L_dag, sum_LdagL = lindblad_operators(e_edges, masses, operators)

    vec_rho = np.asarray(initial_value, dtype=np.complex128).reshape(-1)
    dy_map = unravelled_master_eqn(L, H, L_ops, L_dag, sum_LdagL)
    solution = dy_map @ vec_rho

    return solution.reshape(H.shape)


def kraus(initial_value, L, e_edges, masses, operators=None, hamiltonian=None):
    """Solve the Lindblad equation via the Kraus operator decomposition.

    This is equivalent to ``dynam`` but converts each energy-bin dynamical-map
    block to Kraus operators explicitly before applying it.
    """

    H = _hamiltonian(e_edges, masses, hamiltonian)
    L_ops, L_dag, sum_LdagL = lindblad_operators(e_edges, masses, operators)

    n_bins = len(H)
    Ndim   = H.shape[-1]
    initial_value = np.asarray(initial_value, dtype=np.complex128)

    dy_map = unravelled_master_eqn(L, H, L_ops, L_dag, sum_LdagL)

    solution = np.zeros((n_bins, Ndim, Ndim), dtype=np.complex128)
    for n in range(n_bins):
        block = dy_map[
            sq(Ndim) * n : sq(Ndim) * (n + 1),
            sq(Ndim) * n : sq(Ndim) * (n + 1),
        ]
        for M in c2k(s2c(block)):
            solution[n] += M @ initial_value[n] @ dagger(M)

    return solution
