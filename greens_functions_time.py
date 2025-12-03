import numpy as np
from scipy.linalg import expm
#from build_hamiltonian import read_hamiltonian_and_params, build_hamiltonian
from build_hamiltonian import (
    read_hamiltonian_and_params,
    build_hamiltonian,
    build_H0,
)

from fermionic_operators import FermionicOperators
from tqdm import tqdm

def greens_function_two_time(H, f_ops, times, i, j, n):
    """
    Full two-time Green's function:
        G_ij(t, t') = -i <psi_n | { c_i(t), c_j^†(t') } | psi_n >
    H: Hamiltonian matrix
    f_ops: FermionicOperators object
    times: array of times
    i, j: indices of fermionic operators
    psi_n: initial state (vector of size 2^N)
    """
    dim = H.shape[0]
    c_i = f_ops.get_annihilation(i)
    c_jd = f_ops.get_creation(j)

    psi_n = np.zeros(dim)
    psi_n[0] = 1.0  # vacuum state
    psi_n = n @ psi_n
    f_ops.print_occupation(psi_n)

    G_tt = np.zeros((len(times), len(times)), dtype=complex)

    #for t_idx, t in enumerate(times):a
    for t_idx, t in enumerate(tqdm(times, desc="Calculating G(t,t')")):
        U_t = expm(-1j * H * t)
        U_t_dag = expm(1j * H * t)
        c_i_t = U_t_dag @ c_i @ U_t  # Heisenberg picture

        for tp_idx, tp in enumerate(times):
            U_tp = expm(-1j * H * tp)
            U_tp_dag = expm(1j * H * tp)
            c_j_tp = U_tp_dag @ c_jd @ U_tp

            term1 = psi_n.conj().T @ (c_i_t @ (c_j_tp @ psi_n))
            term2 = psi_n.conj().T @ (c_j_tp @ (c_i_t @ psi_n))
            G_tt[t_idx, tp_idx] = -1j * term1 - 1j * term2

    Nt = len(times)
    # Create a mask: t > t' -- this gives us the retarded GF
    mask = np.tril(np.ones((Nt, Nt), dtype=bool), k=0)  # upper triangular, k=0 includes diagonal
    # Apply mask to G_tt
    G_tt_masked = np.where(mask, G_tt, 0.0)
    G_tt = G_tt_masked
    return G_tt



if __name__ == "__main__":
    pass
