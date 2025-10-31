import numpy as np
from scipy.linalg import expm
from build_hamiltonian import read_hamiltonian_and_params, build_hamiltonian
from fermionic_operators import FermionicOperators
from tqdm import tqdm



def greens_function_lehmann_general(H, f_ops, i, j, omega, n, eta=1e-3):
    evals, evecs = np.linalg.eigh(H)
    N = len(evals)
    c_i = f_ops.get_annihilation(i)
    c_jd = f_ops.get_creation(j)

    dim = H.shape[0]
    psi_n = np.zeros(dim)
    psi_n[0] = 1.0  # vacuum state
    psi_init = n @ psi_n

    # Expansion coefficients of psi_init in eigenbasis
    coeffs = evecs.conj().T @ psi_init

    G_w = np.zeros(len(omega), dtype=complex)

    for n in range(N):
        for m in range(N):
            En, Em = evals[n], evals[m]
            cn, cm = coeffs[n], coeffs[m]

            # matrix elements
            A_nm = np.vdot(evecs[:, n], c_i @ evecs[:, m])
            B_mn = np.vdot(evecs[:, m], c_jd @ evecs[:, n])

            # contribution
            G_w += (cn.conj() * cn) * A_nm * B_mn / (omega - (Em - En) + 1j*eta)
            G_w += (cn.conj() * cn) * np.conj(A_nm * B_mn) / (omega + (Em - En) + 1j*eta)

    return G_w



if __name__ == "__main__":
    pass
