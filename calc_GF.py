import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from build_hamiltonian import read_hamiltonian_and_params, build_hamiltonian
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

def extract_diagonal(G_tt):
    """Extract diagonal t=t'"""
    return np.diag(G_tt)

def extract_antidiagonal(G_tt):
    """Extract antidiagonal t+t' = constant (here we take t_max)"""
    return np.diag(np.fliplr(G_tt))


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

            # (optional) include coherence terms if desired
            # G_w += cn.conj() * cm * A_nm * B_mn / (omega - (Em - (En+Em)/2) + 1j*eta)
    return G_w


def plot_cuts(times, G_tt, i, j):
    G_diag = extract_diagonal(G_tt)
    G_anti = extract_antidiagonal(G_tt)

    plt.figure(figsize=(10,4))
    plt.plot(times, G_diag.real, label="Re Diagonal t=t'")
    plt.plot(times, G_diag.imag, label="Im Diagonal t=t'")
    plt.xlabel("time t")
    plt.ylabel(f'G_{i}{j}(t,t)')
    plt.legend()
    plt.axhline(y=0, color='k', linestyle='--')
    plt.axhline(y=1, color='k', linestyle='--')
    plt.axhline(y=-1, color='k', linestyle='--')
    plt.title(f"Cuts along G_{i}{j}(t,t)  (population)")

    plt.figure(figsize=(10,4))
    plt.plot(times, G_anti.real, '-', color="tab:green", label="Re Antidiagonal t+t'=t_max")
    plt.plot(times, G_anti.imag, '-', color="tab:red", label="Im Antidiagonal t+t'=t_max")
    plt.xlabel("time t")
    plt.ylabel(f'G_{i}{j}(t,t)')
    plt.legend()
    plt.axvline(x=times[int(len(times)/2)], color='k', linestyle='--', label="t-t'")
    plt.axhline(y=0, color='k', linestyle='--')
    plt.title(f"Cuts along G_{i}{j}(t-t')  (off-diagonal elements)")


def plot_gf_and_spectrum(times, G_tt, i, j, omega=None, G_w=None):
    # Time-domain 2D plot
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.imshow(np.real(G_tt), extent=[times[0], times[-1], times[0], times[-1]],
               origin='lower', aspect='auto', cmap='RdBu')
    plt.colorbar(label=f'Re G_{i}{j}(t,t\')')
    plt.xlabel("t'")
    plt.ylabel("t")
    plt.title(f"Two-time Green's function Re G_{i}{j}(t, t')")

    # Fourier transform along diagonal t-t' (simple approximation)
    dt = times[1] - times[0]
    G_anti = extract_antidiagonal(G_tt)
    G_omega = np.fft.fft(G_anti)
    freqs = np.fft.fftfreq(len(G_anti), dt)

    plt.subplot(1,2,2)
    plt.plot(np.fft.fftshift(freqs), np.fft.fftshift(G_omega.real), label="real")
    plt.plot(np.fft.fftshift(freqs), np.fft.fftshift(G_omega.imag), label="imag")
    if omega is not None and G_w is not None:
        plt.plot(omega, G_w, "k--")

    plt.xlabel("Frequency ω")
    plt.ylabel("|G(ω)|")
    plt.tight_layout()



if __name__ == "__main__":
    # Read all parameters from text file
    filename = "input_params.txt"
    N, one_body, two_body, t_max, dt, i_gf, j_gf = read_hamiltonian_and_params(filename)

    # Build Hamiltonian and fermionic operators
    H = build_hamiltonian(N, one_body, two_body)
    f_ops = FermionicOperators(N)

    # Time grid
    times = np.arange(0, t_max, dt)

    # Compute full two-time Green's function
    G_tt = greens_function_two_time(H, f_ops, times, i_gf, j_gf, 
                                    f_ops.get_creation(1)@f_ops.get_creation(0))

    # Compute Lehman representation for reference
    omega = np.linspace(-10, 10, 1000)
    G_w = greens_function_lehmann_general(H, f_ops, i_gf, j_gf, omega, 
                                          f_ops.get_creation(1)@f_ops.get_creation(0), 
                                          eta=1e-5)

    # Plot results
    plot_gf_and_spectrum(times, G_tt, i_gf, j_gf, omega, G_w)
    plot_cuts(times, G_tt, i_gf, j_gf)

    plt.show()

