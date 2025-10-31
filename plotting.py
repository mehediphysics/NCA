import numpy as np
import matplotlib.pyplot as plt

from build_hamiltonian import read_hamiltonian_and_params, build_hamiltonian
from save_output import *



def extract_diagonal(G_tt):
    """Extract diagonal t=t'"""
    return np.diag(G_tt)


def extract_antidiagonal(G_tt):
    """Extract antidiagonal t+t' = constant (here we take t_max)"""
    return np.diag(np.fliplr(G_tt))


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

    # Fourier transform along anti-diagonal t-t' (simple approximation)
    dt = times[1] - times[0]
    G_anti = extract_antidiagonal(G_tt)
    G_omega = np.fft.fft(G_anti)
    freqs = np.fft.fftfreq(len(G_anti), dt)

    plt.subplot(1,2,2)
    plt.plot(np.fft.fftshift(freqs), np.fft.fftshift(G_omega).real, label="real")
    plt.plot(np.fft.fftshift(freqs), np.fft.fftshift(G_omega).imag, label="imag")
    if omega is not None and G_w is not None:
        plt.plot(omega, G_w, "k--")

    plt.xlabel("Frequency ω")
    plt.ylabel("|G(ω)|")
    plt.tight_layout()
    plt.legend()



if __name__ == "__main__":
    # Read all parameters from text file
    filename = "input_params.txt"
    N, one_body, two_body, t_max, dt, i_gf, j_gf = read_hamiltonian_and_params(filename)

    outfile = "data_"+str(N)+"_"+str(t_max)+"_"+str(dt)+"_"+str(i_gf)+"_"+str(j_gf)
    # Plot results
    times, G_tt, omega, G_w = load_greens_function_npz(outfile)
    plot_gf_and_spectrum(times, G_tt, i_gf, j_gf, omega, G_w.imag)
    plot_cuts(times, G_tt, i_gf, j_gf)

    plt.show()


