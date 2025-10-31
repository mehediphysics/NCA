import numpy as np
import matplotlib.pyplot as plt

from build_hamiltonian import read_hamiltonian_and_params, build_hamiltonian
from greens_functions_time import *
from greens_functions_energy import *

from plotting import *
from save_output import *




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
    omega = np.linspace(-10, 10, 10000)
    G_w = greens_function_lehmann_general(H, f_ops, i_gf, j_gf, omega, 
                                          f_ops.get_creation(1)@f_ops.get_creation(0), 
                                          eta=1e-4)

    # Save output to file
    outfile = "data_"+str(N)+"_"+str(t_max)+"_"+str(dt)+"_"+str(i_gf)+"_"+str(j_gf)
    save_greens_function_npz(outfile, times, G_tt, omega, G_w)

    # Plot results
    plot_gf_and_spectrum(times, G_tt, i_gf, j_gf, omega, G_w.imag)
    plot_cuts(times, G_tt, i_gf, j_gf)

    plt.show()

