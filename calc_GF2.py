import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# These imported modules provide:
#   • Building many-body Hamiltonian H = H₁ + H₂
#   • Time-domain Green's function G(t,t')
#   • Energy-domain (ω) Lehmann Green’s function G(ω)
#   • Plotting tools
#   • Saving results to .npz
# ------------------------------------------------------------
from build_hamiltonian import read_hamiltonian_and_params, build_hamiltonian
from greens_functions_time import *
from greens_functions_energy import *
from plotting import *
from save_output import *



# ============================================================
# MAIN SCRIPT
# ============================================================
if __name__ == "__main__":

    # ============================================================
    # Read all parameters from "input_params.txt".
    #
    # Returned values:
    #   N        -> number of fermionic modes
    #   one_body -> list of one-body terms (i,j,h_ij)
    #   two_body -> list of two-body terms (i,j,k,l,U_ijkl)
    #   t_max    -> maximum simulation time
    #   dt       -> time step
    #   i_gf,j_gf-> indices for Green's function G_{i_gf,j_gf}
    #
    # Hamiltonian structure:
    #   H = Σ_ij h_ij c_i† c_j
    #     + Σ_ijkl U_ijkl c_i† c_j c_k† c_l
    # ============================================================
    filename = "input_params.txt"
    N, one_body, two_body, t_max, dt, i_gf, j_gf = read_hamiltonian_and_params(filename)



    # ============================================================
    # Construct many-body Hamiltonian (size 2^N × 2^N)
    #
    # FermionicOperators provides:
    #   c_i      = annihilation operator
    #   c_i†     = creation operator
    #
    # Jordan-Wigner ensures:
    #   {c_i, c_j†} = δ_ij
    #   {c_i, c_j} = {c_i†, c_j†} = 0
    # ============================================================
    H = build_hamiltonian(N, one_body, two_body)
    f_ops = FermionicOperators(N)



    # ============================================================
    # Time grid:
    #
    #   t ∈ {0, dt, 2dt, ..., t_max}
    #
    # Total number of time steps:
    #   Nt = t_max / dt
    # ============================================================
    times = np.arange(0, t_max, dt)



    # ============================================================
    # Thermal-prefactor parameters for initial state
    #
    # We are constructing something like a *thermal mixture*
    # of different many-body occupations with Boltzmann weights:
    #
    #   w_α = e^{-β E_α} / Z
    #
    # where:
    #   β      = inverse temperature
    #   E_α    = energy of configuration α
    #   Z      = partition function = Σ_α e^{-β E_α}
    #
    # Here:
    #   ε_u    : "up" level energy
    #   ε_d    : "down" level energy
    #   U      : interaction energy when both are occupied
    # ============================================================
    beta  = 1.0    # inverse temperature β (you can change)
    eps_u = 0.07   # ε_up
    eps_d = 0.08   # ε_down
    U     = 0.9    # interaction energy



    # ============================================================
    # Define the energies of four many-body configurations:
    #
    #   |0>               → E_0      = 0
    #   |u>   (one ↑)     → E_u      = ε_u
    #   |d>   (one ↓)     → E_d      = ε_d
    #   |ud>  (↑ and ↓)   → E_ud     = ε_u + ε_d + U
    #
    # Corresponding Boltzmann weights:
    #
    #   w_0  ∝ e^{-β E_0}      = e^{0}              = 1
    #   w_u  ∝ e^{-β ε_u}
    #   w_d  ∝ e^{-β ε_d}
    #   w_ud ∝ e^{-β (ε_u + ε_d + U)}
    #
    # Partition function:
    #
    #   Z = 1
    #       + e^{-β ε_u}
    #       + e^{-β ε_d}
    #       + e^{-β (ε_u + ε_d + U)}
    #
    # Below, we *use these same weights* as coefficients in
    # the operator that prepares the initial state.
    #
    # NOTE: You can choose which mode index corresponds to
    #       "up" and "down". Here as an example:
    #           mode 0 → main site
    #           mode 2 → "up"-like
    #           mode 3 → "down"-like
    # ============================================================
    w0  = 1.0
    wu  = np.exp(-beta * eps_u)
    wd  = np.exp(-beta * eps_d)
    wud = np.exp(-beta * (eps_u + eps_d + U))

    Z = w0 + wu + wd + wud  # partition function



    # ============================================================
    # Define initial operator O such that:
    #
    #   |ψ₀> ∝ O |0>
    #
    # Example operator superposition:
    #
    #   O = (1/Z) [
    #         1        · c_0†
    #       + e^{-β ε_u}              · c_0† c_2†
    #       + e^{-β ε_d}              · c_0† c_3†
    #       + e^{-β (ε_u+ε_d+U)}      · c_0† c_2† c_3†
    #       ]
    #
    # This mimics a *Boltzmann-weighted* mixture of:
    #   |0>          (empty aux modes)
    #   |u>          (mode 2 occupied)
    #   |d>          (mode 3 occupied)
    #   |ud>         (modes 2 and 3 occupied)
    #
    # all tied to creation on mode 0 as the "observed" site.
    # ============================================================
    psi0_operator = (
          (w0  / Z) * (f_ops.get_creation(0))
        + (wu  / Z) * (f_ops.get_creation(0) @ f_ops.get_creation(2))
        + (wd  / Z) * (f_ops.get_creation(0) @ f_ops.get_creation(3))
        + (wud / Z) * (f_ops.get_creation(0) @ f_ops.get_creation(2) @ f_ops.get_creation(3))
    )

    # Now the many-body initial state is:
    #   |ψ₀> = psi0_operator |0>



    # ============================================================
    # Compute the full two-time Green’s function G(t,t’):
    #
    # Heisenberg operators:
    #       c_i(t)   = e^(+iHt) c_i e^(-iHt)
    #       c_j†(t') = e^(+iHt') c_j† e^(-iHt')
    #
    # Anti-commutator Green’s function:
    #
    #   G_ij(t,t') = -i [ ⟨ψ₀| c_i(t) c_j†(t') |ψ₀⟩
    #                     + ⟨ψ₀| c_j†(t') c_i(t) |ψ₀⟩ ]
    #
    # Returned object:
    #   G_tt is an Nt × Nt complex matrix
    # ============================================================
    G_tt = greens_function_two_time(
        H, f_ops, times, i_gf, j_gf, psi0_operator
    )



    # ============================================================
    # Compute energy-domain Green’s function G(ω)
    #
    # Lehmann representation:
    #
    #   G_ij(ω) = Σ_nm [
    #       ( ⟨ψ|c_i|m⟩ ⟨m|c_j†|ψ⟩ ) / (ω - (E_m - E_ψ) + iη )
    #       +
    #       ( ⟨ψ|c_j†|m⟩ ⟨m|c_i|ψ⟩ ) / (ω + (E_m - E_ψ) + iη )
    #   ]
    #
    # where:
    #   H|m⟩ = E_m |m⟩
    #
    # Spectral function:
    #   A(ω) = -(1/π) Im G(ω)
    # ============================================================
    omega = np.linspace(-10, 10, 10000)

    G_w = greens_function_lehmann_general(
        H, f_ops, i_gf, j_gf, omega,
        psi0_operator,
        eta=1e-4
    )



    # ============================================================
    # Save all results:
    #
    #   outfile = data_N_tmax_dt_i_j.npz
    #
    # Contains:
    #   times[]   - time grid
    #   G_tt      - two-time Green’s function matrix
    #   omega[]   - frequency grid
    #   G_w       - frequency-domain Green’s function
    # ============================================================
    outfile = (
        "data_"
        + str(N) + "_"
        + str(t_max) + "_"
        + str(dt) + "_"
        + str(i_gf) + "_"
        + str(j_gf)
    )

    save_greens_function_npz(outfile, times, G_tt, omega, G_w)



    # ============================================================
    # Plot results:
    #
    # plot_gf_and_spectrum():
    #     - Heatmap of Re[G(t,t')]
    #     - FFT of anti-diagonal: G(t - t')
    #     - Overlay exact G(ω)
    #
    # plot_cuts():
    #     - Diagonal G(t,t)
    #     - Anti-diagonal G(t − t')
    # ============================================================
    plot_gf_and_spectrum(times, G_tt, i_gf, j_gf, omega, G_w.imag)
    plot_cuts(times, G_tt, i_gf, j_gf)

    plt.show()
