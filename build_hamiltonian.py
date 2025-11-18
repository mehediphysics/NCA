import numpy as np
from fermionic_operators import FermionicOperators



def read_hamiltonian_and_params(filename):
    """
    Read N, one-body, two-body Hamiltonian terms, and simulation parameters
    (t_max, dt, i, j) from a text file.
    Expected fields:
        N <int>
        t_max <float>
        dt <float>
        i <int>
        j <int>
        beta <float>

    One-body terms:  i  j  h_ij
    Two-body terms:  i  j  k  l  U_ijkl
    """
    one_body = []
    two_body = []
    N = None
    t_max = None
    dt = None
    i_gf = None
    j_gf = None
    beta = None  #New parameter

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            key = parts[0].lower()
            
            if key == 'n':
                N = int(parts[1])
            elif key == 't_max':
                t_max = float(parts[1])
            elif key == 'dt':
                dt = float(parts[1])
            elif key == 'i':
                i_gf = int(parts[1])
            elif key == 'j':
                j_gf = int(parts[1])
            elif key == 'beta':
                beta = float(parts[1])   # <-- NEW: beta successfully read
            elif len(parts) == 3:
                one_body.append((int(parts[0]), int(parts[1]), float(parts[2])))
            elif len(parts) == 5:
                two_body.append((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), float(parts[4])))
            else:
                raise ValueError(f"Invalid line: {line}")

    # Check required parameters
    if N is None:
        raise ValueError("N not specified in the file.")
    if t_max is None or dt is None:
        raise ValueError("Time parameters t_max or dt not specified.")
    if i_gf is None or j_gf is None:
        raise ValueError("Green's function indices i or j not specified.")
    if beta is None:
        raise ValueError("Error: beta missing in input file.")
    return N, one_body, two_body, t_max, dt, i_gf, j_gf, beta



def build_hamiltonian(N, one_body, two_body):
    """Construct the full Hamiltonian matrix for N states."""
    f = FermionicOperators(N)
    H = np.zeros((2**N, 2**N), dtype=complex)

    # One-body terms
    for i, j, h in one_body:
        H += h * f.get_creation(i) @ f.get_annihilation(j)

    # Two-body terms
    for i, j, k, l, U in two_body:
        H += U * f.get_creation(i) @ f.get_annihilation(j) @ f.get_creation(k) @ f.get_annihilation(l)

    return H



# Example usage:
if __name__ == "__main__":

    filename = "input_params.txt"
    N, one_body, two_body, t_max, dt, i_gf, j_gf = read_hamiltonian_and_params(filename)
    H = build_hamiltonian(N, one_body, two_body)
    
    print("Number of states:", N)
    print("beta =", beta)
    print("Hamiltonian matrix:")
    print(H)
