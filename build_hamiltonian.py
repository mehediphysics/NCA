import numpy as np
from fermionic_operators import FermionicOperators

def read_hamiltonian_txt(filename):
    """Read N, one-body and two-body terms from a plain text file (0-based indices)."""
    one_body = []
    two_body = []
    N = None
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0].upper() == 'N':
                N = int(parts[1])
            elif len(parts) == 3:
                # one-body term: i j coeff
                i, j, coeff = int(parts[0]), int(parts[1]), float(parts[2])
                one_body.append((i, j, coeff))
            elif len(parts) == 5:
                # two-body term: i j k l coeff
                i, j, k, l, coeff = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), float(parts[4])
                two_body.append((i, j, k, l, coeff))
            else:
                raise ValueError(f"Invalid line: {line}")
    
    if N is None:
        raise ValueError("Number of states N not specified in the file.")
    
    return N, one_body, two_body

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
    N, one_body, two_body = read_hamiltonian_txt(filename)
    H = build_hamiltonian(N, one_body, two_body)
    
    print("Number of states:", N)
    print("Hamiltonian matrix:")
    print(H)
