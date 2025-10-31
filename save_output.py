import numpy as np

def save_greens_function_npz(filename, times, G_tt, omega=None, G_w=None):
    """
    Save the Green's function data to a NumPy .npz file.
    
    ```
    Parameters
    ----------
    filename : str
        Output filename (e.g. 'greens_function_output.npz').
    times : array
        Time grid used for both t and t'.
    G_tt : 2D array (complex)
        Two-time Green's function G(t, t').
    omega : array, optional
        Frequency grid for the spectral (Fourier-transformed) data.
    G_w : array, optional
        Frequency-domain Green's function G(ω).
    """
    if not filename.endswith(".npz"):
        filename += ".npz"
    np.savez(filename, times=times, G_tt=G_tt, omega=omega, G_w=G_w)
    print(f"Saved Green's function data to '{filename}'.")


def load_greens_function_npz(filename):
    """
    Load Green's function data from a NumPy .npz file.
    
    ```
    Returns
    -------
    times : array
        Time grid for G(t, t').
    G_tt : 2D array (complex)
        Two-time Green's function G(t, t').
    omega : array or None
        Frequency grid, if present.
    G_w : array or None
        Frequency-domain Green's function G(ω), if present.
    """
    if not filename.endswith(".npz"):
        filename += ".npz"
    data = np.load(filename, allow_pickle=True)
    times = data["times"]
    G_tt = data["G_tt"]
    omega = data["omega"] if data["omega"] is not None else None
    G_w = data["G_w"] if data["G_w"] is not None else None
    return times, G_tt, omega, G_w




if __name__ == "__main__":
    pass

