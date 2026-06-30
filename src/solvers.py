import numpy as np
from scipy.linalg import eig

def solve_generalized_eigenproblem(A, B):
    """
    Solves the linear system network and isolates the most unstable modal growth settings.
    """
    eigenvalues, eigenfunctions = eig(A, B)
    omega = eigenvalues  
    
    # Identify unstable indices where imaginary growth profile matches Im(omega) > 0
    unstable_modes = np.where(np.imag(omega) > 0)[0]
    
    if len(unstable_modes) == 0:
        return omega, 0, None, None, None
        
    # Isolate maximum structural growth parameter characteristics
    max_growth_idx = np.argmax(np.imag(omega))
    c_max = omega[max_growth_idx]
    psi_unstable = np.real(eigenfunctions[:, max_growth_idx])
    
    return omega, len(unstable_modes), c_max, max_growth_idx, psi_unstable
