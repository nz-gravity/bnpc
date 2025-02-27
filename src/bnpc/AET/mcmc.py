from .sampler import Sampler, MCMCResult
import numpy as np

'''
This file contains the function that runs the MCMC.
'''

def mcmc(
    T: np.ndarray,
    n: int,
    k: int,
    burnin: int,
    A: np.ndarray = None,
    E: np.ndarray = None,
    Spar: np.ndarray = 1,
    Spar_A: np.ndarray = None,
    degree: int = 3,
    modelnum: int = 1,
    f: np.ndarray = None,
    fs: float = None,
    blocked: bool = False,
    signal_model: int = None,
    data_bin_edges: np.ndarray = None,
    data_bin_weights: np.ndarray = None,
    log_data: bool = True
):
    """
    Top-level convenience function that:
      1) Instantiates the internal _MCMC class,
      2) Immediately runs MCMCloop(),
      3) Returns the results (dictionary).
    """
    sampler = Sampler(
        T=T, A=A, E=E,
        n=n, k=k, burnin=burnin, Spar=Spar, Spar_A=Spar_A, degree=degree,
        modelnum=modelnum, f=f, fs=fs, blocked=blocked, signal_model=signal_model,
        data_bin_edges=data_bin_edges, data_bin_weights=data_bin_weights,
        log_data=log_data
    )
    sampler.MCMCloop()
    return MCMCResult(sampler=sampler)