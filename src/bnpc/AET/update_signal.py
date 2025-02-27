import numpy as np
from core import dens, lamb_A_lprior, loglike_A, lpost, prior_sum, psd, tot_psd

from bnpc.signal.utils import signal_density, signal_prior_sum

"""
This file contains the functions for updating the signal parameters.
"""


def S_and_prisum(self, b_val, g_val, psi_val, ind):
    """
    Calculate total PSD (S) and prior sum for signal parameters
    :param b_val: log amplitude
    :param g_val: slope
    :param psi_val: psi
    :param ind: index
    :return: total PSD and prior sum
    """
    sig = signal_density(b_val, g_val, psi_val, self.f, self.signal_model)
    noise_A = psd(
        dens(self.logpsplines_A.lam_mat[ind, :], self.logpsplines_A.splines.T),
        Spar=self.Spar_A,
        modelnum=self.modelnum,
    )  # noise PSD of A channel
    prisum = prior_sum(
        lamb_lpri_A=lamb_A_lprior(
            self.logpsplines_A.lam_mat[ind, :],
            self.logpsplines_T.lam_mat[: ind + 1, :],
            self.logpsplines_A.P,
            self.k,
        ),
        sig_prior_sum=signal_prior_sum(
            b_val, g_val, psi_val, self.signal_model
        ),
    )
    S = tot_psd(noise_A, sig)
    return S, prisum


def update_signal_param(self, ind):
    """
    Updates signal parameters using the current state in self.

    Parameters:
    ind: int
        Index for updating the parameters.

    Returns:
    Updated values of b, g, and psi (if applicable).
    """

    if self.signal_model == 2:
        self.psi[ind] = np.random.normal(self.signal.psi[ind - 1], 1)
    else:
        # Sample `b` and `g` from a reflective normal distribution
        self.b[ind] = np.random.normal(self.signal.b[ind - 1], 0.1)
        self.g[ind] = np.random.normal(self.signal.g[ind - 1], 0.1)

    S, prisum = S_and_prisum(
        self,
        self.signal.b[ind - 1],
        self.signal.g[ind - 1],
        self.signal.psi[ind - 1],
        ind,
    )
    ftheta = lpost(loglike_A(A=self.A, E=self.E, S=S), prisum)

    # Calculate `S` and `prisum` for the current proposed values of `b`, `g`, and `psi`
    S, prisum = S_and_prisum(
        self, self.signal.b[ind], self.signal.g[ind], self.signal.psi[ind], ind
    )
    ftheta_star = lpost(loglike_A(A=self.A, E=self.E, S=S), prisum)

    # Acceptance or rejection
    fac = min(0, ftheta_star - ftheta)
    if fac is np.nan:
        fac = -1000

    if np.log(np.random.rand()) > fac:
        # Reject the proposal; revert `b`, `g`, and possibly `psi`
        self.signal.b[ind] = self.signal.b[ind - 1]
        self.signal.g[ind] = self.signal.g[ind - 1]
        if self.signal_model == 2:
            self.signal.psi[ind] = self.signal.psi[ind - 1]
