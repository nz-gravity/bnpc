import numpy as np

from bnpc.signal.utils import signal_density, signal_prior_sum

from .core import (
    N_A,
    N_T,
    delta_lprior,
    dens,
    lamb_lprior,
    loglike,
    lpost,
    phi_lprior,
    prior_sum,
    psd,
    tot_psd,
)

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
    npsdNx = psd(
        Snpar=self.logpsplines_x.splines_mat[ind, :],
        Spar=self.Spar_x,
        modelnum=self.modelnum,
    )  # Noise PSD of T channel
    npsdNxy = psd(
        Snpar=self.logpsplines_xy.splines_mat[ind, :],
        Spar=self.Spar_xy,
        modelnum=self.modelnum,
    )  # Noise PSD of T channel
    npsdT = N_T(npsdNx, npsdNxy)  # Noise PSD of T channel

    npsdA = N_A(npsdNx, npsdNxy)  # Noise PSD of T channel

    S = tot_psd(s_n=npsdA, s_s=sig)  # Total PSD
    prisum = prior_sum(
        lamb_lpri=lamb_lprior(
            self.logpsplines_x.lam_mat[ind, :],
            self.logpsplines_x.phi[ind],
            self.logpsplines_x.P,
            self.k,
        ),
        phi_lpri=phi_lprior(
            self.logpsplines_x.phi[ind], self.logpsplines_x.delta[ind]
        ),
        delta_lpri=delta_lprior(self.logpsplines_x.delta[ind]),
        lamb_lpri_xy=lamb_lprior(
            self.logpsplines_xy.lam_mat[ind, :],
            self.logpsplines_xy.phi[ind],
            self.logpsplines_xy.P,
            self.k,
        ),
        phi_lpri_xy=phi_lprior(
            self.logpsplines_xy.phi[ind], self.logpsplines_xy.delta[ind]
        ),
        delta_lpri_xy=delta_lprior(self.logpsplines_xy.delta[ind]),
        sig_prior_sum=signal_prior_sum(
            b_val, g_val, psi_val, self.signal_model
        ),
    )
    return npsdT, S, prisum


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

    s_n, S, prisum = S_and_prisum(
        self,
        self.signal.b[ind - 1],
        self.signal.g[ind - 1],
        self.signal.psi[ind - 1],
        ind,
    )
    ftheta = lpost(loglike(A=self.A, E=self.E, T=self.T, S=S, s_n=s_n), prisum)

    # Calculate `S` and `prisum` for the current proposed values of `b`, `g`, and `psi`
    s_n, S, prisum = S_and_prisum(
        self, self.signal.b[ind], self.signal.g[ind], self.signal.psi[ind], ind
    )

    ftheta_star = lpost(
        loglike(A=self.A, E=self.E, T=self.T, S=S, s_n=s_n), prisum
    )

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
