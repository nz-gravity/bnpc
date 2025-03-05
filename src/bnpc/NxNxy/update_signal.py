import numpy as np

from bnpc.signal.utils import signal_density, signal_prior_sum

from .core import (
    loglike,
    lpost,
    prior_sum,
    tot_psd,
    spline_prior_sum,
)

"""
This file contains the functions for updating the signal parameters.
"""




def PSD_prisum(
    self,
    b_val,
    g_val,
    psi_val,
    ind,
):
    """
    Calculate total PSD and prior sum for signal parameters
    :param b_val: log amplitude
    :param g_val: slope
    :param psi_val: psi
    :param ind: index
    :return: total PSD and prior sum
    """
    sig = signal_density(b_val, g_val, psi_val, self.f, self.signal_model)
    S = tot_psd(s_n=self.npsdA[ind, :], s_s=sig)  # Total PSD
    prisum = prior_sum(
        splines_x_prior_sum=spline_prior_sum(lam=self.logpsplines_x.lam_mat[ind, :],
                     phi=self.logpsplines_x.phi[ind],
                     delta=self.logpsplines_x.delta[ind],
                     P=self.logpsplines_x.P, k=self.k
                     ),
        splines_xy_prior_sum=spline_prior_sum(lam=self.logpsplines_xy.lam_mat[ind, :],
                                       phi=self.logpsplines_xy.phi[ind],
                                       delta=self.logpsplines_xy.delta[ind],
                                       P=self.logpsplines_xy.P, k=self.k
                                       ),
        signal_prior_sum= signal_prior_sum(
            b_val, g_val, psi_val, self.signal_model
        ),
    )
    llike=loglike(A=self.A, E=self.E, T=self.T, S=S, s_n=self.npsdT[ind, :])
    return S, prisum, llike

def update_signal_param(self, ind):
    """
    Updates signal parameters.

    Parameters:
    ind: int
        Index for updating the parameters.

    Returns:
    Updated values of b, g, and psi (if applicable).
    """


    # Sample `b` and `g` from a reflective normal distribution
    self.signal.b[ind] = np.random.normal(self.signal.b[ind - 1], 0.05)
    self.signal.g[ind] = np.random.normal(self.signal.g[ind - 1], 0.05)
    psi_val = None
    if self.signal_model == 2:
        self.signal.psi[ind] = np.random.normal(self.signal.psi[ind - 1], 1)
        psi_val = self.signal.psi[ind - 1]

    S, prisum, llike = PSD_prisum(
        self,
        b_val=self.signal.b[ind - 1],
        g_val=self.signal.g[ind - 1],
        psi_val=psi_val,
        ind=ind,
    )
    ftheta = lpost(llike, prisum)


    if self.signal_model == 2:
        psi_val=self.signal.psi[ind]
    # Calculate `S` and `prisum` for the current proposed values of `b`, `g`, and `psi`
    S, prisum, llike = PSD_prisum(
        self, b_val=self.signal.b[ind], g_val=self.signal.g[ind], psi_val=psi_val, ind=ind
    )

    ftheta_star = lpost(llike, prisum)

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
