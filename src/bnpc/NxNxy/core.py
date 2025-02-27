import numpy as np
from scipy.stats import gamma

from bnpc.signal.utils import signal_prior_sum

from .utils import determinant, updata_phi_A

"""
This file contains the core functions for the algorithm.
"""


def dens(lam: np.ndarray, splines: np.ndarray) -> np.ndarray:
    """
    This function is defined to calculate the log of the density of the lambda
    :param lam: lambda
    :param splines: Splines
    :return: log density
    """
    return np.sum(lam[:, None] * splines, axis=0)


def psd(
    Snpar: np.ndarray, Spar: np.ndarray = 1, modelnum: int = 0
) -> np.ndarray:
    """
    models for log PSD of noise
    :param Snpar: Non-parametric log PSD (splines)
    :param Spar: Parametric log PSD
    :param modelnum: model number
    :return: Noise log PSD
    """

    if modelnum == 0:  # Only splines
        S = Snpar
    elif modelnum == 1:
        S = Snpar + np.log(Spar)
    elif modelnum == 4:
        S = 0.5 * Snpar + np.log(Spar)
    else:
        S = np.log(Spar) + Snpar * np.log(10)
    return S


def logdiffexp(a, b):
    """
    Computes log( e^a - e^b ) in a numerically stable way, elementwise.
    Returns -inf where a <= b (i.e., e^a - e^b <= 0 in normal space).
    a and b can be scalars or NumPy arrays of the same shape.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    # Make an output array
    out = np.empty_like(a, dtype=float)

    # Identify where a <= b
    mask = a <= b

    # Where a <= b, set output to -inf
    out[mask] = -np.inf

    # Where a > b, compute log(e^a - e^b)
    valid_mask = ~mask
    if np.any(valid_mask):
        # 1 - exp(b - a) can be close to 1 if (b-a) is very negative,
        # so there's no immediate risk of numerical blow-up, but it can underflow
        # if b is much larger than a. However, that corresponds to a <= b anyway.
        out[valid_mask] = a[valid_mask] + np.log(
            1.0 - np.exp(b[valid_mask] - a[valid_mask])
        )

    return out


def logsumexp(logNx, lognegNxy):
    sth = logNx > lognegNxy
    S = np.where(
        sth,
        logNx + np.log(1 + np.exp(lognegNxy - logNx)),
        lognegNxy + np.log(1 + np.exp(logNx - lognegNxy)),
    )
    return S


# A channel noise
def N_A(logNx, lognegNxy):
    return np.log(2 / 3) + logsumexp(logNx, lognegNxy)


# T channel noise
def N_T(logNx, lognegNxy):
    logNxy2 = np.log(2) + lognegNxy
    return np.log(1 / 3) + logdiffexp(logNx, logNxy2)


def tot_psd(s_n: np.ndarray, s_s: np.ndarray) -> np.ndarray:
    """
    This function is defined to calculate the log of sums of the PSD which is going to be used in the log likelihood
    :param s_n: log PSD of noise of A channel
    :param s_s: log PSD of signal
    :return:
    """
    sth = s_n > s_s
    S = np.where(
        sth,
        s_n + np.log(1 + np.exp(s_s - s_n)),
        s_s + np.log(1 + np.exp(s_n - s_s)),
    )
    return S


# Likelihood
def loglike1(pdgrm: np.ndarray, S: np.ndarray) -> float:
    """
    likelihood for one channel
    :param pdgrm: Periodogram
    :param S: Estimated PSD
    :return: log likelihood
    """
    lnlike = -1 * np.sum(S + np.exp(np.log(pdgrm) - S))
    return lnlike


def loglike(
    A: np.ndarray, E: np.ndarray, T: np.ndarray, S: np.ndarray, s_n: np.ndarray
) -> float:
    # log likelihood for three channels
    # A and E contains the signal. However, T does not contain the signal
    lnlike = loglike1(A, S) + loglike1(E, S) + loglike(T, s_n)
    return lnlike


# priors


def lamb_lprior(lam: np.ndarray, phi: float, P: np.ndarray, k: int) -> float:
    """
    Prior for lambda
    :param lam: lambda vector
    :param phi: phi
    :param P: Panelty matrix
    :param k: number of weights
    :return: log prior
    """
    return (
        k * np.log(phi) / 2
        - phi * np.matmul(np.transpose(lam), np.matmul(P, lam)) / 2
    )


def phi_lprior(phi: float, delta: float) -> float:
    return gamma.logpdf(phi, a=1, scale=1 / delta)


def delta_lprior(delta: float) -> float:
    return gamma.logpdf(delta, a=1e-4, scale=1 / 1e-4)


def prior_sum(
    lamb_lpri: float,
    phi_lpri: float,
    delta_lpri: float,
    lamb_lpri_xy: float,
    phi_lpri_xy: float,
    delta_lpri_xy: float,
    sig_prior_sum: float,
) -> float:
    return (
        lamb_lpri
        + phi_lpri
        + delta_lpri
        + lamb_lpri_xy
        + phi_lpri_xy
        + delta_lpri_xy
        + sig_prior_sum
    )


def lpost(loglike: float, lpriorsum: float) -> float:
    _lpost = loglike + lpriorsum
    if np.isnan(_lpost):
        raise ValueError("log posterior is nan")
    return _lpost


def llike_prisum_psd(obj, ind):
    obj.logpsplines_x.splines_mat[ind, :] = dens(
        lam=obj.logpsplines_x.lam_mat[ind, :],
        splines=obj.logpsplines_x.splines.T,
    )  # Spline PSD of T channel
    obj.npsdNx[ind, :] = psd(
        Snpar=obj.logpsplines_x.splines_mat[ind, :],
        Spar=obj.Spar_x,
        modelnum=obj.modelnum,
    )  # Noise PSD of T channel
    obj.npsdNxy[ind, :] = psd(
        Snpar=obj.logpsplines_xy.splines_mat[ind, :],
        Spar=obj.Spar_xy,
        modelnum=obj.modelnum,
    )  # Noise PSD of T channel
    obj.npsdT[ind, :] = N_T(
        obj.npsdNx[ind, :], obj.npsdNxy[ind, :]
    )  # Noise PSD of T channel

    obj.npsdA[ind, :] = N_A(
        obj.npsdNx[ind, :], obj.npsdNxy[ind, :]
    )  # Noise PSD of T channel

    obj.totpsd[ind, :] = tot_psd(
        s_n=obj.npsdA[ind, :], s_s=obj.signal.s_s[ind, :]
    )  # Total PSD

    obj.llike[ind] = loglike(
        A=obj.A, E=obj.E, T=obj.T, S=obj.totpsd[ind, :], s_n=obj.npsdT[ind, :]
    )  # Log likelihood of T channel

    obj.prisum[ind] = prior_sum(
        lamb_lpri=lamb_lprior(
            obj.logpsplines_x.lam_mat[ind, :],
            obj.logpsplines_x.phi[ind],
            obj.logpsplines_x.P,
            obj.k,
        ),
        phi_lpri=phi_lprior(
            obj.logpsplines_x.phi[ind], obj.logpsplines_x.delta[ind]
        ),
        delta_lpri=delta_lprior(obj.logpsplines_x.delta[ind]),
        lamb_lpri_xy=lamb_lprior(
            obj.logpsplines_xy.lam_mat[ind, :],
            obj.logpsplines_xy.phi[ind],
            obj.logpsplines_xy.P,
            obj.k,
        ),
        phi_lpri_xy=phi_lprior(
            obj.logpsplines_xy.phi[ind], obj.logpsplines_xy.delta[ind]
        ),
        delta_lpri_xy=delta_lprior(obj.logpsplines_xy.delta[ind]),
        sig_prior_sum=signal_prior_sum(
            obj.signal.b[ind],
            obj.signal.g[ind],
            obj.signal.psi[ind],
            obj.signal_model,
        ),
    )  # Prior sum
    obj.logpost[ind] = lpost(obj.llike[ind], obj.prisum[ind])  # posterior
