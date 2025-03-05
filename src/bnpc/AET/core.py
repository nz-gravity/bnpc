import numpy as np
from scipy.stats import gamma

from bnpc.signal.utils import signal_prior_sum

from ..utils import determinant, inverse

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
def loglike(pdgrm: np.ndarray, S: np.ndarray) -> float:
    """
    likelihood for one channel
    :param pdgrm: Periodogram
    :param S: Estimated PSD
    :return: log likelihood
    """
    lnlike = -1 * np.sum(S + np.exp(np.log(pdgrm) - S))
    return lnlike


def loglike_A(A: np.ndarray, E: np.ndarray, S: np.ndarray) -> float:
    """
    likelihood for A and E channels
    :param A: A periodogram
    :param E: E periodogram
    :param S: Estimated PSD in A and E
    :return: log likelihood
    """
    lnlike = loglike(A, S) + loglike(E, S)
    return lnlike


def loglike_AET(
    A: np.ndarray, E: np.ndarray, T: np.ndarray, S: np.ndarray, s_n: np.ndarray
) -> float:
    # log likelihood for three channels
    # A and E contains the signal. However, T does not contain the signal
    lnlike = loglike_A(A, E, S) + loglike(T, s_n)
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

def updata_phi_A(lam_mat, P):
    std_lam = np.std(lam_mat, axis=0)
    std_lam[std_lam == 0] = 1e-16
    diag_p_inv = np.diag(inverse(P))  # diagonal elements of inverse of P
    phi = np.sqrt(diag_p_inv) / std_lam
    return np.diag(phi)


def lamb_A_lprior(
    lam_A: np.ndarray, lam_mat: np.ndarray, P_A: np.ndarray, k: int
) -> float:
    """
    Prior for lambda in A channel. This is based on the standard daviation in T channel weights. In this way, we are restricting the A channel weights using
    the variability in T channel weights.
    :param lam_A: lambda A vector
    :param lam_mat: lambda matrix
    :param P_A: Panelty matrix for A channel
    :param k: number of weights
    :return: log prior
    """
    phi = updata_phi_A(lam_mat, P_A)
    mean_vec = np.mean(lam_mat, axis=0)
    mult = phi @ P_A @ phi
    prior = (
        -k * np.log(determinant(mult)) / 2
        - np.matmul(
            np.transpose(lam_A - mean_vec), np.matmul(mult, (lam_A - mean_vec))
        )
        / 2
    )
    if prior is np.inf:
        return -np.inf
    return prior


def phi_lprior(phi: float, delta: float) -> float:
    return gamma.logpdf(phi, a=1, scale=1 / delta)


def delta_lprior(delta: float) -> float:
    return gamma.logpdf(delta, a=1e-4, scale=1 / 1e-4)


def prior_sum(
    lamb_lpri: float = 0,
    phi_lpri: float = 0,
    delta_lpri: float = 0,
    lamb_lpri_A: float = 0,
    sig_prior_sum: float = 0,
) -> float:
    return lamb_lpri + phi_lpri + delta_lpri + lamb_lpri_A + sig_prior_sum


def lpost(loglike: float, lpriorsum: float) -> float:
    _lpost = loglike + lpriorsum
    if np.isnan(_lpost):
        raise ValueError("log posterior is nan")
    return _lpost


def llike_prisum_psd(obj, ind):
    obj.logpsplines_T.splines_mat[ind, :] = dens(
        lam=obj.logpsplines_T.lam_mat[ind, :],
        splines=obj.logpsplines_T.splines.T,
    )  # Spline PSD of T channel
    obj.npsdT[ind, :] = psd(
        Snpar=obj.logpsplines_T.splines_mat[ind, :],
        Spar=obj.Spar,
        modelnum=obj.modelnum,
    )  # Noise PSD of T channel
    obj.llike[ind] = loglike(
        pdgrm=obj.T, S=obj.npsdT[ind, :]
    )  # Log likelihood of T channel
    obj.prisum[ind] = prior_sum(
        lamb_lpri=lamb_lprior(
            obj.logpsplines_T.lam_mat[ind, :],
            obj.logpsplines_T.phi[ind],
            obj.logpsplines_T.P,
            obj.k,
        ),
        phi_lpri=phi_lprior(
            obj.logpsplines_T.phi[ind], obj.logpsplines_T.delta[ind]
        ),
        delta_lpri=delta_lprior(obj.logpsplines_T.delta[ind]),
    )  # Prior sum of T channel
    if obj.A is not None:
        obj.logpsplines_A.splines_mat[ind, :] = dens(
            lam=obj.logpsplines_A.lam_mat[ind, :],
            splines=obj.logpsplines_A.splines.T,
        )

        obj.npsdA[ind, :] = psd(
            Snpar=obj.logpsplines_A.splines_mat[ind, :],
            Spar=obj.Spar_A,
            modelnum=obj.modelnum,
        )  # Noise PSD of A channel
        obj.totpsd[ind, :] = tot_psd(
            s_n=obj.npsdA[ind, :], s_s=obj.signal.s_s[ind, :]
        )  # Total PSD

        obj.llike[ind] += loglike_A(A=obj.A, E=obj.E, S=obj.totpsd[ind, :])

        obj.prisum[ind] += prior_sum(
            lamb_lpri_A=lamb_A_lprior(
                obj.logpsplines_A.lam_mat[ind, :],
                obj.logpsplines_T.lam_mat[: ind + 2, :],
                obj.logpsplines_A.P,
                obj.k,
            ),
            sig_prior_sum=signal_prior_sum(
                obj.signal.b[ind],
                obj.signal.g[ind],
                obj.signal.psi[ind],
                obj.signal_model,
            ),
        )

    obj.logpost[ind] = lpost(obj.llike[ind], obj.prisum[ind])  # posterior
