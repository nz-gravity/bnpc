import numpy as np
from scipy.stats import gamma
from ..logSplines import update_delta, update_phi
from bnpc.signal.utils import signal_prior_sum, signal_density

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

def log_noise_psd(lam, splines, Spar, modelnum):
    splines_psd = dens(
        lam=lam,
        splines=splines,
    )
    npsd = psd(
        Snpar=splines_psd,
        Spar=Spar,
        modelnum=modelnum,
    )
    if any(np.isnan(splines_psd)):
        raise ValueError("log spline PSD is nan")
    if any(np.isnan(npsd)):
        raise ValueError("log noise PSD is nan")

    return splines_psd, npsd

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
    '''
    This function calculates the log of sum using logsumexp technique
    :param logNx: log density of Nx
    :param lognegNxy: log of negative density of Nxy
    :return: log of sum
    '''
    sth = logNx > lognegNxy
    S = np.where(
        sth,
        logNx + np.log(1 + np.exp(lognegNxy - logNx)),
        lognegNxy + np.log(1 + np.exp(logNx - lognegNxy)),
    )
    return S


# A channel noise
def N_A(logNx, lognegNxy):
    '''
    This function calculates the noise PSD of A channel
    :param logNx: log density of Nx
    :param lognegNxy: log of negative density of Nxy
    :return: noise PSD of A channel
    '''
    return np.log(2 / 3) + logsumexp(logNx, lognegNxy)


# T channel noise
def N_T(logNx, lognegNxy):
    '''
    This function calculates the noise PSD of T channel
    :param logNx: log density of Nx
    :param lognegNxy: log of negative density of Nxy
    :return: noise PSD of T channel
    '''
    logNxy2 = np.log(2) + lognegNxy
    return np.log(1 / 3) + logdiffexp(logNx, logNxy2)


def noises(logNx, lognegNxy):
    noise_T=N_T(logNx, lognegNxy)
    noise_A=N_A(logNx, lognegNxy)
    if any(np.isnan(noise_T)):
        raise ValueError("log PSD of T channel is nan")
    if any(np.isnan(noise_A)):
        raise ValueError("log PSD of A channel is nan")
    return noise_T, noise_A

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
    lnlike = loglike1(A, S) + loglike1(E, S) + loglike1(T, s_n)
    if np.isnan(lnlike):
        raise ValueError("log likelihood is nan")
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
    res=k * np.log(phi) / 2 - phi * np.matmul(np.transpose(lam), np.matmul(P, lam)) / 2
    if np.isnan(res):
        raise ValueError("lambda log prior is nan")

    return res


def phi_lprior(phi: float, delta: float) -> float:
    '''
    This function calculates the prior of phi
    :param phi: phi
    :param delta: delta
    :return: prior of phi
    '''
    res=gamma.logpdf(phi, a=1, scale=1 / delta)
    if np.isnan(res):
        raise ValueError("phi log prior is nan")
    return res


def delta_lprior(delta: float) -> float:
    '''
    this function calculates the prior of delta
    :param delta: delta
    :return: prior of delta
    '''
    res=gamma.logpdf(delta, a=1e-4, scale=1 / 1e-4)
    if np.isnan(res):
        raise ValueError("delta log prior is nan")
    return res



def lpost(loglike: float, lpriorsum: float) -> float:
    '''
    This function calculates the log posterior
    :param loglike: log likelihood
    :param lpriorsum: log prior sum
    :return: log posterior
    '''
    _lpost = loglike + lpriorsum
    if np.isnan(_lpost):
        raise ValueError("log posterior is nan")
    return _lpost


def p_s_spline(
        lamb_lpri: float,
        phi_lpri: float,
        delta_lpri: float
) -> float:
    '''
    This function calculates the prior sum of splines
    :param lamb_lpri: Lambda log prior
    :param phi_lpri: phi log prior
    :param delta_lpri: delta log prior
    :return: prior sum of splines
    '''
    return lamb_lpri + phi_lpri + delta_lpri

def spline_prior_sum(lam, phi, delta, P, k):
    '''
    This function calls the prior sum of splines
    :param lam: lambda
    :param phi: phi
    :param delta: delta
    :param P: Panelty matrix
    :param k: number of weights
    :return: prior sum of splines
    '''
    return p_s_spline(
        lamb_lpri=lamb_lprior(
            lam=lam,
            phi=phi,
            P=P,
            k=k,
        ),
        phi_lpri=phi_lprior(
            phi=phi,
            delta=delta,
        ),
        delta_lpri=delta_lprior(
            delta=delta
        ),
    )

def prior_sum(
        splines_x_prior_sum: float,
        splines_xy_prior_sum: float,
        signal_prior_sum: float,
) -> float:
    '''
    This function calculates the prior sum
    :param splines_x_prior_sum: prior sum of splines of Nx
    :param splines_xy_prior_sum: prior sum of splines of Nxy
    :param signal_prior_sum: prior sum of signal
    :return:
    '''
    return splines_x_prior_sum + splines_xy_prior_sum + signal_prior_sum


def noise_psd_cal(obj, ind):
    '''
    This function calculates the noise PSD of Nx, Nxy, T and A channels
    :param obj: Object containing all the parameters
    :param ind: index
    :return: Noise PSD of Nx, Nxy, T and A channels
    '''
    obj.logpsplines_x.splines_mat[ind, :] , obj.npsdNx[ind, :] = log_noise_psd(lam=obj.logpsplines_x.lam_mat[ind, :],
                                                                               splines=obj.logpsplines_x.splines.T,
                                                                               Spar=obj.Spar_x, modelnum=obj.modelnum)

    obj.logpsplines_xy.splines_mat[ind, :] , obj.npsdNxy[ind, :] = log_noise_psd(lam=obj.logpsplines_xy.lam_mat[ind, :],
                                                                               splines=obj.logpsplines_xy.splines.T,
                                                                               Spar=obj.Spar_xy, modelnum=obj.modelnum)

    obj.npsdT[ind, :], obj.npsdA[ind, :]  = noises(obj.npsdNx[ind, :], obj.npsdNxy[ind, :])  # Noise PSD of T and A channels

def tot_sig_psd_cal(obj, ind):
    '''
    This function calculates total noise and signal PSD
    :param obj: Object containing all the parameters
    :param ind: index
    :return: Total PSD and signal PSD
    '''
    obj.signal.s_s[ind,:]= signal_density(b=obj.signal.b[ind], g=obj.signal.g[ind], psi=obj.signal.psi[ind], f=obj.f, signal_model=obj.signal_model)
    obj.totpsd[ind, :] = tot_psd(obj.npsdA[ind, :], obj.signal.s_s[ind,:])


def llike_prisum_lpost(obj, ind):
    '''
    This function calculates log likelihood, prior sum and log posterior
    :param obj: object containing all the parameters
    :param ind: index
    :return: log likelihood, prior sum and log posterior
    '''
    obj.llike[ind] = loglike(
        A=obj.A, E=obj.E, T=obj.T, S=obj.totpsd[ind, :], s_n=obj.npsdT[ind, :]
    )

    #calculating the prior sums:
    obj.prisum[ind] = prior_sum(
        splines_x_prior_sum=spline_prior_sum(lam=obj.logpsplines_x.lam_mat[ind, :],
                                         phi=obj.logpsplines_x.phi[ind],
                                         delta=obj.logpsplines_x.delta[ind],
                                         P=obj.logpsplines_x.P, k=obj.k
                                         ),
        splines_xy_prior_sum=spline_prior_sum(lam=obj.logpsplines_xy.lam_mat[ind, :],
                                         phi=obj.logpsplines_xy.phi[ind],
                                         delta=obj.logpsplines_xy.delta[ind],
                                         P=obj.logpsplines_xy.P, k=obj.k
                                         ),
        signal_prior_sum=signal_prior_sum(
            obj.signal.b[ind],
            obj.signal.g[ind],
            obj.signal.psi[ind],
            obj.signal_model,
        ),
    )
    obj.logpost[ind] = lpost(obj.llike[ind], obj.prisum[ind])  # posterior

def update_phi_delta(obj,ind):
    '''
    This function updates the phi and delta
    :param obj: object containing all the parameters
    :param ind: index
    :return: phi and delta
    '''
    obj.phi[ind] = update_phi(
        lam=obj.lam_mat[ind, :],
        P=obj.P,
        delta=obj.delta[ind - 1],
        a_phi=obj.a_phi,
    )

    # sample delta
    obj.delta[ind] = update_delta(
        phi=obj.phi[ind], a_delta=obj.a_delta
    )


def nxcondi(logNx:np.ndarray, lognegNxy:np.ndarray)-> bool:
    '''
    This function checks the condition for a defined NT channel log PSD
    :param logNx: log density of Nx
    :param lognegNxy: log of negative density of Nxy
    :return: true if the condition is satisfied
    '''
    if any(logNx <= np.log(2) + lognegNxy):
        return True
    return False

