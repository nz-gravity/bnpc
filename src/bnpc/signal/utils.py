import numbers
from typing import Tuple, Union

import numpy as np
from bilby.core.prior import Uniform as uniform
from scipy.stats import norm


# LISA response to GW
def response(f: np.ndarray) -> np.ndarray:
    f_ref = 299792458 / (4 * 2.5e9)
    # f_ref = 299792458  / (2 * np.pi * 2.5e9)
    W_sq = (abs(1 - np.exp(-2j * f / f_ref))) ** 2
    R_A = 9 * W_sq * (1 / (1 + f / (4 * f_ref / 3))) / 20
    return R_A


def psd_sgwb(omega: np.ndarray, f: np.ndarray) -> np.ndarray:
    resp = response(f)
    h0 = 2.7e-18
    fact = np.log((3 * (h0**2) * resp) / (4 * (np.pi**2) * f**3))
    return omega + fact


def power_law(b: float, g: float, f: np.ndarray) -> np.ndarray:
    # log PSD for power law for SGWB
    y = b + g * np.log(f)
    return y


def brokenPower(
    amp: float, gam: float, psi: float, frequencies: np.ndarray
) -> np.ndarray:
    # log PSD for broken power law for SGWB
    f_ref = 0.002
    f_br = 0.002
    ome_gw = np.piecewise(
        frequencies,
        [frequencies <= f_br, frequencies > f_br],
        [
            lambda f: amp + gam * (np.log(f) - np.log(f_ref)),
            lambda f: amp
            + gam * (np.log(f_br) - np.log(f_ref))
            + psi * (np.log(f) - np.log(f_br)),
        ],
    )
    return ome_gw


##Signal priors
def b_lprior(b: float) -> float:
    """
    Prior for log amplitude of power law
    :param b: slope
    :return: log prior
    """
    return uniform(62.5, 63.1).ln_prob(b)


def glprior(g: float) -> float:
    """
    Prior for slope of power law
    :param g: log amplitude
    :return: log prior
    """
    return uniform(-0.68, -0.63).ln_prob(g)


def psilprior(psi: float) -> float:
    """
    Prior for psi (broken powerlaw parameter)
    :param psi: psi
    :return: log prior
    """
    return norm.logpdf(psi, loc=-4, scale=0.1)


def signal_density(
    b: float, g: float, psi: float, f: np.ndarray, signal_model: int = 1
) -> np.ndarray:
    """
    Signal density
    :param b: slope
    :param g: log amplitude
    :param psi: psi
    :param f: frequency
    :return: signal density
    """
    if signal_model == 2:
        return psd_sgwb(omega=brokenPower(b, g, psi, f), f=f)
    return psd_sgwb(omega=power_law(b, g, f), f=f)


def signal_prior_sum(
    b: float, g: float, psi: float, signal_model: int = 1
) -> float:
    """
    signal prior sum
    :param b: slope
    :param g: log amplitude
    :param psi: psi
    :param signal_model:
    :return: prior sum based on the signal model
    """
    if signal_model == 2:
        return b_lprior(b) + glprior(g) + psilprior(psi)
    return b_lprior(b) + glprior(g)
