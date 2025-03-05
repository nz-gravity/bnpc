import numpy as np
from collections import namedtuple
from scipy.stats import median_abs_deviation

"""
This file contains the utility functions used in the MCMC.
"""


def get_pdgrm(blocked, A, T, re="T"):
    if blocked:
        if A is None:
            return T[0]
        else:
            if re == "T":
                return T[0]
            else:
                return A[0]
    else:
        if A is None:
            return T
        else:
            if re == "T":
                return T
            else:
                return A


def determinant(matrix, eps=1e-16):
    det_val = np.linalg.det(matrix)
    det_val = max(det_val, eps)
    return np.log(det_val)


def inverse(matrix):
    return np.linalg.pinv(matrix)

def updateCov(lambda_matrix, covObj, i, adapt_start=0, epsilon=1e-10):
    """
    Update the covariance matrix adaptively using all samples up to iteration i.

    """
    # Only adapt if we're past the `adapt_start` iteration
    if i < adapt_start:
        return covObj

    # param_chain[:i+1, :] is all accepted samples up to iteration i
    # rowvar=False means each row is an observation, columns are variables
    new_cov = np.cov(lambda_matrix[: i + 1, :].T, ddof=1)

    # Add small diagonal term for numerical stability (regularization)
    new_cov += epsilon * np.eye(new_cov.shape[0])

    return new_cov


def basic_summaries(parameter):
    """
    Returns mean, standard deviation, and 95% credible interval for a parameter array.
    """
    mean_val = np.mean(parameter)
    std_val = np.std(parameter)
    ci_lower, ci_upper = np.percentile(parameter, [2.5, 97.5])
    return mean_val, std_val, ci_lower, ci_upper


def mad(x):
    med_abs_dav = np.median(abs(x - np.median(x)))
    if med_abs_dav == 0:
        med_abs_dav = 1e-10
    return med_abs_dav


def uniformmax(sample):
    median = np.median(sample)
    mad1 = mad(sample)
    abs_deviation = np.abs(sample - median)

    normalized_deviation = abs_deviation / mad1
    max_deviation = np.nanmax(normalized_deviation)

    return max_deviation


def cent_series(series):
    return ((series - np.mean(series)) / np.std(series))


def compute_iae(psd, truepsd, n):  # note use PSD not log PSD
    return sum(abs(psd - truepsd)) * 2 * np.pi / n


def compute_prop(u05, u95, truepsd):
    v = []
    for x in range(len(u05)):
        if (truepsd[x] >= u05[x]) and (truepsd[x] <= u95[x]):
            v.append(1)
        else:
            v.append(0)
    return (np.mean(v))


def compute_ci(psds):
    CI = namedtuple('CI', ['u05', 'u95', 'med', 'label'])
    psd_help = np.apply_along_axis(uniformmax, 0, psds)
    psd_mad = median_abs_deviation(psds, axis=0)
    c_value = np.quantile(psd_help, 0.9)
    psd_med = np.median(psds, axis=0)
    psd_u95 = psd_med + c_value * psd_mad
    psd_u05 = psd_med - c_value * psd_mad
    return CI(u05=psd_u05, u95=psd_u95, med=psd_med, label='pypsd')

