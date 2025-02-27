import numpy as np

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


def updata_phi_A(lam_mat, P):
    std_lam = np.std(lam_mat, axis=0)
    std_lam[std_lam == 0] = 1e-16
    diag_p_inv = np.diag(inverse(P))  # diagonal elements of inverse of P
    phi = np.sqrt(diag_p_inv) / std_lam
    return np.diag(phi)


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
