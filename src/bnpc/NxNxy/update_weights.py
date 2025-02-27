import numpy as np
from .utils import updateCov




def lam_loop(self,  i:int)->np.ndarray:
    '''
    Updates lambda vector for A channel using Adaptive Metropolis-Hastings (AMH).
    :param self: Object of T channel and Data
    :param i: index
    :return: lambda vector
    '''
    lam = self.logpsplines_A.lam_mat[i, :] #initial value
    ftheta = posterior_cal(self, lam=lam, post_i=i)
    if (self.Uv_am[i] < 0.05) or (i <= 2*self.k):
         lam_star = multivariate_normal.rvs(mean=lam, cov=self.logpsplines_A.Ik, size=1)
    else:
         lam_star = multivariate_normal.rvs(mean=lam, cov=self.logpsplines_A.c_amh * self.logpsplines_A.covObj, size=1)

    ftheta_star = posterior_cal(self, lam=lam_star, post_i=i)

    # Acceptance or rejection
    fac = np.min([0, ftheta_star - ftheta])
    if self.logpsplines_A.Uv[i] > fac:
        return lam
    return lam_star


def update_lambda_A(self,  i:int=0):
    """
    Updates lambda vector for A channel using Metropolis-Hastings (MH) with current parameter values.
    :param self: Object of T channel and Data
    :param i: index
    """
    self.logpsplines_A.lam_mat[i, :] = lam_loop(self, i=i - 1)
    self.logpsplines_A.covObj = updateCov(self.logpsplines_A.lam_mat, self.logpsplines_A.covObj, i)
