import numpy as np
from ..utils import updateCov
from scipy.stats import multivariate_normal
from .core import lpost, loglike, prior_sum, signal_prior_sum, spline_prior_sum, log_noise_psd, noises, tot_psd, nxcondi


def llike_cal(self,npsdNx,npsdNxy, s_s):
    npsdT, npsdA = noises(npsdNx, npsdNxy )
    totpsd = tot_psd(npsdA,  s_s)
    llike = loglike(
        A=self.A, E=self.E, T=self.T, S=totpsd, s_n=npsdT
    )
    return llike

def calc_likelihood_nx(self, lam: np.ndarray, i: int) -> float:
    _,npsdNx = log_noise_psd(lam=lam,splines=self.logpsplines_x.splines.T,Spar=self.Spar_x,modelnum=self.modelnum)
    llike = llike_cal(self, npsdNx=npsdNx, npsdNxy=self.npsdNxy[i, :], s_s= self.signal.s_s[i, :])
    return llike


def calc_likelihood_nxy(self, lam: np.ndarray, i: int) -> float:
    _,npsdNx = log_noise_psd(lam=self.logpsplines_x.lam_mat[i+1, :], splines=self.logpsplines_x.splines.T, Spar=self.Spar_x, modelnum=self.modelnum)
    _,npsdNxy = log_noise_psd(lam=lam, splines=self.logpsplines_xy.splines.T, Spar=self.Spar_xy, modelnum=self.modelnum)
    llike = llike_cal(self, npsdNx=npsdNx, npsdNxy=npsdNxy, s_s=self.signal.s_s[i, :])
    return llike


def calc_prior_nx(self, lam: np.ndarray, i: int) -> float:
    prisum = prior_sum(
        splines_x_prior_sum=spline_prior_sum(lam=lam,
                     phi=self.logpsplines_x.phi[i],
                     delta=self.logpsplines_x.delta[i],
                     P=self.logpsplines_x.P, k=self.k
                     ),
        splines_xy_prior_sum=spline_prior_sum(lam=self.logpsplines_xy.lam_mat[i, :],
                                       phi=self.logpsplines_xy.phi[i],
                                       delta=self.logpsplines_xy.delta[i],
                                       P=self.logpsplines_xy.P, k=self.k
                                       ),
        signal_prior_sum= signal_prior_sum(
        self.signal.b[i],
        self.signal.g[i],
        self.signal.psi[i],
        self.signal_model,
    ),
    )
    return prisum

def calc_prior_nxy(self, lam: np.ndarray, i: int) -> float:
    prisum = prior_sum(
        splines_x_prior_sum=spline_prior_sum(lam=self.logpsplines_xy.lam_mat[i+1, :],
                     phi=self.logpsplines_x.phi[i],
                     delta=self.logpsplines_x.delta[i],
                     P=self.logpsplines_x.P, k=self.k
                     ),
        splines_xy_prior_sum=spline_prior_sum(lam=lam,
                                       phi=self.logpsplines_xy.phi[i],
                                       delta=self.logpsplines_xy.delta[i],
                                       P=self.logpsplines_xy.P, k=self.k
                                       ),
        signal_prior_sum= signal_prior_sum(
        self.signal.b[i],
        self.signal.g[i],
        self.signal.psi[i],
        self.signal_model,
    ),
    )
    return prisum

def l_p(self, lam: np.ndarray, ind: int, log_nxy_loop: bool) -> tuple[float, float]:
    if log_nxy_loop:
        llike = calc_likelihood_nxy(self, lam, ind)
        prisum = calc_prior_nxy(self, lam, ind)
        return llike, prisum
    llike = calc_likelihood_nx(self, lam, ind)
    prisum = calc_prior_nx(self, lam, ind)
    return llike, prisum


def posterior_cal(self, lam: np.ndarray, post_i: int, log_nxy_loop: bool) -> float:
    """
    Posterior calculation
    :param lam: lambda vector
    :param post_i: posterior index
    :return: posterior value
    """
    llike, prisum = l_p(self, lam=lam, ind=post_i, log_nxy_loop=log_nxy_loop)
    ftheta = lpost(llike, prisum)
    return ftheta


def lam_condition(self, i:int, lam:np.ndarray, covObj:np.ndarray, log_nxy_loop: bool):
    # updating lambda in a way to make sure that the noises are always defined
    # log(Nx)>log(2)+log(Nxy)
    def get_noises(lam_nx, lam_nxy):
        _,npsdNx = log_noise_psd(lam=lam_nx, splines=self.logpsplines_x.splines.T,
                               Spar=self.Spar_x, modelnum=self.modelnum)
        _,npsdNxy = log_noise_psd(lam=lam_nxy, splines=self.logpsplines_xy.splines.T, Spar=self.Spar_xy,
                                modelnum=self.modelnum)
        return npsdNx, npsdNxy

    def get_noises_vals(lam_val):
        if log_nxy_loop:
            return get_noises(self.logpsplines_x.lam_mat[i, :], lam_val)

        return get_noises(lam_val, self.logpsplines_xy.lam_mat[i - 1, :])

    def get_lamstar():
        if (self.Uv_am[i] < 0.05) or (i <= 2*self.k):
             lam_star = multivariate_normal.rvs(mean=lam, cov=self.Ik, size=1)
        else:
             lam_star = multivariate_normal.rvs(mean=lam, cov=self.c_amh * covObj, size=1)
        return lam_star

    lam_star=get_lamstar()
    logNx, lognegNxy = get_noises_vals(lam_star)
    while_break = 0
    # making sure the condition is satisfied:
    while nxcondi(logNx, lognegNxy):
        lam_star=get_lamstar()
        logNx, lognegNxy = get_noises_vals(lam_star)

        while_break += 1
        if while_break > 1000:
            np.savetxt('/home/naim769/oneMonth/pcode/lisatest/signal/exact/cut/figures/exit.txt',
                       np.array([[f'{i} exceeded 1000 iterations in while, returning previous value for lambda']]),
                       fmt='%s')
            print(i)
            return lam
    return lam_star


def lam_loop(self, covObj,  i:int, log_nxy_loop: bool)->np.ndarray:
    '''
    Updates lambda vector using Adaptive Metropolis-Hastings (AMH).
    :param self: Object of T channel and Data
    :param i: index
    :return: lambda vector
    '''
    lam = (self.logpsplines_xy.lam_mat[i, :] if log_nxy_loop else self.logpsplines_x.lam_mat[i, :])
    ftheta = posterior_cal(self, lam=lam, post_i=i, log_nxy_loop=log_nxy_loop)
    lam_star = lam_condition(self, i, lam, covObj, log_nxy_loop)
    ftheta_star = posterior_cal(self, lam=lam_star, post_i=i, log_nxy_loop=log_nxy_loop)

    # Acceptance or rejection
    fac = np.min([0, ftheta_star - ftheta])
    if self.Uv[i] > fac:
        return lam
    return lam_star


def update_lambda_fun(self, lam_mat, covObj,  i:int, log_nxy_loop: bool=False):
    '''
    This function updates lambda vector and covariance matrix for Nx and Nxy channel using Adaptive Metropolis-Hastings (MH)
    :param covObj: covariance matrix
    :param i: index
    :param log_nxy_loop: condition for Nx and Nxy channel
    :return: lambda vector and covariance matrix
    '''
    lam_mat[i, :] = lam_loop(self, covObj=covObj, i=i - 1, log_nxy_loop=log_nxy_loop)
    covObj = updateCov(lam_mat, covObj, i)
    return lam_mat[i, :], covObj


def update_lambda(self, i: int):
    """
    Updates lambda vector for Nx and Nxy channels using Adaptive Metropolis-Hastings (MH).
    :param i: index
    """
    self.logpsplines_x.lam_mat[i, :], self.logpsplines_x.covObj= update_lambda_fun(self, lam_mat=self.logpsplines_x.lam_mat, covObj= self.logpsplines_x.covObj, i=i) #updating lambda for Nx
    self.logpsplines_xy.lam_mat[i, :], self.logpsplines_xy.covObj= update_lambda_fun(self, lam_mat=self.logpsplines_xy.lam_mat, covObj= self.logpsplines_xy.covObj, i=i, log_nxy_loop=True) #updating lambda for Nxy
    