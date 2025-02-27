#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 21:29:57 2024

@author: naim769from

"""
import numpy as np
from core import dens, lpost, psd, tot_psd, loglike,loglike_A, prior_sum, lamb_lprior, lamb_A_lprior, phi_lprior, delta_lprior
from bnpc.signal.utils import signal_density, signal_prior_sum
from scipy.stats import multivariate_normal
from .utils import updateCov

def sigmaupdate(accept_frac:float, sigma:float)->float:
    if accept_frac < 0.30:
        sigma *= 0.90
    elif accept_frac > 0.50:
        sigma *= 1.10
    return sigma

def l_p_T(self, lam:np.ndarray, j:int):
    '''
    Log likelihood and prior sum for T channel
    :param lam: lambda vector
    :param j: index
    :return: log likelihood and prior sum
    '''
    S = psd(dens(lam, self.logpsplines_T.splines.T), Spar=self.Spar, modelnum=self.modelnum) #noise PSD of T channel
    prisum = prior_sum(lamb_lprior(lam, self.logpsplines_T.phi[j], self.P, self.k),
                       phi_lprior(self.logpsplines_T.phi[j], self.logpsplines_T.delta[j]),
                       delta_lprior(self.logpsplines_T.delta[j]))
    llike = loglike(self.T, S)#log likelihood
    return (llike, prisum)


def l_p_A(self, lam:np.ndarray, j:int):
    '''
    Log likelihood and prior sum for A channel
    :param lam: lambda vector
    :param j: index
    :param self: object of T channel and data
    :return: log likelihood and prior sum
    '''
    sig=signal_density(self.signal.b[j], self.signal.g[j], self.signal.psi[j], self.f, self.signal_model)
    prisum = prior_sum(lamb_lpri_A=lamb_A_lprior(lam, self.logpsplines_T.lam_mat[:j + 2, :], self.logpsplines_A.P, self.k),
                        sig_prior_sum=signal_prior_sum(self.signal.b[j], self.signal.g[j], self.signal.psi[j], self.signal_model))
    noise_A = psd(dens(lam, self.logpsplines_A.splines.T), Spar=self.Spar_A, modelnum=self.modelnum) #noise PSD of A channel
    S = tot_psd(noise_A, sig)# signal + noise PSD

    llike = loglike_A(A=self.A, E=self.E, S=S) #log likelihood

    return (llike, prisum)



def posterior_cal(self, lam:np.ndarray, post_i:int)->float:
    '''
    Posterior calculation
    :param lam: lambda vector
    :param post_i: posterior index
    :param self: object of T channel and data
    :return: posterior value
    '''
    if self.A is not None:
        llike, prisum = l_p_A(self, lam=lam, j=post_i)
    else:
        llike, prisum = l_p_T(self, lam=lam, j=post_i)

    ftheta = lpost(llike, prisum)
    return ftheta



def lam_loop(self, loop_index:int):
    '''
    Loop for updating lambda vector for T channel using Metropolis-Hastings (MH).
    :param loop_index: Loop index
    :return: lambda vector and acceptance fraction
    '''
    lam = self.logpsplines_T.lam_mat[loop_index, :]
    accept_count = 0
    aux = np.arange(0, self.k)
    np.random.shuffle(aux)
    for sth in range(0, len(lam)):
        u = np.log(np.random.uniform())
        pos = aux[sth]
        z = np.random.normal()
        lam_p = lam[pos]

        ftheta = posterior_cal(self, lam=lam, post_i=loop_index)

        lam_star = lam_p + self.sigma * z

        lam[pos] = lam_star

        ftheta_star = posterior_cal(self, lam=lam, post_i=loop_index)

        fac = np.min([0, ftheta_star - ftheta])

        if fac is np.nan:
            fac = -1000

        if u > fac:
            # Reject update
            lam[pos] = lam_p
        else:
            accept_count += 1

    accept_frac = accept_count / self.k
    return lam, accept_frac

def update_lambda_T(self,i:int):
    '''
    Updates lambda vector for T channel using Metropolis-Hastings (MH) with current parameter values.
    :param i: index
    :return: updated lambda vector
    '''
    self.logpsplines_T.sigma = sigmaupdate(self.logpsplines_T.accept_frac, self.logpsplines_T.sigma)
    self.logpsplines_T.lam_mat[i, :], self.logpsplines_T.accept_frac = lam_loop(self, loop_index=i - 1)

def lam_A_loop(self,  i:int)->np.ndarray:
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
    self.logpsplines_A.lam_mat[i, :] = lam_A_loop(self, i=i - 1)
    self.logpsplines_A.covObj = updateCov(self.logpsplines_A.lam_mat, self.logpsplines_A.covObj, i)


def update_lambda(self, i:int):
    '''
    Updates lambda vector for T and A channel using Metropolis-Hastings (MH) with current parameter values.
    :param i: index
    '''
    update_lambda_T(self, i)
    if self.A is not None:
        update_lambda_A(self, i)