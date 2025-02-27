#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:22:54 2024

"""

#code used in pyslipper to 
import numpy as np
from scipy.optimize import minimize
from .core import loglike, psd, dens
import logging
logger = logging.getLogger(__name__)
'''
This file contains the functions for optimizing the starting weights.
'''
def __optimize_ith_x(splines:np.ndarray, Spar: np.ndarray, modelnum:int, data:np.ndarray, x:np.ndarray, i:int, xi:float)->float:
    x_new = x.copy()
    x_new[i] = xi
    S = psd(dens(x_new, splines), Spar=Spar, modelnum=modelnum)
    return -loglike(data, S)


def __optimization_loop(splines:np.ndarray, Spar:np.ndarray, modelnum:int, data:np.ndarray, x:np.ndarray, kwgs, fast_optim:bool=False):
    def optim_all(_x):
        S = psd(dens(_x, splines), Spar=Spar, modelnum=modelnum)
        return -loglike(data, S)

    # optim_all = lambda _x: -pspline.lnlikelihood(data, **{xkey: _x})
    x = minimize(optim_all, x0=x, **kwgs).x
    S = psd(dens(x, splines), Spar=Spar, modelnum=modelnum)
    current_lnl = loglike(data, S)    
    logger.debug(f"LnL={current_lnl:.3E}")

    if fast_optim:
        return x  # optimized enough
    
    for i in range(len(x)):
            optim_ith = lambda _xi: __optimize_ith_x(
                splines, Spar, modelnum, data, x, i, _xi
            )
            x[i] = minimize(optim_ith, x0=x[i], **kwgs).x
            S = psd(dens(x, splines), Spar=Spar, modelnum=modelnum)
            current_lnl = loglike(data, S) 
            logger.debug(f"LnL={current_lnl:.3E}")

    x = minimize(optim_all, x0=x, **kwgs).x
    S = psd(dens(x, splines), Spar=Spar, modelnum=modelnum)
    current_lnl = loglike(data, S) 
    logger.debug(f"LnL={current_lnl:.3E}")

    return x



def optimize_starting_weights(
    splines:np.ndarray,
    Spar:np.ndarray,
    modelnum:int,
    n_basis:int,
    data:np.ndarray,
    init_x:np.ndarray,
    n_optimization_steps:int=100,
):
    '''
    Optimize the starting weights for the pspline model
    :param splines: Spline basis
    :param Spar: Parametric model
    :param modelnum: PSD model
    :param n_basis: Number of basis functions
    :param data: Data
    :param init_x: Initial weights
    :param n_optimization_steps: optimization steps
    :return: Optimized weights
    '''

    # ignore the 1st and last datapoints (there are often issues with the start/end points)
    data = data[1:-1]# T channel data
    n = len(data)

    # orig_grid_n = params.n_grid_points
    # params.n_grid_points = n
    
    
    noise=psd(dens(init_x, splines), Spar=Spar, modelnum=modelnum)
    init_lnl = loglike(data, noise)
    kwgs = dict(
        options=dict(
            maxiter=n_basis * n_optimization_steps,
            xatol=1e-30,
            gtol=1e-30,
            disp=False,
            adaptive=True,
            return_all=False,
        ),
        # bounds=bounds,
        tol=1e-50,
        method="L-BFGS-B",
    )

    x = __optimization_loop(
        splines, Spar, modelnum, data, init_x, kwgs, fast_optim=False
    )
    logger.debug(f"Optimized parameters: {x}")

    return x.ravel()



