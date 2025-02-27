import numpy as np
from skfda.representation.basis import BSplineBasis
from skfda.misc.operators import LinearDifferentialOperator
from skfda.misc.regularization import L2Regularization
from scipy.stats import gamma
'''
This file contains the utility functions for the splines.
'''

def diffMatrix(k:int, d:int=2) -> np.ndarray:
    '''
    Generate the difference matrix of order d
    :param k: number of weights
    :param d: order of the difference matrix
    :return: difference matrix
    '''
    out = np.eye(k)
    for i in range(d):
        out = np.diff(out, axis=0)
    return out

def generate_basis_matrix(knots:np.ndarray, grid_points:np.ndarray, degree:int, normalised: bool = True) -> np.ndarray:  # slipper pspline psd
        '''
        Generate the basis matrix for the given knots and grid points
        :param knots: knots vector
        :param grid_points: Grid points
        :param degree: Degree of the spline
        :param normalised: condition to normalize the basis functions
        :return: basis matrix
        '''
        basis = BSplineBasis(knots=knots, order=degree + 1).to_basis()
        basis_matrix = basis.to_grid(grid_points).data_matrix.squeeze().T

        if normalised:
            # normalize the basis functions
            knots_with_boundary = np.concatenate(
                [
                    np.repeat(knots[0], degree),
                    knots,
                    np.repeat(knots[-1], degree),
                ]
            )
            n_knots = len(knots_with_boundary)
            mid_to_end_knots = knots_with_boundary[degree + 1:]
            start_to_mid_knots = knots_with_boundary[
                                 : (n_knots - degree - 1)
                                 ]
            bs_int = (mid_to_end_knots - start_to_mid_knots) / (
                    degree + 1
            )
            bs_int[bs_int == 0] = np.inf
            basis_matrix = basis_matrix / bs_int
        return basis_matrix


def panelty_linear(k:int, d:int)->np.ndarray:
    '''
    Generate the penalty matrix for the given order and degree assuming linear basis
    :param k: number of weights
    :param d: difference matrix order
    :return: panelty matrix
    '''
    # linear
    P = diffMatrix(k, d)
    P = np.matmul(np.transpose(P), P)
    return (P)

def panelty_mat(d:int, knots:np.ndarray, degree:int=3, epsi:float=1e-6 , linear:bool=False, k:int=0)->np.ndarray:
    '''
    Generate the penalty matrix for the given knots and degree
    :param d: order of the difference matrix
    :param knots: knots vector
    :param degree: degree of the spline
    :param epsi: small value to avoid singular matrix
    :return: panelty matrix
    '''
    if linear:
        return panelty_linear(k, d)
    basis = BSplineBasis(knots=knots, order=degree + 1)
    regularization = L2Regularization(
        LinearDifferentialOperator(d)
    )
    p = regularization.penalty_matrix(basis)
    p / np.max(p)
    return p + epsi * np.eye(p.shape[1])


def update_phi(lam, P, delta, a_phi):
    """
    conditional posterior distribution of phi.

    Parameters:
    lam (array-like): Lambda vector.
    P (array-like): Panelty matrix.
    delta (float): delta.
    a_phi (float): Shape parameter for the gamma distribution.

    Returns:
    float: Updated phi value.
    """
    b_phi = 0.5 * np.matmul(np.transpose(lam), np.matmul(P, lam)) + delta
    phi_value = gamma.rvs(a=a_phi, scale=1 / b_phi, size=1)
    return phi_value


def update_delta(phi, a_delta):
    """
    conditional posterior distribution of delta.

    Parameters:
    phi (float): phi.
    a_delta (float): Shape parameter for the gamma distribution.

    Returns:
    float: Updated delta value.
    """
    b_delta = phi + 1e-4
    delta_value = gamma.rvs(a=a_delta, scale=1 / b_delta, size=1)
    return delta_value

