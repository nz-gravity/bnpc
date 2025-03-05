from math import degrees

import matplotlib.pyplot as plt
import numpy as np

from .knot_allocation import knot_loc
from .utils import diffMatrix, generate_basis_matrix, panelty_mat

"""
This file contains the log Psplines class
"""


class logPsplines:
    """
    Log Psplines class
    """

    def __init__(self, dataobj):
        """
        Initialize the log Psplines object
        """
        self.knots = knot_loc(
            pdgrm=dataobj.data,
            Spar=dataobj.Spar,
            n_knots=dataobj.n_knots,
            degree=dataobj.degree,
            f=dataobj.f,
            data_bin_edges=dataobj.data_bin_edges,
            data_bin_weights=dataobj.data_bin_weights,
            log_data=dataobj.log_data,
            equidistant=dataobj.equidistant,
        )
        self.gridp = np.linspace(
            self.knots[0], self.knots[-1], len(dataobj.data)
        )
        self.splines = generate_basis_matrix(
            knots=self.knots, grid_points=self.gridp, degree=dataobj.degree
        )
        self.n_gridpoints, self.n_basis = self.splines.shape
        self.P = panelty_mat(
            d=1, knots=self.knots, degree=dataobj.degree
        )
        lam = dataobj.data / np.sum(dataobj.data)
        lam = lam[
            np.round(np.linspace(0, len(lam) - 1, dataobj.n_knots)).astype(int)
        ]
        lam[lam == 0] = 1e-50
        self.lam_mat = np.zeros((dataobj.n, dataobj.n_knots))
        self.lam_mat[0, :] = lam
        self.delta = np.zeros(dataobj.n)
        self.phi = np.zeros(dataobj.n)
        self.splines_mat = np.zeros(
            (dataobj.n, len(dataobj.data))
        )  # splines PSD
        self.delta[0] = 1
        self.phi[0] = 1
        self.a_phi = dataobj.n_knots / 2 + 1
        self.a_delta = 1 + 1e-4
        self.sigma = 1
        self.accept_frac = 0.4
