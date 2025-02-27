"""
This module contains the data object for the log-spline model.
"""


class LogSplineData:
    def __init__(
        self,
        data,
        Spar,
        n,
        n_knots,
        degree,
        f,
        data_bin_edges,
        data_bin_weights,
        log_data,
        equidistant,
    ):
        self.data = data
        self.Spar = Spar
        self.n = n
        self.n_knots = n_knots
        self.degree = degree
        self.f = f
        self.data_bin_edges = data_bin_edges
        self.data_bin_weights = data_bin_weights
        self.log_data = log_data
        self.equidistant = equidistant
