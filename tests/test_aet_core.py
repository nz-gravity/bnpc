import numpy as np
import pytest

from bnpc.AET.core import (
    delta_lprior,
    lamb_A_lprior,
    lamb_lprior,
    loglike,
    loglike_A,
    lpost,
    phi_lprior,
    prior_sum,
    tot_psd,
)
from bnpc.signal.utils import b_lprior, glprior, power_law, psilprior


def test_lpost():
    assert lpost(1.0, 2.0) == 3.0


def test_tot_psd():
    s_n = np.array([0.0, 2.0, 1.0])
    s_s = np.array([1.0, 1.0, 2.0])
    expected = np.array([0.6867383, 2.3132617, 2.3132617])
    assert np.allclose(tot_psd(s_n, s_s), expected, atol=1e-6)


def test_loglike():
    pdgrm = np.array([10.0, 20.0])
    S = np.array([2.0, 3.0])
    expected = -7.349092
    assert pytest.approx(loglike(pdgrm, S), rel=1e-5) == expected


def test_loglike_A():
    A = np.array([10.0, 20.0])
    E = np.array([5.0, 15.0])
    S = np.array([2.0, 3.0])
    expected = loglike(A, S) + loglike(E, S)
    assert pytest.approx(loglike_A(A, E, S), rel=1e-5) == expected


def test_prior_sum():
    assert prior_sum(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0) == 28.0


def test_lamb_lprior():
    lam = np.array([1.0, 2.0])
    phi = 0.5
    P = np.array([[2.0, 0.0], [0.0, 2.0]])
    k = len(lam)
    expected = -3.193147
    assert pytest.approx(lamb_lprior(lam, phi, P, k), rel=1e-5) == expected


def test_lamb_A_lprior():
    lam_A = np.array([1.0, 1.0])
    lam_mat = np.array([[0.5, 1.5], [1.0, 2.0]])
    P_A = np.eye(2)
    k = 2
    val = lamb_A_lprior(lam_A, lam_mat, P_A, k)
    assert not np.isnan(val) and val not in (float("inf"), float("-inf"))


def test_phi_lprior():
    assert pytest.approx(phi_lprior(0.5, 1.0), rel=1e-5) == -0.5


def test_delta_lprior():
    assert not np.isnan(delta_lprior(1.0))


def test_b_lprior():
    assert b_lprior(61.25) != float("-inf")
    assert b_lprior(60.9) == float("-inf")


def test_glprior():
    assert glprior(-0.65) != float("-inf")
    assert glprior(-0.7) == float("-inf")


def test_psilprior():
    assert not np.isnan(psilprior(-4.0))


def test_power_law():
    b = 10.0
    g = 2.0
    f = np.array([1.0, 10.0])
    expected = np.array([10.0, 14.60517])
    assert np.allclose(power_law(b, g, f), expected, atol=1e-5)
