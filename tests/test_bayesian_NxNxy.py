import unittest

import numpy as np

from bnpc.NxNxy.core import (
    delta_lprior,
    lamb_lprior,
    loglike1,
    lpost,
    phi_lprior,
    prior_sum,
    tot_psd,
)
from bnpc.signal.utils import b_lprior, glprior, power_law, psilprior


class TestCoreFunctions(unittest.TestCase):
    def test_lpost(self):
        """
        Test the log posterior combining log-likelihood and log-prior.
        """
        self.assertEqual(lpost(1.0, 2.0), 3.0)

    def test_tot_psd(self):
        """
        Test the tot_psd function, which computes log of sums of two PSD arrays.
        """
        s_n = np.array([0.0, 2.0, 1.0])  # log PSD of noise
        s_s = np.array([1.0, 1.0, 2.0])  # log PSD of signal
        expected = np.array([1.313262, 2.3132617, 2.3132617])

        result = tot_psd(s_n, s_s)
        self.assertTrue(np.allclose(result, expected, atol=1e-6))

    def test_loglike(self):
        """
        Test the loglike function, which calculates
        -sum( S + exp(log(pdgrm) - S) ).
        """
        pdgrm = np.array([10.0, 20.0])
        S = np.array([2.0, 3.0])
        expected = -7.349092  # approximate

        lnlike_val = loglike1(pdgrm, S)
        self.assertAlmostEqual(lnlike_val, expected, places=5)

    def test_prior_sum(self):
        """
        Test summation of multiple prior components.
        """
        val = prior_sum(1.0, 2.0, 3.0)
        self.assertEqual(val, 6.0)

    def test_lamb_lprior(self):
        """
        Test lamb_lprior.
        This function is k*(log(phi))/2 - (phi/2)*(lam^T * P * lam).
        """
        lam = np.array([1.0, 2.0])
        phi = 0.5
        P = np.array([[2.0, 0.0], [0.0, 2.0]])  # a simple penalty matrix
        k = len(lam)  # = 2
        expected = -3.193147

        val = lamb_lprior(lam, phi, P, k)
        self.assertAlmostEqual(val, expected, places=5)

    def test_phi_lprior(self):
        """
        phi ~ Gamma(a=1, scale=1/delta).
        """
        from math import isclose

        val = phi_lprior(0.5, 1.0)
        self.assertTrue(isclose(val, -0.5, abs_tol=1e-5))

    def test_delta_lprior(self):
        """
        delta ~ Gamma(a=1e-4, scale=1 / 1e-4).
        """
        val = delta_lprior(1.0)
        self.assertFalse(np.isnan(val))

    def test_b_lprior(self):
        """
        b is uniform(61, 61.5).
        """
        # in-range
        self.assertNotEqual(b_lprior(62.8), float("-inf"))
        # out-of-range => should be -inf
        self.assertEqual(b_lprior(60.9), float("-inf"))
        self.assertFalse(np.isnan(b_lprior(70)))


    def test_glprior(self):
        """
        g is uniform(-0.68, -0.63).
        """
        self.assertNotEqual(glprior(-0.65), float("-inf"))
        self.assertEqual(glprior(-0.7), float("-inf"))
        self.assertFalse(np.isnan(glprior(0.7)))


    def test_psilprior(self):
        """
        psi ~ Normal(loc=-4, scale=0.1).
        """
        val = psilprior(-4.0)
        self.assertFalse(np.isnan(val))

    def test_power_law(self):
        """
        power_law(b, g, f).
        => y = b + g * log(f).
        """
        b = 10.0
        g = 2.0
        f = np.array([1.0, 10.0])
        expected = np.array([10.0, 14.60517])
        result = power_law(b, g, f)
        self.assertTrue(np.allclose(result, expected, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
