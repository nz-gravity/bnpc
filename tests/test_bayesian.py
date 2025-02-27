import unittest

import numpy as np

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


class TestCoreFunctions(unittest.TestCase):
    def test_lpost(self):
        """
        Test the log posterior combining log-likelihood and log-prior.
        """
        # Simple check: lpost(loglike=1, lpriorsum=2) => expected 3
        self.assertEqual(lpost(1.0, 2.0), 3.0)

    def test_tot_psd(self):
        """
        Test the tot_psd function, which computes log of sums of two PSD arrays.
        We need to pass np.ndarray, not scalars.
        """
        # Example small arrays
        s_n = np.array([0.0, 2.0, 1.0])  # log PSD of noise
        s_s = np.array([1.0, 1.0, 2.0])  # log PSD of signal

        # Expected result for each index:
        # idx=0 => s_n=0.0, s_s=1.0 => use s_s + log(1 + exp(s_n - s_s))
        #         = 1.0 + log(1 + exp(-1)) = 1 + log(1 + 0.3679...) = 1 + -0.3132617... = 0.6867383...
        # idx=1 => s_n=2.0, s_s=1.0 => s_n + log(1 + exp(s_s - s_n))
        #         = 2.0 + log(1 + exp(-1)) = 2.0 + log(1 + 0.3679...) = 2 + 0.3132617... = 2.3132617...
        # idx=2 => s_n=1.0, s_s=2.0 => s_s + log(1 + exp(s_n - s_s))
        #         = 2.0 + log(1 + exp(-1)) = 2.3132617...
        expected = np.array([0.6867383, 2.3132617, 2.3132617])

        result = tot_psd(s_n, s_s)

        # Use np.allclose for floating comparisons
        self.assertTrue(np.allclose(result, expected, atol=1e-6))

    def test_loglike(self):
        """
        Test the loglike function, which calculates
        -sum( S + exp(log(pdgrm) - S) ).
        """
        pdgrm = np.array([10.0, 20.0])
        S = np.array([2.0, 3.0])
        # => For each index i: S[i] + exp(log(pdgrm[i]) - S[i])
        # i=0 => 2 + exp(log(10) - 2) = 2 + exp(2.302585 - 2) = 2 + exp(0.302585) = 2 + 1.353352 = 3.353352
        # i=1 => 3 + exp(log(20) - 3) = 3 + exp(2.995732 - 3) = 3 + exp(-0.004268) = 3 + 0.99574 = 3.99574
        # sum(...) = 3.353352 + 3.99574 = 7.349092
        # => lnlike = -7.349092
        expected = -7.349092  # approximate

        lnlike_val = loglike(pdgrm, S)
        self.assertAlmostEqual(lnlike_val, expected, places=5)

    def test_loglike_A(self):
        """
        Test the loglike_A function, which is loglike(A, S) + loglike(E, S).
        """
        # Just make small test arrays
        A = np.array([10.0, 20.0])
        E = np.array([5.0, 15.0])
        S = np.array([2.0, 3.0])  # same logic as above

        # Reusing the loglike test approach
        lnlike_A_part = loglike(A, S)  # from the test above
        lnlike_E_part = loglike(E, S)
        # total
        lnlike_AE = lnlike_A_part + lnlike_E_part

        # call loglike_A
        result = loglike_A(A, E, S)
        self.assertAlmostEqual(result, lnlike_AE, places=5)

    def test_prior_sum(self):
        """
        Test summation of multiple prior components.
        """
        # trivial check
        val = prior_sum(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
        # => should be 1+2+3+4+5+6+7 = 28
        self.assertEqual(val, 28.0)

    def test_lamb_lprior(self):
        """
        Test lamb_lprior.
        This function is k*(log(phi))/2 - (phi/2)*(lam^T * P * lam).
        """
        lam = np.array([1.0, 2.0])
        phi = 0.5
        P = np.array([[2.0, 0.0], [0.0, 2.0]])  # a simple penalty matrix
        k = len(lam)  # = 2
        # => k*log(phi)/2 = 2*log(0.5)/2 = log(0.5) = -0.693147...
        # => lam^T * P * lam = [1,2]*[[2,0],[0,2]]*[1,2]^T
        #    => [1,2]*[2,4] = 1*2 + 2*4 = 2 + 8 =10
        # => (phi/2)*10 = 0.5/2 * 10 = 0.25 * 10 =2.5
        # => total = -0.693147 -2.5 = -3.193147
        expected = -3.193147

        val = lamb_lprior(lam, phi, P, k)
        self.assertAlmostEqual(val, expected, places=5)

    def test_lamb_A_lprior(self):
        """
        Test lamb_A_lprior.
        """
        lam_A = np.array([1.0, 1.0])
        lam_mat = np.array([[0.5, 1.5], [1.0, 2.0]])  # 2 x 2
        P_A = np.eye(2)  # 2x2 Identity
        k = 2

        # We can only do an approximate test unless we replicate the entire logic of updata_phi_A.
        # For now, just call the function to ensure it runs without errors.
        val = lamb_A_lprior(lam_A, lam_mat, P_A, k)
        # We can't know "expected" exactly without dissecting updata_phi_A.
        # At least check it doesn't return NaN or inf under normal conditions:
        self.assertFalse(np.isnan(val))
        self.assertNotEqual(val, float("inf"))
        self.assertNotEqual(val, float("-inf"))

    def test_phi_lprior(self):
        """
        phi ~ Gamma(a=1, scale=1/delta).
        We'll pick delta=1, phi=0.5 => gamma.logpdf(0.5, a=1, scale=1)
        => gamma(1, scale=1) is Exp(1) distribution => pdf = exp(-x) for x>0,
           so logpdf(0.5) = -0.5
        """
        from math import isclose

        val = phi_lprior(0.5, 1.0)
        # logpdf for Exp(1) at x=0.5 => -0.5
        self.assertTrue(isclose(val, -0.5, abs_tol=1e-5))

    def test_delta_lprior(self):
        """
        delta ~ Gamma(a=1e-4, scale=1 / 1e-4).
        => a=1e-4, scale=1e4 => mean ~ a*scale=1, but let's do a small numeric check:
        """
        # We'll just ensure it doesn't crash for a typical value:
        val = delta_lprior(1.0)
        self.assertFalse(np.isnan(val))

    def test_b_lprior(self):
        """
        b is uniform(61, 61.5).
        => log prior is 0 if b in [61, 61.5], else -inf
        """
        # in-range
        self.assertNotEqual(b_lprior(61.25), float("-inf"))
        # out-of-range => should be -inf
        self.assertEqual(b_lprior(60.9), float("-inf"))

    def test_glprior(self):
        """
        g is uniform(-0.68, -0.63).
        => log prior is 0 if g in [-0.68, -0.63], else -inf
        """
        self.assertNotEqual(glprior(-0.65), float("-inf"))
        self.assertEqual(glprior(-0.7), float("-inf"))

    def test_psilprior(self):
        """
        psi ~ Normal(loc=-4, scale=0.1).
        We just check it doesn't produce NaN for typical psi.
        """
        val = psilprior(-4.0)
        self.assertFalse(np.isnan(val))

    def test_power_law(self):
        """
        power_law(b, g, f). We can do a small numeric check.
        => y = b + g * log(f).
        """
        b = 10.0
        g = 2.0
        f = np.array([1.0, 10.0])
        # => y[0] = 10 + 2*log(1) = 10
        # => y[1] = 10 + 2*log(10) = 10 + 2*2.302585 = 14.60517
        expected = np.array([10.0, 14.60517])
        result = power_law(b, g, f)
        self.assertTrue(np.allclose(result, expected, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
