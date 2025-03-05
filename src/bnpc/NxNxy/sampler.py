import matplotlib.pyplot as plt
import numpy as np

from ..logSplines import LogSplineData, logPsplines
from ..signal.signal_lisa import Signal
from .core import noise_psd_cal, tot_sig_psd_cal, llike_prisum_lpost, update_phi_delta
from .update_signal import update_signal_param
from .update_weights import update_lambda
from ..utils import basic_summaries, get_pdgrm

"""
This file contains the Sampler class which is used to run the MCMC.
"""


class Sampler:
    def __init__(
        self,
        T: np.ndarray,
        n: int,
        k: int,
        burnin: int,
        A: np.ndarray = None,
        E: np.ndarray = None,
        Spar_x: np.ndarray = 1,
        Spar_xy: np.ndarray = None,
        degree: int = 3,
        modelnum: int = 1,
        f: np.ndarray = None,
        fs: float = None,
        blocked: bool = False,
        signal_model: int = None,
        data_bin_edges: np.ndarray = None,
        data_bin_weights: np.ndarray = None,
        log_data: bool = True,
    ):
        # Store MCMC input parameters
        self.T = T
        self.n = n
        self.k = k
        self.burnin = burnin
        self.A = A
        self.E = E
        self.Spar_x = Spar_x
        self.Spar_xy = Spar_xy
        self.degree = degree
        self.modelnum = modelnum
        self.fs = fs
        self.blocked = blocked
        self.signal_model = signal_model
        self.data_bin_edges = data_bin_edges
        self.data_bin_weights = data_bin_weights
        self.log_data = log_data

        # Checking for blocked data and whether given one or three channels
        pdgrm = get_pdgrm(self.blocked, self.A, self.T)
        # Frequency setup
        if f is None:
            f = np.linspace(0, self.fs / 2, len(pdgrm + 1))[1:]
        self.f = f
        self.llike = np.zeros(n)
        self.prisum = np.zeros(n)
        self.logpost = np.zeros(n)
        self.npsdNx = np.zeros((n, len(pdgrm)))  # noise PSD Nx
        self.npsdNxy = np.zeros((n, len(pdgrm)))  # noise PSD Nxy
        self.npsdT = np.zeros((n, len(pdgrm)))  # noise PSD T channel
        dataobj_x = LogSplineData(
            data=pdgrm,
            Spar=self.Spar_x,
            n=self.n,
            n_knots=self.k,
            degree=self.degree,
            f=self.f,
            data_bin_edges=self.data_bin_edges,
            data_bin_weights=self.data_bin_weights,
            log_data=self.log_data,
            equidistant=True,
        )
        self.logpsplines_x = logPsplines(dataobj=dataobj_x)
        self.logpsplines_x.covObj = np.eye(k)
        self.logpsplines_x.a_phi = self.k / 2 + 1
        self.logpsplines_x.a_delta = 1 + 1e-4
        self.logpsplines_x.count = []
        self.logpsplines_x.lam_mat[0,:] = np.ones(int(k)) * 1e-5
        dataobj_xy = LogSplineData(
                data=pdgrm,
                Spar=self.Spar_xy,
                n=self.n,
                n_knots=self.k,
                degree=self.degree,
                f=self.f,
                data_bin_edges=self.data_bin_edges,
                data_bin_weights=self.data_bin_weights,
                log_data=self.log_data,
                equidistant=True,
            )
        self.logpsplines_xy = logPsplines(dataobj=dataobj_xy)
        self.logpsplines_xy.covObj = np.eye(k)
        self.logpsplines_xy.a_phi = self.k / 2 + 1
        self.logpsplines_xy.a_delta = 1 + 1e-4
        self.logpsplines_xy.lam_mat[0,:] = np.ones(int(k)) * 1e-5
        self.Uv_am = np.random.uniform(0, 1, n)
        self.Uv = np.log(np.random.uniform(0, 1, n))
        self.Ik = (1e-7) * np.diag(np.ones(self.k) / self.k)
        self.c_amh = (2.38**2) / k
        self.logpsplines_xy.count = []
        # signal
        self.signal = Signal(
                n=self.n, signal_model=self.signal_model, f=self.f, data=pdgrm
        )
        self.npsdA = np.zeros((n, len(pdgrm)))  # noise PSD of A channel
        self.totpsd = np.zeros((n, len(pdgrm)))  # total PSD

        noise_psd_cal(self, 0)
        tot_sig_psd_cal(self, 0)
        llike_prisum_lpost(self, 0)

    def MCMCloop(self):
        for i in range(1, self.n):
            # update lambda
            update_lambda(self, i)


            # sample phi delta Nx:
            update_phi_delta(self.logpsplines_x, i)
            update_phi_delta(self.logpsplines_xy, i)
            
            #calculating the noises in A and T channels:
            noise_psd_cal(self, i)

            # Updating the signal
            update_signal_param(self, i)

            #total PSD and signal PSD
            tot_sig_psd_cal(self, i)
            
            #log likelihood, prior sum and log posterior
            llike_prisum_lpost(self, i)


class MCMCResult:
    '''
    MCMC results
    '''
    def __init__(self, sampler):
        self.A = sampler.A
        self.E = sampler.E
        self.T = sampler.T
        self.f = sampler.f

        # Burn-in region
        burnin = sampler.burnin
        n = sampler.n
        cond = slice(burnin, n)

        self.phi = sampler.logpsplines_x.phi[cond]
        self.delta = sampler.logpsplines_xy.delta[cond]

        self.loglike = sampler.llike[cond]
        self.logpost = sampler.logpost[cond]

        # Noise PSD from T channel
        self.noise_psd_T = sampler.npsdT[cond, :]

        self.noise_psd_Nx = sampler.npsdNx[cond, :]
        self.noise_psd_Nxy = sampler.npsdNxy[cond, :]

        # Lam_mat in Nx channel
        self.lambda_matrix = sampler.logpsplines_x.lam_mat[cond, :]

        self.knots = sampler.logpsplines_x.knots

        self.splines_psd_x = sampler.logpsplines_x.splines_mat
        self.splines_psd_xy = sampler.logpsplines_xy.splines_mat

        self.lambda_matrix_xy = sampler.logpsplines_xy.lam_mat[cond, :]
        self.noise_psd_A = sampler.npsdA[cond, :]
        self.tot_psd = sampler.totpsd[cond, :]
        self.knots_xy = sampler.logpsplines_xy.knots
        self.splines_psd = sampler.logpsplines_xy.splines_mat
        self.b = sampler.signal.b[cond]
        self.g = sampler.signal.g[cond]
        self.psi = sampler.signal.psi[cond] if hasattr(sampler.signal, "psi") else None
        self.sigpsd = sampler.signal.s_s[cond, :]
        self.cov_xy = sampler.logpsplines_xy.covObj

    def summary(self):
        """
        Provide a quick summary of the MCMC results.
        """
        # Example: create trace plots for b, g, and log-likelihood
        n_plots = 0
        fig, axes = plt.subplots(
            n_plots, 1, figsize=(8, 3 * n_plots), sharex=True
        )
        if n_plots == 1:
            axes = [axes]

        idx = 0
        axes[idx].plot(self.phi, label="log amplitude", color="blue")
        axes[idx].set_title("Trace of log amplitude")
        axes[idx].legend()
        idx += 1


        axes[idx].plot(
                self.delta, label="slope of powerlaw", color="orange"
            )
        axes[idx].set_title("Trace of slope of powerlaw")
        axes[idx].legend()
        idx += 1


        axes[idx].plot(self.loglike, label="log-likelihood", color="green")
        axes[idx].set_title("Trace of log-likelihood")
        axes[idx].legend()
        idx += 1

        plt.tight_layout()
        plt.show()
        return fig

    def basic_summary(self):
        """
        Print numeric summaries (mean, sd, 95% CI) for each parameter.
        """
        mp, sp, lp, up = basic_summaries(self.b)
        print(f"log amplitude: mean={mp:.3f}, sd={sp:.3f}, 95% CI=({lp:.3f}, {up:.3f})")

        md, sd, ld, ud = basic_summaries(self.g)
        print(f"slope: mean={md:.3f}, sd={sd:.3f}, 95% CI=({ld:.3f}, {ud:.3f})")
