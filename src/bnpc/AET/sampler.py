import numpy as np
from .init_weights import optimize_starting_weights
from bnpc.logSplines.log_Psplines import logPsplines
from .utils import get_pdgrm, basic_summaries
from bnpc.logSplines.dataobj import LogSplineData
from bnpc.signal.signal_lisa import Signal
from core import llike_prisum_psd
from bnpc.logSplines.utils import update_delta, update_phi
from .update_weights import update_lambda
from .update_signal import update_signal_param
import matplotlib.pyplot as plt

'''
This file contains the Sampler class which is used to run the MCMC. 
'''
class Sampler:
    def __init__(self, T:np.ndarray, n:int, k:int, burnin:int, A:np.ndarray=None,
                 E:np.ndarray=None, Spar:np.ndarray=1, Spar_A:np.ndarray=None, degree:int=3, modelnum:int=1,
                 f:np.ndarray=None, fs:float=None, blocked:bool=False, signal_model:int=None,
                 data_bin_edges:np.ndarray=None, data_bin_weights:np.ndarray=None,
                 log_data:bool=True):
        # Store MCMC input parameters
        self.T = T
        self.n = n
        self.k = k
        self.burnin = burnin
        self.A = A
        self.E = E
        self.Spar = Spar
        self.Spar_A = Spar_A
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
        self.npsdT = np.zeros((n, len(pdgrm))) #noise PSD
        dataobj_T = LogSplineData(
            data=pdgrm,
            Spar=self.Spar,
            n=self.n,
            n_knots=self.k,
            degree=self.degree,
            f=self.f,
            data_bin_edges=self.data_bin_edges,
            data_bin_weights=self.data_bin_weights,
            log_data=self.log_data,
            equidistant=False
        )
        self.logpsplines_T = logPsplines(dataobj=dataobj_T)
        self.logpsplines_T.lam_mat[0, :]=optimize_starting_weights(
            splines=self.logpsplines_T.splines.T,
            Spar=self.Spar,
            modelnum=self.modelnum,
            n_basis=self.logpsplines_T.n_basis,
            data=self.T,
            init_x=self.logpsplines_T.lam_mat[0, :],
            n_optimization_steps=100,
        )

        if self.A is not None:
            dataobj_A = LogSplineData(
                data=get_pdgrm(self.blocked, self.A, self.E),
                Spar=self.Spar_A,
                n=self.n,
                n_knots=self.k,
                degree=self.degree,
                f=self.f,
                data_bin_edges=self.data_bin_edges,
                data_bin_weights=self.data_bin_weights,
                log_data=self.log_data,
                equidistant=False
            )
            self.logpsplines_A = logPsplines(dataobj=dataobj_A)
            self.logpsplines_A.lam_mat[0, :]=self.logpsplines_T.lam_mat[0, :]
            self.logpsplines_A.Uv_am = np.random.uniform(0, 1, n)
            self.logpsplines_A.Uv = np.log(np.random.uniform(0, 1, n))
            self.logpsplines_A.covObj = np.eye(k)
            self.logpsplines_A.Ik = (1e-7) * np.diag(np.ones(self.k) / self.k)
            self.logpsplines_A.c_amh = (2.38 ** 2) / k
            self.logpsplines_A.count = []
            #signal
            self.signal = Signal(
                n=self.n,
                signal_model=self.signal_model,
                f=self.f,
                data=pdgrm
            )
            self.npsdA = np.zeros((n, len(pdgrm)))  # noise PSD of A channel
            self.totpsd = np.zeros((n, len(pdgrm)))  # total PSD

        llike_prisum_psd(self, 0)

    def MCMCloop(self):
        for i in range(1, self.n):
            # update lambda
            update_lambda(self, i)

            # sample phi
            self.logpsplines_T.phi[i] = update_phi(self.logpsplines_T.lam_mat[i, :], None, self.logpsplines_T.delta[i - 1], self.logpsplines_T.a_phi)

            # sample delta
            self.logpsplines_T.delta[i] = update_delta(self.logpsplines_T.phi[i], self.logpsplines_T.a_delta)

            # If A channel is given
            if self.A is not None:
                update_signal_param(self, i)

            llike_prisum_psd(self, i)




class MCMCResult:
    def __init__(self, sampler):
        """
        Take a completed Sampler instance (after MCMCloop() has run),
        slice out the burn-in region, and store the MCMC outputs.

        Parameters
        ----------
        sampler : Sampler
            A Sampler instance that has run MCMCloop().
        """

        self.A = sampler.A
        self.E = sampler.E
        self.T = sampler.T
        self.f = sampler.f

        # Burn-in region
        burnin = sampler.burnin
        n = sampler.n
        cond = slice(burnin, n)

        self.phi = sampler.logpsplines_T.phi[cond]
        self.delta = sampler.logpsplines_T.delta[cond]

        self.loglike = sampler.llike[cond]
        self.logpost = sampler.logpost[cond]

        # Noise PSD from T channel
        self.noise_psd = sampler.npsdT[cond, :]

        # Lam_mat in T channel
        self.lambda_matrix = sampler.logpsplines_T.lam_mat[cond, :]

        self.knots = sampler.logpsplines_T.knots

        self.splines_psd = sampler.logpsplines_T.splines_mat

        # If A-channel is present
        if sampler.A is not None:
            self.lambda_matrix_A = sampler.logpsplines_A.lam_mat[cond, :]
            self.noise_psd_A = sampler.npsdA[cond, :]
            self.tot_psd = sampler.totpsd[cond, :]
            self.knots_A = sampler.logpsplines_A.knots
            self.splines_psd = sampler.logpsplines_A.splines_mat
            self.b = sampler.b[cond] if hasattr(sampler, 'b') else None
            self.g = sampler.g[cond] if hasattr(sampler, 'g') else None
            self.psi = sampler.psi[cond] if hasattr(sampler, 'psi') else None
            self.sigpsd=sampler.signal.s_s[cond, :]
            self.cov_A=sampler.logpsplines_A.covObj

    def summary(self):
        """
        Provide a quick summary of the MCMC results.
        """
        # Example: create trace plots for phi, delta, and log-likelihood if they exist
        n_plots = 0
        fig, axes = plt.subplots(n_plots, 1, figsize=(8, 3 * n_plots), sharex=True)
        if n_plots == 1:
            # Make it a list to index axes consistently
            axes = [axes]

        idx = 0
        if self.b is not None:
            axes[idx].plot(self.phi, label='log amplitude', color='blue')
            axes[idx].set_title('Trace of log amplitude')
            axes[idx].legend()
            idx += 1

        if self.g is not None:
            axes[idx].plot(self.delta, label='slope of powerlaw', color='orange')
            axes[idx].set_title('Trace of slope of powerlaw')
            axes[idx].legend()
            idx += 1

        if self.loglike is not None:
            axes[idx].plot(self.loglike, label='log-likelihood', color='green')
            axes[idx].set_title('Trace of log-likelihood')
            axes[idx].legend()
            idx += 1

        plt.tight_layout()
        plt.show()
        return fig


    def basic_summary(self):
        """
        Print numeric summaries (mean, sd, 95% CI) for each parameter.
        """
        if self.b is not None:
            mp, sp, lp, up = basic_summaries(self.b)
            print(f"log amplitude: mean={mp:.3f}, sd={sp:.3f}, 95% CI=({lp:.3f}, {up:.3f})")

        if self.g is not None:
            md, sd, ld, ud = basic_summaries(self.g)
            print(f"slope: mean={md:.3f}, sd={sd:.3f}, 95% CI=({ld:.3f}, {ud:.3f})")













