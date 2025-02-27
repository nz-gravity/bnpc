import numpy as np
from .init_weights import optimize_starting_weights
from bnpc.logSplines.log_Psplines import logPsplines
from .utils import get_pdgrm
from bnpc.logSplines.dataobj import LogSplineData
from bnpc.signal.signal_lisa import Signal
from core import llike_prisum_psd

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

















