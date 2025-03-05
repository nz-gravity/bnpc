import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from .utils import signal_density


class Signal:
    """
    Signal class
    """

    def __init__(self, n, signal_model, f, data):
        """
        Initialize the Signal object
        """
        # Example: ensure self.n, self.signal_model, and self.f, self.data are defined somewhere
        # For demonstration, let's assume they are already set: self.n, self.signal_model, self.f, self.data, etc.

        # Initialize b, g
        self.b = np.concatenate(([np.log(1.9e27)], np.zeros(n - 1)))
        self.g = np.concatenate(([-2 / 3], np.zeros(n - 1)))
        self.psi = np.zeros(n)



        # Initialize the signal array
        self.s_s = np.zeros((n, len(data)))  # signal density
        self.s_s[0] = signal_density(
            b=self.b[0],
            g=self.g[0],
            psi=self.psi[0],
            f=f,
            signal_model=signal_model,
        )

    def plot_signal_with_prior(self, f, signal_model, num_samples=200):
        """
        Plot the initial signal and the region of prior distributions.
        """
        # Initial signal density
        init_signal = signal_density(
            b=self.b[0],
            g=self.g[0],
            psi=self.psi[0],
            f=f,
            signal_model=signal_model,
        )

        b_prior = np.random.uniform(62.5, 63.1, size=num_samples)
        g_prior = np.random.uniform(-0.67, -0.65, size=num_samples)
        psi_prior = norm(loc=-4, scale=0.1)

        signals = []
        for i in range(num_samples):
            # Signal PSD
            s_draw = signal_density(
                b=b_prior[i],
                g=g_prior[i],
                psi=psi_prior[i],
                f=f,
                signal_model=signal_model,
            )
            signals.append(s_draw)

        signals = np.exp(
            np.array(signals)
        )

        # Percentile ranges
        lower = np.percentile(signals, 2.5, axis=0)
        median = np.percentile(signals, 50.0, axis=0)
        upper = np.percentile(signals, 97.5, axis=0)

        plt.figure(figsize=(8, 5))

        plt.fill_between(
            f, lower, upper, color="cyan", alpha=0.2, label="95% prior region"
        )

        # Median prior-based signal
        plt.plot(f, median, "r--", label="Prior median signal")

        # Initial signal
        plt.plot(f, init_signal, "k-", label="Initial signal")

        plt.xlabel("Frequency [Hz]")
        plt.ylabel("PSD")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
