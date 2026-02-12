import torch
import numpy as np
from scipy.stats import gaussian_kde


class Sampler(object):
    def __init__(self, D):
        self.D = D
        # Avar: (n_constraints × D)
        # yvar: (n_constraints × 1)
        self.Avar = torch.zeros((0, D), dtype=torch.float64)
        self.yvar = torch.zeros((0, 1), dtype=torch.float64)

    @property
    def A(self):
        return self.Avar.numpy()

    @A.setter
    def A(self, value):
        if len(value) == 0:
            self.Avar = torch.zeros((0, self.D), dtype=torch.float64)
        else:
            self.Avar = torch.tensor(value, dtype=torch.float64)

    @property
    def y(self):
        return self.yvar.numpy()

    @y.setter
    def y(self, value):
        if len(value) == 0:
            self.yvar = torch.zeros((0, 1), dtype=torch.float64)
        else:
            self.yvar = torch.tensor(value, dtype=torch.float64)

    # -----------------------------------------------
    # Log probability function (analogous to self.f in Theano)
    # -----------------------------------------------
    def f(self, x: torch.Tensor):
        """Compute the 'energy' for sample x."""
        if self.Avar.numel() == 0:
            return torch.tensor(0.0, dtype=torch.float64)

        # Equivalent to: -sum(relu(dot(-yvar * Avar, x)))
        prod = -self.yvar * self.Avar  # element-wise multiply rows
        val = torch.matmul(prod, x)
        return -torch.sum(torch.relu(val))

    # -----------------------------------------------
    # Posterior sampling (MCMC via simple Metropolis-Hastings)
    # -----------------------------------------------
    def sample(self, N, T=50, burn=1000, step_size=0.05):
        """
        Draw N samples using a lightweight Metropolis-Hastings sampler.
        Arguments:
            N : int — number of samples
            T : int — thinning interval
            burn : int — number of burn-in steps
            step_size : float — proposal std deviation
        """
        device = torch.device("cpu")
        x = torch.zeros(self.D, dtype=torch.float64, device=device)
        samples = []

        def log_prob(z):
            # Uniform prior on unit ball
            if torch.sum(z ** 2) >= 1.0:
                return -torch.inf
            return self.f(z)

        current_lp = log_prob(x)

        total_steps = N * T + burn
        for step in range(total_steps):
            # propose new sample
            proposal = x + step_size * torch.randn_like(x)
            lp_prop = log_prob(proposal)

            # acceptance rule
            accept_ratio = torch.exp(lp_prop - current_lp)
            if torch.rand(1) < accept_ratio:
                x = proposal
                current_lp = lp_prop

            # record every T steps after burn-in
            if step >= burn and (step - burn) % T == 0:
                x_unit = x / (torch.norm(x) + 1e-8)
                samples.append(x_unit.cpu().numpy())

        samples = np.stack(samples)
        return samples
