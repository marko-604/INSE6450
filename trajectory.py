import torch
import utils_driving as utils  # keep if you use other stuff from utils; otherwise you can remove it


class Trajectory(object):
    """
    PyTorch version of the original Theano-based Trajectory.

    Differences vs original:
    - self.x0 is now a torch.Tensor, not a Theano shared variable.
    - self.u is a list of torch.Tensor, not Theano shared variables.
    - No theano.function / graph compilation. We just call dyn directly.
    - We keep a roll-out of states in self.x for convenience, recomputed as needed.
    """

    def __init__(self, T, dyn, x0=None):
        """
        Parameters
        ----------
        T : int
            Horizon length.
        dyn : callable
            Dynamics function: next_state = dyn(x, u)
            Must accept and return torch.Tensors.
        x0 : array-like or torch.Tensor, optional
            Initial state. If None, initialized to zeros.
        """
        self.dyn = dyn
        self.T = T

        nx = dyn.nx  # assuming dyn exposes state dimension
        nu = dyn.nu  # assuming dyn exposes control dimension

        # Initial state
        if x0 is None:
            self.x0 = torch.zeros(nx, dtype=torch.float32)
        else:
            self.x0 = torch.as_tensor(x0, dtype=torch.float32)

        # Control sequence over horizon: list of T vectors of size nu
        self.u = [
            torch.zeros(nu, dtype=torch.float32) for _ in range(self.T)
        ]

        # Rollout of states along the horizon, starting from x0 and applying u
        self.x = []
        self.rollout()

    def rollout(self):
        """
        Recompute the full trajectory self.x by rolling out the dynamics
        from the current x0 and control sequence u.
        """
        self.x = []
        z = self.x0
        for t in range(self.T):
            z = self.dyn(z, self.u[t])
            self.x.append(z)

    def tick(self):
        """
        Advance the trajectory by one time step using a receding-horizon scheme:
        - x0 <- dyn(x0, u[0])
        - shift control sequence left: u[t] <- u[t+1]
        - set last control to zero
        - recompute self.x for the new horizon

        This is the conceptual equivalent of the original Theano:
          self.x0.set_value(self.next_x())
          u[t].set_value(u[t+1].get_value())
          u[-1].set_value(zeros)
        """
        # One-step advance of the initial state
        x_next = self.dyn(self.x0, self.u[0])
        self.x0 = x_next

        # Shift controls
        for t in range(self.T - 1):
            self.u[t] = self.u[t + 1].clone()

        # Last control becomes zero
        self.u[self.T - 1] = torch.zeros(self.dyn.nu, dtype=torch.float32)

        # Recompute rollout based on new x0 and u
        self.rollout()
