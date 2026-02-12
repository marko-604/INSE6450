import torch

import utils_driving as utils
from trajectory import Trajectory
import feature


class Car(object):
    """
    PyTorch-based Car wrapper around a dynamics model and Trajectory.

    This replaces the Theano shared-variable logic with direct torch.Tensor use.
    """

    def __init__(self, dyn, x0, color='yellow', T=5):
        """
        Parameters
        ----------
        dyn : object
            Dynamics model; must expose at least:
              - dyn.nx : int, state dimension
              - dyn.nu : int, control dimension
              - be callable: next_x = dyn(x, u) with torch.Tensor inputs
        x0 : array-like or torch.Tensor
            Initial state.
        color : str
            Used externally, e.g., for visualization.
        T : int
            Planning horizon.
        """
        x0_tensor = torch.as_tensor(x0, dtype=torch.float32)

        self.data0 = {'x0': x0_tensor.clone()}
        self.bounds = [(-1., 1.), (-1., 1.)]
        self.T = T
        self.dyn = dyn

        # Main trajectory
        self.traj = Trajectory(T, dyn, x0=x0_tensor)
        # Linear / reference trajectory (if you use it for linearization, MPC, etc.)
        self.linear = Trajectory(T, dyn, x0=x0_tensor)

        self.color = color

        # Default control vector
        self.default_u = torch.zeros(self.dyn.nu, dtype=torch.float32)

    def reset(self):
        """
        Reset the car to its original initial state and default controls.
        """
        # Reset states
        self.traj.x0 = self.data0['x0'].clone()
        self.linear.x0 = self.data0['x0'].clone()

        # Reset controls for the full horizon
        for t in range(self.T):
            self.traj.u[t] = torch.zeros(self.dyn.nu, dtype=torch.float32)
            self.linear.u[t] = self.default_u.clone()

        # Recompute rollouts for both trajectories
        self.traj.rollout()
        self.linear.rollout()

    def move(self):
        """
        Advance the main trajectory by one tick and update the linear trajectory's
        starting state to the new current state.
        """
        # Advance main trajectory (receding horizon update)
        self.traj.tick()

        # Sync linear model to current state and recompute its rollout
        self.linear.x0 = self.traj.x0.clone()
        self.linear.rollout()

    @property
    def x(self):
        """
        Current state (torch.Tensor).
        """
        return self.traj.x0

    @x.setter
    def x(self, value):
        """
        Set current state.
        Accepts array-like or torch.Tensor.
        """
        self.traj.x0 = torch.as_tensor(value, dtype=torch.float32)
        # Optional: maintain consistency by recomputing rollout
        self.traj.rollout()

    @property
    def u(self):
        """
        Current control at time step 0 (torch.Tensor).
        """
        return self.traj.u[0]

    @u.setter
    def u(self, value):
        """
        Set current control at time step 0.
        Accepts array-like or torch.Tensor.
        """
        self.traj.u[0] = torch.as_tensor(value, dtype=torch.float32)
        # Optional: recompute rollout if you rely on self.traj.x later
        self.traj.rollout()

    def control(self, steer, gas):
        """
        Hook for applying a control command (e.g. from a policy or user input).

        A simple implementation, if dyn.nu == 2, could be:
            self.u = torch.tensor([steer, gas], dtype=torch.float32)

        Leaving as pass here since your control law / policy is project-specific.
        """
        pass
