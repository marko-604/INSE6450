# dynamics.py (PyTorch version)

import torch


class Dynamics(object):
    def __init__(self, nx, nu, f, dt=None):
        """
        Generic dynamics wrapper.

        Parameters
        ----------
        nx : int
            State dimension.
        nu : int
            Control dimension.
        f : callable
            Base continuous-time or discrete-time dynamics:
                next_x = f(x, u)
            where x and u are torch.Tensors.
        dt : float or None
            If dt is provided, we wrap f as:
                x_next = x + dt * f(x, u)
            which is a simple forward-Euler integration step.
            If dt is None, we just use f directly.
        """
        self.nx = nx
        self.nu = nu
        self.dt = dt

        if dt is None:
            self.f = f
        else:
            # Wrap with forward Euler step: x_{k+1} = x_k + dt * f(x_k, u_k)
            def f_euler(x, u):
                return x + dt * f(x, u)
            self.f = f_euler

    def __call__(self, x, u):
        """
        Apply the dynamics function.

        Parameters
        ----------
        x : torch.Tensor, shape (nx,)
        u : torch.Tensor, shape (nu,)
        """
        return self.f(x, u)


class CarDynamics(Dynamics):
    def __init__(self, dt=0.1, ub=[(-3., 3.), (-1., 1.)], friction=1.0):
        """
        Simple car-like dynamics in 2D.

        State x = [x_pos, y_pos, heading, speed]
        Control u = [steer, gas]

        Continuous-time model f(x, u) (used inside Euler step if dt is not None):

            x_dot = v * cos(theta)
            y_dot = v * sin(theta)
            theta_dot = v * steer
            v_dot = gas - friction * v

        Parameters
        ----------
        dt : float
            Time step for Euler integration.
        ub : list of tuples
            Control bounds [(steer_min, steer_max), (gas_min, gas_max)].
            Not enforced here; just stored for external use.
        friction : float
            Linear friction coefficient for speed dynamics.
        """
        self.ub = ub
        self.friction = friction

        def f(x, u):
            # x, u are expected to be 1D torch.Tensor of shapes (4,) and (2,)
            # x = [x_pos, y_pos, heading, speed]
            # u = [steer, gas]
            return torch.stack([
                x[3] * torch.cos(x[2]),                  # x_dot
                x[3] * torch.sin(x[2]),                  # y_dot
                x[3] * u[0],                             # theta_dot
                u[1] - x[3] * self.friction              # v_dot
            ])

        # nx = 4 (state dim), nu = 2 (control dim)
        super(CarDynamics, self).__init__(nx=4, nu=2, f=f, dt=dt)


if __name__ == '__main__':
    dyn = CarDynamics(dt=0.1)
    x = torch.zeros(4)   # example state
    u = torch.zeros(2)   # example control
    next_x = dyn(x, u)
    print(next_x)
