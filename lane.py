import numpy as np
import torch
import feature


class Lane(object):
    """Base lane class (placeholder for inheritance)."""
    pass


class StraightLane(Lane):
    """
    Represents a straight lane segment between two points p and q, with a given width w.
    """

    def __init__(self, p, q, w):
        """
        Parameters
        ----------
        p : array-like
            Start point [x, y].
        q : array-like
            End point [x, y].
        w : float
            Lane width.
        """
        self.p = np.asarray(p, dtype=float)
        self.q = np.asarray(q, dtype=float)
        self.w = w

        # Unit direction vector along the lane (p -> q)
        self.m = (self.q - self.p) / np.linalg.norm(self.q - self.p)
        # Unit normal vector perpendicular to the lane
        self.n = np.asarray([-self.m[1], self.m[0]])

    def shifted(self, m):
        """
        Create a shifted parallel lane at multiplier `m` of the lane width.

        Example:
            shifted(+1) = lane to the left
            shifted(-1) = lane to the right
        """
        return StraightLane(
            self.p + self.n * self.w * m,
            self.q + self.n * self.w * m,
            self.w
        )

    def dist2(self, x):
        """
        Squared perpendicular distance from point x to the lane centerline.

        Parameters
        ----------
        x : array-like or torch.Tensor of shape (2,)
            Position [x, y].
        """
        if isinstance(x, torch.Tensor):
            r = (x[0] - self.p[0]) * self.n[0] + (x[1] - self.p[1]) * self.n[1]
            return r ** 2
        else:
            # Fallback for NumPy input
            r = (x[0] - self.p[0]) * self.n[0] + (x[1] - self.p[1]) * self.n[1]
            return r * r

    def gaussian(self, width=0.5):
        """
        Returns a Gaussian-shaped feature centered on the lane centerline.

        Parameters
        ----------
        width : float
            Controls how wide the Gaussian is (in relative lane-width units).

        Returns
        -------
        feature.Feature
            Callable that takes (t, x, u) and returns exp(-0.5 * dist2 / scale^2)
        """
        @feature.feature
        def f(t, x, u):
            # Ensure x is a torch.Tensor
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x, dtype=torch.float32)

            # Compute the Gaussian penalty around the lane centerline
            return torch.exp(-0.5 * self.dist2(x) / (width ** 2 * self.w * self.w / 4.0))

        return f


if __name__ == '__main__':
    lane = StraightLane([0.0, -1.0], [0.0, 1.0], 0.1)
    x = torch.tensor([0.05, 0.3])
    f = lane.gaussian()
    print("Gaussian feature value:", f(0, x, 0))
