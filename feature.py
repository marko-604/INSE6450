import torch


class Feature(object):
    """
    Wrapper for differentiable feature functions that operate on tensors.

    Example usage:
        f1 = Feature(lambda t, x, u: -(x[3] - 1.0)**2)
        f2 = Feature(lambda t, x, u: -u[0]**2 - u[1]**2)
        combined = f1 + 0.5 * f2
        val = combined(t, x, u)
    """

    def __init__(self, f):
        self.f = f

    def __call__(self, *args):
        return self.f(*args)

    def __add__(self, r):
        return Feature(lambda *args: self(*args) + r(*args))

    def __radd__(self, r):
        return Feature(lambda *args: r(*args) + self(*args))

    def __mul__(self, r):
        return Feature(lambda *args: self(*args) * r)

    def __rmul__(self, r):
        return Feature(lambda *args: r * self(*args))

    def __pos__(self):
        return self

    def __neg__(self):
        return Feature(lambda *args: -self(*args))

    def __sub__(self, r):
        return Feature(lambda *args: self(*args) - r(*args))

    def __rsub__(self, r):
        return Feature(lambda *args: r(*args) - self(*args))


def feature(f):
    """
    Decorator for creating Feature objects.
    """
    return Feature(f)


def speed(s=1.0):
    """
    Encourages maintaining a desired speed s.
    Penalizes deviation of x[3] (velocity) from s.
    """
    @feature
    def f(t, x, u):
        # -(v - s)^2
        return -((x[3] - s) ** 2)
    return f


def control():
    """
    Penalizes high control effort (quadratic cost on u).
    """
    @feature
    def f(t, x, u):
        # -u[0]^2 - u[1]^2
        return -(u[0] ** 2) - (u[1] ** 2)
    return f


def bounded_control(bounds, width=0.05):
    """
    Adds a smooth penalty when controls approach given bounds.

    Parameters
    ----------
    bounds : list of (float, float)
        [(min, max), ...] for each control component.
    width : float
        Controls how quickly the exponential penalty grows near the limits.
    """
    @feature
    def f(t, x, u):
        ret = 0.0
        for i, (a, b) in enumerate(bounds):
            # Smooth exponential penalty near limits
            ret += -torch.exp((u[i] - b) / width) - torch.exp((a - u[i]) / width)
        return ret
    return f


if __name__ == '__main__':
    # quick test
    t = torch.tensor(0.0)
    x = torch.tensor([0.0, 0.0, 0.0, 1.2])
    u = torch.tensor([0.2, -0.1])

    f_speed = speed(1.0)
    f_control = control()
    f_bounded = bounded_control([(-1, 1), (-1, 1)])

    print("Speed feature:", f_speed(t, x, u))
    print("Control feature:", f_control(t, x, u))
    print("Bounded feature:", f_bounded(t, x, u))
