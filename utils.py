import numpy as np
import torch
import scipy.optimize

# Use double precision like Theano typically does
torch.set_default_dtype(torch.float64)


# ---------- Helpers to mimic your "shared" vectors ----------

def scalar():
    """Scalar parameter (like a Theano shared scalar)."""
    return torch.nn.Parameter(torch.tensor(0.0))


def vector(n):
    """Vector parameter of length n."""
    return torch.nn.Parameter(torch.zeros(n))


def matrix(n, m):
    """Matrix parameter (n x m)."""
    return torch.nn.Parameter(torch.zeros(n, m))


def randomize(var):
    """In-place random initialization (Gaussian)."""
    with torch.no_grad():
        var.copy_(torch.randn_like(var))


def get_flat(vars_):
    """Flatten a list of parameters into a single numpy vector."""
    return np.concatenate([v.detach().numpy().ravel() for v in vars_])


def set_flat(vars_, x_np):
    """Set a list of parameters from a single numpy vector."""
    offset = 0
    x_np = np.asarray(x_np, dtype=np.float64)
    for v in vars_:
        sz = v.numel()
        with torch.no_grad():
            v.copy_(torch.from_numpy(x_np[offset:offset + sz]).view_as(v))
        offset += sz


# ---------- Autograd helpers: grad, Jacobian, Hessian ----------

def grad_all(f, vars_, create_graph=False, retain_graph=True):
    """
    Flattened gradient of scalar f w.r.t. a list of tensors vars_.
    Returns a 1D torch tensor.
    """
    grads = torch.autograd.grad(
        f,
        vars_,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=False
    )
    return torch.cat([g.reshape(-1) for g in grads])


def hessian_wrt(f, vars_):
    """
    Full Hessian of scalar f w.r.t. vars_.
    vars_ is a list of tensors; result is (n, n) where n = total numel(vars_).
    """
    g = grad_all(f, vars_, create_graph=True, retain_graph=True)
    n = g.numel()
    rows = []
    for i in range(n):
        gi = g[i]
        grads = torch.autograd.grad(gi, vars_, retain_graph=True)
        rows.append(torch.cat([gg.reshape(-1) for gg in grads]))
    return torch.stack(rows, dim=0)


def jacobian_vec_wrt(vec, vars_):
    """
    Jacobian of a vector-valued expression `vec` w.r.t. vars_.
    vec: 1D tensor (length k)
    vars_: list of tensors (total dimension n)
    Returns (k, n) Jacobian.
    """
    k = vec.numel()
    rows = []
    for i in range(k):
        vi = vec[i]
        grads = torch.autograd.grad(vi, vars_, retain_graph=True)
        rows.append(torch.cat([g.reshape(-1) for g in grads]))
    return torch.stack(rows, dim=0)


# ---------- Single-level maximizer (outer or inner) ----------

class MaximizerTorch(object):
    """
    Maximizes a scalar function f(vs) using SciPy L-BFGS-B,
    where `f_func` is a *zero-argument* function that returns
    a scalar torch.Tensor depending on a list of torch.nn.Parameter `vs`.
    """

    def __init__(self, f_func, vs, pre=None):
        """
        f_func : () -> scalar torch.Tensor
        vs     : list of torch.nn.Parameter
        pre    : optional function called before each evaluation (e.g. extra constraints)
        """
        self.f_func = f_func
        self.vs = vs
        self.pre = pre

    def _f_and_df(self, x_np):
        # Set parameters from flat numpy vector
        set_flat(self.vs, x_np)

        # Optional preprocessing hook
        if self.pre is not None:
            self.pre()

        # Zero gradients
        for v in self.vs:
            if v.grad is not None:
                v.grad.zero_()

        # Forward pass
        f = self.f_func()        # scalar tensor
        f_val = f.item()

        # Backward pass
        g = grad_all(f, self.vs, create_graph=False, retain_graph=False)
        g_np = g.detach().numpy().astype(np.float64)

        # SciPy minimizes; we want to maximize
        return -f_val, -g_np

    def argmax(self, bounds=None, x0=None):
        if bounds is None:
            bounds = [(None, None)] * sum(v.numel() for v in self.vs)

        # Initial point
        if x0 is None:
            x0 = get_flat(self.vs)

        opt_x, f_opt, info = scipy.optimize.fmin_l_bfgs_b(
            self._f_and_df,
            x0,
            bounds=bounds
        )

        # Update parameters to optimum
        set_flat(self.vs, opt_x)

        # Return optimum point, maximum value, and SciPy info
        return opt_x, -f_opt, info

    def maximize(self, bounds=None, x0=None):
        opt_x, f_val, info = self.argmax(bounds=bounds, x0=x0)
        return f_val


# ---------- Nested (bi-level) maximizer with implicit differentiation ----------

class NestedMaximizerTorch(object):
    """
    Nested maximization:

        Inner:  x1* = argmax_{x1} f1(x1, x2)
        Outer:  max_{x2} f2(x1*(x2), x2)

    Uses implicit differentiation to get d f2 / d x2.
    """

    def __init__(self, f1_func, vs1, f2_func, vs2):
        """
        f1_func : () -> scalar torch.Tensor   (inner objective)
        f2_func : () -> scalar torch.Tensor   (outer objective)
        vs1     : list of Parameters for inner variables (x1)
        vs2     : list of Parameters for outer variables (x2)
        """
        self.f1_func = f1_func
        self.vs1 = vs1
        self.f2_func = f2_func
        self.vs2 = vs2

    # ----- inner maximization (argmax w.r.t. x1) -----

    def maximize1(self):
        inner = MaximizerTorch(self.f1_func, self.vs1)
        x0 = get_flat(self.vs1)
        inner.argmax(x0=x0)

    # ----- derivative of outer objective wrt x2 using implicit differentiation -----

    def _outer_grad(self):
        """
        Compute df2/dx2 via:

            df2/dx2 = - J * H^{-1} * g + grad_x2 f2

        where:
            J = d/dx1 [ grad_x2 f1 ]        (shape: n2 x n1)
            H = d/dx1 [ grad_x1 f1 ] = Hessian_x1 f1 (shape: n1 x n1)
            g = grad_x1 f2                  (shape: n1)
        """

        # f1 and f2 at current (x1*, x2)
        f1 = self.f1_func()
        f2 = self.f2_func()

        # grad f1 wrt x2, then its Jacobian wrt x1
        g1_x2 = grad_all(f1, self.vs2, create_graph=True, retain_graph=True)
        J = jacobian_vec_wrt(g1_x2, self.vs1)  # (n2, n1)

        # grad f2 wrt x1
        g2_x1 = grad_all(f2, self.vs1, create_graph=True, retain_graph=True)

        # Hessian of f1 wrt x1
        H = hessian_wrt(f1, self.vs1)          # (n1, n1)

        # Solve H v = g2_x1
        v = torch.linalg.solve(H, g2_x1)

        term1 = -J @ v
        term2 = grad_all(f2, self.vs2, create_graph=False, retain_graph=False)

        return term1 + term2  # gradient wrt vs2

    # ----- function for SciPy: f2(x2), df2/dx2 -----

    def _f2_and_df2(self, x_np):
        # Set outer variables x2
        set_flat(self.vs2, x_np)

        # For this x2, maximize inner f1 over x1
        self.maximize1()

        # Zero grads
        for v in self.vs1 + self.vs2:
            if v.grad is not None:
                v.grad.zero_()

        # Outer objective value
        f2 = self.f2_func()
        f_val = f2.item()

        # Outer gradient via implicit differentiation
        df2 = self._outer_grad()
        g_np = df2.detach().numpy().astype(np.float64)

        # SciPy minimizes, we maximize
        return -f_val, -g_np

    def maximize(self, bounds=None, x0=None):
        if bounds is None:
            bounds = [(None, None)] * sum(v.numel() for v in self.vs2)

        if x0 is None:
            x0 = get_flat(self.vs2)

        opt_x, f_opt, info = scipy.optimize.fmin_l_bfgs_b(
            self._f2_and_df2,
            x0,
            bounds=bounds
        )

        # Update outer variables
        set_flat(self.vs2, opt_x)

        # Make sure inner optimum is updated too
        self.maximize1()

        return opt_x, -f_opt, info


# ---------- Example usage (translated from your __main__) ----------

if __name__ == '__main__':
    # Define variables as torch parameters
    x1 = vector(2)
    x2 = vector(1)

    # IMPORTANT CHANGE: f1 and f2 must now be *functions*, not precomputed tensors.
    def f1():
        return -((x1[0] - x2[0] - 1)**2 +
                 (x1[1] - x2[0])**2) - 100. * torch.exp(40. * (x1[0] - 4))

    def f2():
        return -((x1[0] - 2.)**2 +
                 (x1[1] - 4.)**2) - (x2[0] - 6.)**2

    optimizer = NestedMaximizerTorch(f1, [x1], f2, [x2])

    # Bounds for x2: you had bounds=[(0., 10.)]
    bounds = [(0., 10.)] * x2.numel()

    opt_x2, f2_max, info = optimizer.maximize(bounds=bounds)

    print("x2 =", x2.detach().numpy())
    print("x1 =", x1.detach().numpy())
    print("f1(x1*, x2) =", f1().item())
    print("f2(x1*, x2) =", f2().item())
