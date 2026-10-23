"""Shared differentiable objectives for optimization algorithms.

The loss formulas are retained from the supplied multilevel_newton.py.
The smoothed L1 term is NOT the nonsmooth ||x||_1 penalty.
"""

import torch
import torch.nn.functional as F


def make_objective(A, b, loss='quadratic', l1=0.0, l2=0.0,
                   smoothing=1e-3, huber_delta=0.01, rho=1.0):
    """Return f(x): scalar torch Tensor. A,b and x must use the same device/dtype.

    Supported losses: quadratic, log_sum, logistic, poisson, svm,
    huber, smooth_hinge. Quadratic, log_sum, logistic, poisson (without
    exp overflow), and the smoothed regularizer are C^2, as required
    by the Hessian-Lipschitz regularized Newton analysis. The squared
    hinge, Huber and smooth hinge options are not globally C^2.
    """
    names = {'quadratic', 'log_sum', 'logistic', 'poisson', 'svm',
             'huber', 'smooth_hinge'}
    if loss not in names:
        raise ValueError(f'Unknown loss {loss!r}; choose from {sorted(names)}')
    if l1 < 0 or l2 < 0 or smoothing <= 0 or huber_delta <= 0 or rho <= 0:
        raise ValueError('Regularization and smoothing parameters must be valid.')

    b = b.reshape(-1)
    if A.ndim != 2 or A.shape[0] != b.numel():
        raise ValueError('A.shape[0] must equal len(b).')
    if loss in {'logistic', 'svm', 'smooth_hinge'}:
        unique = torch.unique(b)
        if bool(torch.any((unique != -1) & (unique != 1))):
            raise ValueError(f'{loss} expects labels b in {{-1, +1}}; convert labels first.')
    if loss == 'poisson' and bool(torch.any(b < 0)):
        raise ValueError('Poisson response b must be nonnegative.')

    def objective(x):
        z = A @ x
        if loss == 'quadratic':
            data_loss = 0.5 * torch.sum((z - b).square())
        elif loss == 'log_sum':
            data_loss = rho * torch.logsumexp((z - b) / rho, dim=0)
        elif loss == 'logistic':
            data_loss = F.softplus(-b * z).mean()
        elif loss == 'poisson':
            data_loss = (torch.exp(z) - b * z).mean()
        elif loss == 'svm':
            data_loss = torch.clamp(1 - b * z, min=0).square().mean()
        elif loss == 'huber':
            residual = z - b
            abs_residual = torch.abs(residual)
            data_loss = torch.where(
                abs_residual <= huber_delta,
                0.5 * residual.square(),
                huber_delta * (abs_residual - 0.5 * huber_delta),
            ).mean()
        else:  # smooth_hinge, same piecewise formula as supplied source
            margin = b * z
            data_loss = torch.where(
                margin >= 1,
                torch.zeros_like(margin),
                torch.where(margin <= 0, 0.5 - margin,
                            0.5 * (1 - margin).square()),
            ).mean()

        regularizer = 0.5 * l2 * x.square().sum()
        if l1:
            regularizer = regularizer + l1 * (
                torch.sqrt(smoothing**2 + x.square()) - smoothing
            ).sum()
        return data_loss + regularizer

    return objective
