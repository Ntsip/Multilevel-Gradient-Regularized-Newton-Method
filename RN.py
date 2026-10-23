"""Full-space gradient-regularized Newton with lemma-based adaptive L.

"""

import math
import time
from types import SimpleNamespace

import torch
from torch.func import jacrev


def _elapsed(start, device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return time.perf_counter() - start


def gradient_regularized_newton(
    objective, x0, *, L0=1.0, L_min=1e-12,
    backtracking_factor=2.0, max_ls=40, max_iters=100,
    tolg=1e-10, max_time=100.0, print_every=1,
    f_star=None, tolf=None,
):
    """Always compute the fine-space regularized Newton step.

    At iteration k, solve
        (H_k + alpha I) s = -g_k,
        alpha = sqrt(L_trial * ||g_k|| / 2).

    Increase L_trial until both fine-step inequalities hold:
        f(x_k)-f(x_k+s) >= ||g(x_k+s)||^2 / (6 alpha),
        ||g(x_k+s)|| <= 2 ||g_k||.

    Optional f_star and tolf enable objective-gap stopping.
    Returns the same SimpleNamespace fields as regularized_multilevel_newton;
    iter_hist has [gradient norm, objective, elapsed time, alpha].
    """
    if not (L0 > 0 and L_min > 0 and backtracking_factor > 1):
        raise ValueError('L0, L_min > 0 and backtracking_factor > 1 required.')
    if max_ls < 1 or max_iters < 0 or max_time <= 0:
        raise ValueError('Invalid iteration, line-search, or time limit.')
    # A common, independently computed reference value is optional.
    # Both f_star and tolf must be supplied to enable objective-gap stopping.
    if (f_star is None) != (tolf is None):
        raise ValueError('Pass both f_star and tolf, or neither.')
    if f_star is not None:
        f_star = float(f_star)
        tolf = float(tolf)
        if not (math.isfinite(f_star) and math.isfinite(tolf) and tolf > 0):
            raise ValueError('f_star must be finite and tolf must be positive.')
    if x0.ndim != 1 or not x0.is_floating_point():
        raise ValueError('x0 must be a one-dimensional floating-point tensor.')
    if x0.dtype != torch.float64:
        raise ValueError('Use float64 for reliable second derivatives/backtracking.')

    x = x0.detach().clone()
    grad_fun = jacrev(objective)
    hessian_fun = jacrev(grad_fun)
    eps_float = torch.finfo(x.dtype).eps

    # ---------------------------------------------------------
    # Warm-up: excluded from optimization wall-clock timing
    # ---------------------------------------------------------
    for _ in range(2):
        _ = grad_fun(x)
        H_warm = hessian_fun(x).detach()

        # Warm up the Cholesky backend as well.
        I = torch.eye(
            x.numel(), dtype=x.dtype, device=x.device
        )
        _ = torch.linalg.cholesky_ex(
            0.5 * (H_warm + H_warm.T) + I
        )

    if x.device.type == 'cuda':
        torch.cuda.synchronize()

    # Begin the actual measured optimization
    start = time.perf_counter()

    value_evals = gradient_evals = hessian_evals = factorizations = 0
    with torch.no_grad():
        f = objective(x)
        value_evals += 1
    g = grad_fun(x).detach()
    gradient_evals += 1
    if not bool(torch.isfinite(f)) or not bool(torch.isfinite(g).all()):
        raise FloatingPointError('Initial objective/gradient is not finite.')
    gnorm = float(torch.linalg.vector_norm(g))
    records = [[gnorm, float(f), 0.0, 0.0]]
    L_hist = [float(L0)]
    level_hist = [0]
    backtracking_hist = [0]
    decrease_margin_hist = [float('nan')]
    stability_ratio_hist = [float('nan')]
    count_fine = 0
    L_previous = float(L0)
    status = 'max_iters'
    def stopping_reason():
        if f_star is not None:
            gap = float(f) - f_star
            # If the algorithm meaningfully beats the reference, the reference
            # cannot be trusted as an approximation of the global minimum.
            reference_slack = 32 * eps_float * max(1.0, abs(float(f)), abs(f_star))
            if gap < -reference_slack:
                return 'reference_inaccurate'
            if gap <= tolf:
                return 'tolf'
            if gnorm <= tolg:
                # Stationary to tolg but reference gap not yet achieved.
                return 'tolg_before_tolf'
        elif gnorm <= tolg:
            return 'tolg'
        return None

    if print_every:
        print(f'iter={0:5d}  level={0:2d}  f={float(f):.8e} '
              f' ||g||={gnorm:.4e}  t=0.00s')

    for k in range(max_iters):
        elapsed = _elapsed(start, x.device)
        reason = stopping_reason()
        if reason is not None:
            status = reason
            break
        if elapsed >= max_time:
            status = 'max_time'
            break

        H = hessian_fun(x).detach()
        hessian_evals += 1
        H = 0.5 * (H + H.T)
        if not bool(torch.isfinite(H).all()):
            raise FloatingPointError(f'Non-finite Hessian at iteration {k}.')

        L_trial = max(float(L_min), L0 if k == 0
                      else L_previous / backtracking_factor)
        accepted = False
        precision_rejections = 0
        for j in range(max_ls):
            if _elapsed(start, x.device) >= max_time:
                status = 'max_time'
                break
            alpha = math.sqrt(L_trial * gnorm / 2.0)
            shifted = H + alpha * torch.eye(H.shape[0], dtype=x.dtype, device=x.device)
            chol, info = torch.linalg.cholesky_ex(shifted)
            factorizations += 1
            if int(info) != 0:
                L_trial *= backtracking_factor
                continue
            step = torch.cholesky_solve((-g).unsqueeze(-1), chol).squeeze(-1)
            x_trial = x + step
            with torch.no_grad():
                f_trial = objective(x_trial)
                value_evals += 1
            if not bool(torch.isfinite(f_trial)):
                L_trial *= backtracking_factor
                continue

            g_trial = grad_fun(x_trial).detach()
            gradient_evals += 1
            if not bool(torch.isfinite(g_trial).all()):
                L_trial *= backtracking_factor
                continue
            norm_next = float(torch.linalg.vector_norm(g_trial))
            required_decrease = norm_next**2 / (6.0 * alpha)
            actual_decrease = float(f - f_trial)
            roundoff = 16.0 * eps_float * max(1.0, abs(float(f)), abs(float(f_trial)))
            decrease_margin = actual_decrease - required_decrease
            stability_ratio = norm_next / gnorm
            accepted = (decrease_margin >= 0.0 and
                        stability_ratio <= 2.0 + 16.0 * eps_float)
            if accepted:
                break
            if (stability_ratio <= 2.0 + 16.0 * eps_float and
                    abs(decrease_margin) <= roundoff and
                    abs(actual_decrease) <= roundoff):
                precision_rejections += 1
            else:
                precision_rejections = 0
            if precision_rejections >= 5:
                status = 'numerical_precision'
                break
            L_trial *= backtracking_factor

        # if status in ('max_time', 'numerical_precision'):
        #     break

        if status in ('max_time',):
            break
        if not accepted:
            raise RuntimeError(
                f'Fine-step lemma-based backtracking failed after {max_ls} '
                f'attempts at iteration {k}; try a larger max_ls, L0, '
                'or smoother/rescaled data.'
            )

        x = x_trial.detach()
        f = f_trial.detach()
        g = g_trial
        gnorm = float(torch.linalg.vector_norm(g))
        elapsed = _elapsed(start, x.device)
        L_previous = L_trial
        count_fine += 1
        records.append([gnorm, float(f), elapsed, alpha])
        level_hist.append(1)
        L_hist.append(L_trial)
        backtracking_hist.append(j)
        decrease_margin_hist.append(decrease_margin)
        stability_ratio_hist.append(stability_ratio)
        if print_every and ((k + 1) % print_every == 0):
            print(f'iter={k+1:5d}  level={1:2d} '
                  f' f={float(f):.8e}  ||g||={gnorm:.4e} '
                  f' alpha={alpha:.3e}  L={L_trial:.3e} '
                  f' backtracks={j}  t={elapsed:.2f}s')
    else:
        # The last permitted update may itself reach tolf.
        reason = stopping_reason()
        if reason is not None:
            status = reason
        elif _elapsed(start, x.device) >= max_time:
            status = 'max_time'

    return SimpleNamespace(
        x=x, f=f, g=g, status=status, iterations=len(records)-1,
        count_coarse=0, count_fine=count_fine,
        iter_hist=torch.tensor(records, dtype=torch.float64),
        L_hist=torch.tensor(L_hist, dtype=torch.float64),
        level_hist=torch.tensor(level_hist, dtype=torch.long),
        backtracking_hist=torch.tensor(backtracking_hist, dtype=torch.long),
        value_evaluations=value_evals,
        gradient_evaluations=gradient_evals,
        hessian_evaluations=hessian_evals,
        factorizations=factorizations,
        decrease_margin_hist=torch.tensor(decrease_margin_hist, dtype=torch.float64),
        stability_ratio_hist=torch.tensor(stability_ratio_hist, dtype=torch.float64),
    )
