"""Multilevel Newton with gradient regularization and adaptive L_k.

Each R_i selects coordinates, P_i=R_i.T, and ||R_i||=1. For an
eligible coarse level i, the unit update is

    s_i = -(R_i H_k R_i.T + alpha_i I)^(-1) R_i g_k,
    alpha_i = sqrt(L_k * ||R_i g_k|| / 2),
    x_{k+1} = x_k + R_i.T s_i.

Otherwise use the full Newton Hessian and the same formula with R=I.
The smallest eligible level is used. L_k is estimated by increasing its
trial value until both a sufficient-decrease and a gradient-stability test
from Lemma 2.1 and (2.3)-(2.4) of the manuscript hold:

  coarse: f(x)-f(x+s) >= ||R_i g(x+s)||^2/(8*alpha_i)
          ||R_i g(x+s)|| <= 2||R_i g(x)||
  fine:   f(x)-f(x+s) >= ||g(x+s)||^2/(6*alpha)
          ||g(x+s)|| <= 2||g(x)||.

For Hessian-Lipschitz convex objectives, these hold for trial L_k at
least as large as the global Hessian Lipschitz constant. Computing the
trial gradient during backtracking adds an autodifferentiation call per
trial. No Armijo damping is used.
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


def _validate_levels(levels, n, device):
    if levels is None:
        # No coarse levels: full-space gradient-regularized Newton.
        return [torch.arange(n, dtype=torch.long, device=device)]
    if len(levels) < 1:
        raise ValueError('Expected at least the full fine-level index set.')
    result = []
    for indices in levels:
        ids = torch.as_tensor(indices, dtype=torch.long, device=device).reshape(-1)
        if ids.numel() == 0 or ids.min().item() < 0 or ids.max().item() >= n:
            raise ValueError('Invalid coarse indices.')
        if torch.unique(ids).numel() != ids.numel():
            raise ValueError('Repeated coordinate within a restriction operator.')
        result.append(ids)
    if result[-1].numel() != n or not torch.equal(
        result[-1].sort().values, torch.arange(n, device=device)
    ):
        raise ValueError('Last restriction must contain every coordinate (fine level).')
    if any(a.numel() >= b.numel() for a, b in zip(result, result[1:])):
        raise ValueError('Levels must have strictly increasing dimensions.')
    return result


def regularized_multilevel_newton(
    objective, x0, levels, *, sigma=0.1, epsilon=1e-10,
    L0=1.0, L_min=1e-12, backtracking_factor=2.0, max_ls=40,
    max_iters=100, tolg=1e-10, max_time=100.0, print_every=1,
    f_star=None, tolf=None,
):
    """Run one experiment; the calling run file controls all parameters.

    Parameters
    ----------
    objective : callable(x) -> scalar torch.Tensor
        An objective produced by objectives.make_objective, or any C^2 loss.
    x0 : torch.Tensor (n,)
        Starting vector, float64 is recommended.
    levels : list of torch.LongTensor or None
        Ascending coarse levels followed by the full fine level. Built by
        make_levels. None selects the fine level at every iteration.
    sigma, epsilon : float
        Level i is eligible if ||R_i g|| >= sigma ||g|| and >= epsilon.
    L0, L_min : float
        Initial and minimum estimates for the Lipschitz constant of H.
    backtracking_factor : float
        Multiplier for L on rejected lemma-based trials (typically 2).
    max_ls : int
        Maximum L trials per iteration. Raises an informative error if
        the trial step is not accepted; it never silently takes a failed step.

    Returns
    -------
    SimpleNamespace
      x, f, g, status, iterations, count_coarse, count_fine,
      iter_hist: CPU tensor [||grad||, f, elapsed seconds, regularizer alpha],
      L_hist, level_hist (1..ell for coarse, ell+1 for fine),
      backtracking_hist, decrease_margin_hist, stability_ratio_hist,
      value_evaluations, gradient_evaluations, hessian_evaluations,
      factorizations (counters for reproducibility).
      Histories of acceptance diagnostics include NaN for the initial point.
      If f_star and tolf are given, success means f(x_k)-f_star <= tolf.
    """
    # With coordinate restrictions ||R_i|| = 1. The manuscript requires
    # sigma in (0, 1) and epsilon strictly positive.
    if not (0 < sigma < 1) or not (epsilon > 0):
        raise ValueError('Algorithm 3.1 requires 0 < sigma < 1 and epsilon > 0.')
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
    all_levels = _validate_levels(levels, x.numel(), x.device)
    coarse_levels = all_levels[:-1]
    ncoarse = len(coarse_levels)
    grad_fun = jacrev(objective)

    full_hessian_fun = jacrev(grad_fun)
    eps_float = torch.finfo(x.dtype).eps

    # ---------------------------------------------------------
    # Warm-up: excluded from optimization wall-clock timing
    # ---------------------------------------------------------
    for _ in range(2):
        _ = grad_fun(x)
        H_warm = full_hessian_fun(x).detach()

        # Warm up the Cholesky backend as well.
        I = torch.eye(
            x.numel(), dtype=x.dtype, device=x.device
        )
        _ = torch.linalg.cholesky_ex(
            0.5 * (H_warm + H_warm.T) + I
        )

    if x.device.type == 'cuda':
        torch.cuda.synchronize()

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
    level_hist = [0]  # 0 denotes the initial point, not a step.
    backtracking_hist = [0]
    decrease_margin_hist = [float('nan')]
    stability_ratio_hist = [float('nan')]
    count_coarse = count_fine = 0
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

        # Select the coarsest (lowest dimensional) eligible level. No
        # gradient at x_{k+1} is needed for this eligibility condition.
        index = all_levels[-1]
        level_number = ncoarse + 1
        for i, ids in enumerate(coarse_levels):
            restricted_norm = float(torch.linalg.vector_norm(g[ids]))
            if restricted_norm >= sigma * gnorm and restricted_norm >= epsilon:
                index = ids
                level_number = i + 1
                break

        is_fine = level_number == ncoarse + 1
        restricted_g = g if is_fine else g[index]
        restricted_norm = float(torch.linalg.vector_norm(restricted_g))

        # Compute only the reduced Hessian on the chosen level.
        # For coarse restrictions, use an indexed differentiable copy
        # rather than mutating x in-place within jacrev.
        if is_fine:
            H = full_hessian_fun(x).detach()
        else:
            local = x[index].detach().clone()
            def local_objective(u):
                return objective(x.index_copy(0, index, u))
            H = jacrev(jacrev(local_objective))(local).detach()
        hessian_evals += 1
        H = 0.5 * (H + H.T)
        if not bool(torch.isfinite(H).all()):
            raise FloatingPointError(f'Non-finite Hessian at iteration {k}.')

        # Backtrack on L_k using the manuscript's sufficient decrease
        # (Lemma 2.1 / Lemma 3.1) AND restricted-gradient stability.
        # For fine steps use the full gradient. No step-length search.
        # The first iteration really starts with L0; only subsequent
        # iterations start at L_previous/backtracking_factor.
        L_trial = max(float(L_min), L0 if k == 0
                      else L_previous / backtracking_factor)
        accepted = False
        precision_rejections = 0
        for j in range(max_ls):
            # Enforce the wall-time budget inside backtracking as well.
            if _elapsed(start, x.device) >= max_time:
                status = 'max_time'
                break
            alpha = math.sqrt(L_trial * restricted_norm / 2.0)
            shifted = H + alpha * torch.eye(H.shape[0], dtype=x.dtype, device=x.device)
            chol, info = torch.linalg.cholesky_ex(shifted)
            factorizations += 1
            if int(info) != 0:  # may arise from nonconvex H / small alpha
                L_trial *= backtracking_factor
                continue
            step_reduced = torch.cholesky_solve(
                (-restricted_g).unsqueeze(-1), chol
            ).squeeze(-1)
            if is_fine:
                step_full = step_reduced
            else:
                step_full = torch.zeros_like(x).index_copy(0, index, step_reduced)
            x_trial = x + step_full
            with torch.no_grad():
                f_trial = objective(x_trial)
                value_evals += 1
            if not bool(torch.isfinite(f_trial)):
                L_trial *= backtracking_factor
                continue

            # In contrast to a cubic-model test, Lemma 2.1 needs a
            # trial gradient. One gradient suffices for both checks;
            # cache it so an accepted trial is not differentiated twice.
            g_trial = grad_fun(x_trial).detach()
            gradient_evals += 1
            if not bool(torch.isfinite(g_trial).all()):
                L_trial *= backtracking_factor
                continue
            trial_restricted = g_trial if is_fine else g_trial[index]
            norm_next = float(torch.linalg.vector_norm(trial_restricted))
            required_decrease = norm_next**2 / ((6.0 if is_fine else 8.0) * alpha)
            actual_decrease = float(f - f_trial)

            # Tiny absolute numerical tolerance for cancellation when
            # f(x) and f(x+s) are nearly equal at machine precision.
            roundoff = 16.0 * eps_float * max(1.0, abs(float(f)), abs(float(f_trial)))
            decrease_margin = actual_decrease - required_decrease
            stability_ratio = norm_next / restricted_norm
            accepted = (decrease_margin >= 0.0 and
                        stability_ratio <= 2.0 + 16.0 * eps_float)
            if accepted:
                break
            # A mathematically failed decrease test may not be reliably
            # distinguishable from roundoff near a stationary point. Do
            # NOT silently accept it as satisfying the lemma. Report
            # numerical_precision after several inconclusive trials.
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
                f'Lemma-based backtracking failed after {max_ls} attempts '
                f'at iteration {k}, level {level_number}; '
                'try a larger max_ls, L0, or smoother/rescaled data.'
            )

        x = x_trial.detach()
        f = f_trial.detach()
        g = g_trial
        gnorm = float(torch.linalg.vector_norm(g))
        elapsed = _elapsed(start, x.device)
        L_previous = L_trial
        if is_fine:
            count_fine += 1
        else:
            count_coarse += 1
        records.append([gnorm, float(f), elapsed, alpha])
        level_hist.append(level_number)
        L_hist.append(L_trial)
        backtracking_hist.append(j)
        decrease_margin_hist.append(decrease_margin)
        stability_ratio_hist.append(stability_ratio)

        if print_every and ((k+1) % print_every == 0):
            print(f'iter={k+1:5d}  level={level_number:2d} '
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

    # CPU histories preserve compatibility with the supplied plotting layout.
    return SimpleNamespace(
        x=x, f=f, g=g, status=status, iterations=len(records)-1,
        count_coarse=count_coarse, count_fine=count_fine,
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
