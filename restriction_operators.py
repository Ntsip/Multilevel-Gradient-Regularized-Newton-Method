"""Index-based restrictions R_i (coordinate rows of the identity).

A list of index tensors encodes restriction: R_i v = v[indices_i].
P_i is the transpose, implemented by an index insertion. No dense
restriction matrices are built. The full level is appended at the end
for compatibility with the supplied build_balanced_subsets function.
"""

import torch


def _validate_sizes(N, sizes):
    sizes = [int(v) for v in sizes]
    if not 0 < len(sizes) or any(not (1 <= s < N) for s in sizes):
        raise ValueError('Need at least one coarse level with 1 <= size < N.')
    if any(a >= b for a, b in zip(sizes, sizes[1:])):
        raise ValueError('Coarse dimensions must be strictly increasing.')
    return sizes


def dimensions_from_fractions(N, fractions=(0.1, 0.2, 0.3, 0.4, 0.5)):
    """Convert user-adjustable fractions into strictly increasing dimensions."""
    fractions = tuple(float(t) for t in fractions)
    if not fractions:
        return []  # Empty hierarchy means full-space Newton only.
    if any(not (0 < t < 1) for t in fractions):
        raise ValueError('Fractions must be strictly between 0 and 1.')
    return _validate_sizes(N, [max(1, round(t * N)) for t in fractions])


def build_nested_subsets(N, sizes, *, generator=None):
    """Exactly row-nested operators, R_{i+1} = [R_i; extra rows].

    One random permutation fixes the row ordering of every level.
    In particular index lists must NOT be sorted separately: prefixes
    preserve the actual matrix row-nesting, not just set inclusion.
    """
    sizes = _validate_sizes(N, sizes)
    permutation = torch.randperm(N, generator=generator)
    return [permutation[:s].clone() for s in sizes] + [torch.arange(N)]


def build_balanced_subsets(N, m, low=None, high=None, *,
                           sizes=None, generator=None):
    """Keep the supplied balanced-coverage, non-nested construction.

    Backwards-compatible name and first two arguments. For the original
    evenly-spaced sizes, pass low/high. To choose dimensions explicitly,
    pass sizes (length m). The final torch.arange(N) is the fine level.
    Randomness can be reproduced by passing a CPU torch.Generator.
    """
    if sizes is None:
        if low is None or high is None:
            raise ValueError('Specify low/high or sizes.')
        # Preserve original equally spaced sizes and int rounding.
        sizes = torch.linspace(int(low), int(high), steps=m).round().unique(
            sorted=True).to(torch.int64).tolist()
    sizes = _validate_sizes(N, sizes)
    if len(sizes) != m:
        raise ValueError('m must equal the number of distinct coarse dimensions.')

    covered = torch.zeros(N, dtype=torch.bool)
    usage = torch.zeros(N, dtype=torch.int64)
    subsets = []
    for size in sizes:
        subset_mask = torch.zeros(N, dtype=torch.bool)
        if not bool(covered.all()):
            unc_idx = torch.nonzero(~covered).flatten()
            available = int(unc_idx.numel())
            if available <= size:
                subset_mask[unc_idx] = True
                covered[unc_idx] = True
                usage[unc_idx] += 1
                needed = size - available
                if needed:
                    avail = torch.nonzero(~subset_mask).flatten()
                    pick = avail[torch.randperm(len(avail), generator=generator)[:needed]]
                    subset_mask[pick] = True
                    usage[pick] += 1
            else:
                pick = unc_idx[torch.randperm(available, generator=generator)[:size]]
                subset_mask[pick] = True
                covered[pick] = True
                usage[pick] += 1
        else:
            weights = 1.0 / (usage.to(torch.float64) + 1)
            pick = torch.multinomial(weights, size, replacement=False, generator=generator)
            subset_mask[pick] = True
            usage[pick] += 1
        subsets.append(torch.nonzero(subset_mask).flatten())
    return subsets + [torch.arange(N)]


def make_levels(N, fractions=(0.1, 0.2, 0.3, 0.4, 0.5),
                mode='nested', seed=43):
    """Returns l coarse index tensors plus the fine index tensor.

    mode='nested' produces row-nested restrictions.
    mode='non_nested' uses the original balanced-subset algorithm.
    fractions=() gives only the fine level (full regularized Newton).
    """
    sizes = dimensions_from_fractions(N, fractions)
    if not sizes:
        return [torch.arange(N)]
    rng = torch.Generator(device='cpu').manual_seed(int(seed))
    if mode == 'nested':
        return build_nested_subsets(N, sizes, generator=rng)
    if mode in ('non_nested', 'classical'):
        return build_balanced_subsets(N, len(sizes), sizes=sizes, generator=rng)
    raise ValueError("mode must be 'nested' or 'non_nested'.")
