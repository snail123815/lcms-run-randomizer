#!/usr/bin/env python3
"""
randomize_samples_for_lcmsms.py

Randomize a sample list for LC-MS/MS runs, maximizing the diversity of
consecutive condition transitions so that each unique condition value is
followed by every other value with approximately equal frequency.

Randomization is meant to reduce the impact of carryover by mixing different
conditions in consecutive runs. Time-related confounders (e.g. instrument drift)
is not a consideration here, but will benefit indirectly from the more even
distribution of transitions.

Algorithm
---------
Simulated Annealing (SA) with O(1) incremental score updates per swap proposal.

Why SA instead of Monte Carlo random restarts?
  • SA makes directed improvements from a good state via O(1) incremental deltas.
  • SA escapes local optima through temperature-driven probabilistic acceptance.
  • For N=24 SA converges in well under a second; Monte Carlo needs thousands of
    independent restarts to reach comparable quality.
  • For N > 20, use --fix-sort to decompose the problem into smaller subproblems.

When --fix-sort is used, each row_group is optimised sequentially.  The
TransitionTracker accumulated from all previously fixed row_groups is passed
into the next SA call as a frozen background.  This ensures that the balance
penalty for row_group k steers the SA away from a→b pairs that were already
over-represented in row_groups 0 … k-1, resulting in a globally even
distribution of directed transitions across the full run sequence.

Score
-----
Combined score = diversity_score  -  lambda_bal  * balance_penalty
                                  +  lambda_time * spread_bonus   (higher is better).

  diversity_score   For each consecutive pair (A, B): sum of weights[g] for
                    all groups g where A[g] != B[g].  Rewards items that
                    differ in as many condition dimensions as possible.

  balance_penalty   For each group g: sum_{a != b} T_g[a,b]^2  where
                    T_g[a,b] counts how many times value a was immediately
                    followed by value b in group g.  Penalises over-represented
                    specific (a -> b) pairs, pushing all directed transitions
                    toward equal frequency.

  spread_bonus      For each group g and each unique value v in g: normalized
                    position variance, 4 * Var(positions of v) / n_total^2.
                    Rewards items with the same condition value appearing spread
                    evenly across the run, mitigating instrument-drift confounding.

  lambda_bal        Set automatically to 1 / (n_prefix_transitions +
                    n_cross_boundary + row_group_size - 1).  Keeps diversity
                    and balance on the same numerical scale.

  lambda_time       Controlled by --priority:
                    carryover (default): lambda_time = lambda_bal * 1e-4
                      Spread breaks ties only; carryover reduction dominates.
                    time: lambda_time = lambda_bal; lambda_bal *= 1e-4
                      Spread is the primary objective; balance is tie-breaker.

Input / Output
--------------
Input : file specified via --input (default: 'to_randomise.txt' in the current
        working directory).  Only the first whitespace-separated token on each
        non-empty line is used.  Each token is split on '_' to obtain per-group values.
Output: randomized list, one item per line, to stdout.
Info  : diagnostic / progress messages to stderr.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def read_items(filepath: Path) -> list[str]:
    """Return the first whitespace-token from each non-empty line of *filepath*."""
    items: list[str] = []
    with open(filepath, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if parts:
                items.append(parts[0])
    return items


def parse_groups(items: list[str]) -> list[tuple[str, ...]]:
    """
    Split every item on '_' and return as a list of group-tuples.
    Exits with an error message if items have inconsistent group counts.
    """
    groups = [tuple(item.split("_")) for item in items]
    expected = len(groups[0])
    for idx, g in enumerate(groups):
        if len(g) != expected:
            sys.exit(
                f"Error: '{items[idx]}' (index {idx}) has {len(g)} group(s); "
                f"expected {expected} (same as first item '{items[0]}')."
            )
    return groups


def unique_ordered(seq: list[str]) -> list[str]:
    """Return unique values from *seq* in first-occurrence order."""
    seen: set[str] = set()
    result: list[str] = []
    for v in seq:
        if v not in seen:
            seen.add(v)
            result.append(v)
    return result


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_transition(
    ga: tuple[str, ...],
    gb: tuple[str, ...],
    weights: list[float],
) -> float:
    """Sum of weights[g] for all groups g where ga[g] != gb[g]."""
    return sum(w for w, a, b in zip(weights, ga, gb) if a != b)


def score_sequence(
    order: list[int],
    groups: list[tuple[str, ...]],
    weights: list[float],
    prefix_last: Optional[int] = None,
) -> float:
    """
    Total sequence score (higher = more diverse consecutive pairs).

    If *prefix_last* is provided, the transition from the prefix tail to
    order[0] is included in the score.
    """
    total = 0.0
    if prefix_last is not None and order:
        total += score_transition(
            groups[prefix_last], groups[order[0]], weights
        )
    for k in range(len(order) - 1):
        total += score_transition(
            groups[order[k]], groups[order[k + 1]], weights
        )
    return total


def get_affected_transitions(
    order: list[int],
    groups: list[tuple[str, ...]],
    i: int,
    j: int,
    prefix_last: Optional[int],
) -> tuple[
    list[tuple[tuple[str, ...], tuple[str, ...]]],
    list[tuple[tuple[str, ...], tuple[str, ...]]],
]:
    """
    Return (old_transitions, new_transitions) for swapping positions *i* and *j*.

    Each transition is a (ga, gb) pair of group-tuples.  Boundary pairs where
    one side is None (start of sequence with no prefix, or end of sequence) are
    excluded.  Caller must ensure i < j.
    """
    n = len(order)

    def get_g(pos: int) -> Optional[tuple[str, ...]]:
        if pos < 0:
            return groups[prefix_last] if prefix_last is not None else None
        if pos >= n:
            return None
        return groups[order[pos]]

    ai = groups[order[i]]
    aj = groups[order[j]]

    if j == i + 1:
        # Adjacent: three transitions (i-1→i), (i→j), (j→j+1)
        prev = get_g(i - 1)
        nxt = get_g(j + 1)
        old_raw = [(prev, ai), (ai, aj), (aj, nxt)]
        new_raw = [(prev, aj), (aj, ai), (ai, nxt)]
    else:
        # Non-adjacent: four transitions
        old_raw = [
            (get_g(i - 1), ai),
            (ai, get_g(i + 1)),
            (get_g(j - 1), aj),
            (aj, get_g(j + 1)),
        ]
        new_raw = [
            (get_g(i - 1), aj),
            (aj, get_g(i + 1)),
            (get_g(j - 1), ai),
            (ai, get_g(j + 1)),
        ]

    old = [(a, b) for a, b in old_raw if a is not None and b is not None]
    new = [(a, b) for a, b in new_raw if a is not None and b is not None]
    return old, new


def print_transition_stats(
    order: list[int],
    groups: list[tuple[str, ...]],
    unique_per_group: list[list[str]],
    prefix_last: Optional[int],
    label: str,
    skip_groups: Optional[list[int]] = None,
) -> None:
    """
    Print per-group directed transition frequency tables to stderr.

    For every group g (not in *skip_groups*) and every ordered pair (a, b)
    of its unique values (including a == b), prints the count and percentage
    out of the total transitions.  Zero-count pairs are omitted.
    Groups in *skip_groups* are fixed by --fix-sort and therefore trivially
    constant within a row_group; they are skipped with a note.
    """
    skip_groups = skip_groups or []
    pairs: list[tuple[int, int]] = []
    if prefix_last is not None and order:
        pairs.append((prefix_last, order[0]))
    for k in range(len(order) - 1):
        pairs.append((order[k], order[k + 1]))

    n_trans = len(pairs)
    if n_trans == 0:
        return

    n_groups = len(groups[0])
    counts: list[dict[tuple[str, str], int]] = [{} for _ in range(n_groups)]
    for idx_a, idx_b in pairs:
        for g in range(n_groups):
            key = (groups[idx_a][g], groups[idx_b][g])
            counts[g][key] = counts[g].get(key, 0) + 1

    print(f"# {label}  ({n_trans} transition(s))", file=sys.stderr)
    if skip_groups:
        skipped = ", ".join(str(g) for g in skip_groups)
        print(
            f"#   (Groups {skipped} omitted — fixed by --fix-sort)",
            file=sys.stderr,
        )
    for g, uvals in enumerate(unique_per_group):
        if g in skip_groups:
            continue
        col_w = max(len(v) for v in uvals)
        cnt_w = len(str(n_trans))
        print(f"#   Group {g}:", file=sys.stderr)
        for a in uvals:
            for b in uvals:
                cnt = counts[g].get((a, b), 0)
                if cnt == 0:
                    continue
                pct = 100.0 * cnt / n_trans
                print(
                    f"#     {a:{col_w}} -> {b:{col_w}} :"
                    f" {cnt:{cnt_w}}/{n_trans} = {pct:5.1f}%",
                    file=sys.stderr,
                )


# ---------------------------------------------------------------------------
# Transition frequency tracker (for balance penalty)
# ---------------------------------------------------------------------------


class TransitionTracker:
    """
    Tracks directed transition counts T_g[a, b] per group and their
    sum-of-squares, used to compute and incrementally update the balance penalty.

    Only off-diagonal transitions (a != b) are recorded; same-value pairs
    already score zero in the diversity term and are not balanced further.
    """

    def __init__(self, n_groups: int) -> None:
        self.n_groups = n_groups
        # T[g][(a, b)] = number of times value a was immediately followed by b
        self.T: list[dict[tuple[str, str], int]] = [{} for _ in range(n_groups)]
        # sum_{a != b} T[g][(a, b)]^2 for each group g
        self.sum_sq: list[float] = [0.0] * n_groups

    def _update(self, g: int, a: str, b: str, delta: int) -> None:
        if a == b:
            return
        key = (a, b)
        old = self.T[g].get(key, 0)
        new_count = old + delta
        self.sum_sq[g] += new_count * new_count - old * old
        if new_count == 0:
            self.T[g].pop(key, None)
        else:
            self.T[g][key] = new_count

    def add(self, ga: tuple[str, ...], gb: tuple[str, ...]) -> None:
        """Record a new transition ga -> gb across all groups."""
        for g, (a, b) in enumerate(zip(ga, gb)):
            self._update(g, a, b, +1)

    def remove(self, ga: tuple[str, ...], gb: tuple[str, ...]) -> None:
        """Undo a previously recorded transition ga -> gb across all groups."""
        for g, (a, b) in enumerate(zip(ga, gb)):
            self._update(g, a, b, -1)

    def copy(self) -> "TransitionTracker":
        """Return an independent deep copy of this tracker."""
        new = TransitionTracker(self.n_groups)
        new.T = [{k: v for k, v in d.items()} for d in self.T]
        new.sum_sq = list(self.sum_sq)
        return new

    def balance_penalty(self, weights: list[float]) -> float:
        """Weighted sum of all T_g[a,b]^2 (lower = more evenly distributed)."""
        return sum(w * s for w, s in zip(weights, self.sum_sq))

    def delta_balance_penalty(
        self,
        old_transitions: list[tuple[tuple[str, ...], tuple[str, ...]]],
        new_transitions: list[tuple[tuple[str, ...], tuple[str, ...]]],
        weights: list[float],
    ) -> float:
        """
        O(1) change in balance_penalty from replacing old_transitions with
        new_transitions, without modifying the stored counts.

        Accumulates net changes per (a, b) pair before computing the delta so
        the same pair appearing in both lists is handled correctly.
        """
        net: list[dict[tuple[str, str], int]] = [
            {} for _ in range(self.n_groups)
        ]
        for ga, gb in old_transitions:
            for g, (a, b) in enumerate(zip(ga, gb)):
                if a != b:
                    key = (a, b)
                    net[g][key] = net[g].get(key, 0) - 1
        for ga, gb in new_transitions:
            for g, (a, b) in enumerate(zip(ga, gb)):
                if a != b:
                    key = (a, b)
                    net[g][key] = net[g].get(key, 0) + 1

        delta_total = 0.0
        for g, w in enumerate(weights):
            if w == 0.0:
                continue
            for key, dc in net[g].items():
                if dc != 0:
                    old_count = self.T[g].get(key, 0)
                    new_count = old_count + dc
                    delta_total += w * (
                        new_count * new_count - old_count * old_count
                    )
        return delta_total


# ---------------------------------------------------------------------------
# Position spread tracker (for spread bonus)
# ---------------------------------------------------------------------------


class SpreadTracker:
    """
    Tracks per-(group, value) position statistics to compute the spread_bonus:
    how evenly each condition value is distributed across the run positions.

    For each group g and unique value v, stores the sum and sum-of-squares of
    global run positions (position_offset + local_pos) where value v appears.
    The spread_score is the weighted, normalized sum of position variances:

        4 * Var(positions) / n_total^2

    (= 0.0 if count ≤ 1; maximum ≈ 1 per group-value pair.)

    All updates (delta_spread_score, apply_swap) are O(n_groups).
    """

    def __init__(
        self, n_groups: int, position_offset: int, n_total: int
    ) -> None:
        self.n_groups = n_groups
        self.position_offset = position_offset
        self.n_total = n_total
        self._norm = 4.0 / (n_total * n_total) if n_total > 1 else 0.0
        # sum_p[g][v]  = sum of global positions for value v in group g
        self.sum_p: list[dict[str, float]] = [{} for _ in range(n_groups)]
        # sum_p2[g][v] = sum of (global position)^2 for value v in group g
        self.sum_p2: list[dict[str, float]] = [{} for _ in range(n_groups)]
        # count[g][v]  = number of items with value v in group g
        self.count: list[dict[str, int]] = [{} for _ in range(n_groups)]

    @classmethod
    def from_order(
        cls,
        order: list[int],
        groups: list[tuple[str, ...]],
        n_groups: int,
        position_offset: int,
        n_total: int,
    ) -> "SpreadTracker":
        """Build a SpreadTracker from an existing ordering."""
        st = cls(n_groups, position_offset, n_total)
        for local_pos, idx in enumerate(order):
            gp = float(position_offset + local_pos)
            for g in range(n_groups):
                v = groups[idx][g]
                st.sum_p[g][v] = st.sum_p[g].get(v, 0.0) + gp
                st.sum_p2[g][v] = st.sum_p2[g].get(v, 0.0) + gp * gp
                st.count[g][v] = st.count[g].get(v, 0) + 1
        return st

    def _var(self, g: int, v: str) -> float:
        """Un-normalized variance of global positions for value v in group g."""
        c = self.count[g].get(v, 0)
        if c <= 1:
            return 0.0
        sp = self.sum_p[g][v]
        sp2 = self.sum_p2[g][v]
        mean = sp / c
        return sp2 / c - mean * mean

    def spread_score(self, weights: list[float]) -> float:
        """
        Weighted, normalized sum of position variances (higher = more spread).
        Each group-value pair contributes w_g * 4 * Var / n_total^2.
        Values with count ≤ 1 contribute zero.
        """
        total = 0.0
        for g, w in enumerate(weights):
            if w == 0.0:
                continue
            for v in self.count[g]:
                total += w * self._norm * self._var(g, v)
        return total

    def delta_spread_score(
        self,
        i: int,
        j: int,
        order: list[int],
        groups: list[tuple[str, ...]],
        weights: list[float],
    ) -> float:
        """
        O(n_groups) change in spread_score from swapping positions i and j,
        without modifying stored state.  Must be called before applying the swap.
        """
        p_i = float(self.position_offset + i)
        p_j = float(self.position_offset + j)
        delta = 0.0
        for g, w in enumerate(weights):
            if w == 0.0:
                continue
            va = groups[order[i]][g]
            vb = groups[order[j]][g]
            if va == vb:
                continue
            # va moves from p_i → p_j
            c_a = self.count[g].get(va, 0)
            if c_a >= 2:
                sp_a = self.sum_p[g][va]
                sp2_a = self.sum_p2[g][va]
                new_sp_a = sp_a - p_i + p_j
                new_sp2_a = sp2_a - p_i * p_i + p_j * p_j
                old_var_a = sp2_a / c_a - (sp_a / c_a) ** 2
                new_var_a = new_sp2_a / c_a - (new_sp_a / c_a) ** 2
                delta += w * self._norm * (new_var_a - old_var_a)
            # vb moves from p_j → p_i
            c_b = self.count[g].get(vb, 0)
            if c_b >= 2:
                sp_b = self.sum_p[g][vb]
                sp2_b = self.sum_p2[g][vb]
                new_sp_b = sp_b - p_j + p_i
                new_sp2_b = sp2_b - p_j * p_j + p_i * p_i
                old_var_b = sp2_b / c_b - (sp_b / c_b) ** 2
                new_var_b = new_sp2_b / c_b - (new_sp_b / c_b) ** 2
                delta += w * self._norm * (new_var_b - old_var_b)
        return delta

    def apply_swap(
        self,
        i: int,
        j: int,
        order: list[int],
        groups: list[tuple[str, ...]],
    ) -> None:
        """
        Update tracked sums to reflect swapping positions i and j.
        Must be called BEFORE modifying order[i] and order[j].
        """
        p_i = float(self.position_offset + i)
        p_j = float(self.position_offset + j)
        for g in range(self.n_groups):
            va = groups[order[i]][g]
            vb = groups[order[j]][g]
            if va == vb:
                continue
            # va moves from p_i to p_j
            self.sum_p[g][va] = self.sum_p[g][va] - p_i + p_j
            self.sum_p2[g][va] = self.sum_p2[g][va] - p_i * p_i + p_j * p_j
            # vb moves from p_j to p_i
            self.sum_p[g][vb] = self.sum_p[g][vb] - p_j + p_i
            self.sum_p2[g][vb] = self.sum_p2[g][vb] - p_j * p_j + p_i * p_i


# ---------------------------------------------------------------------------
# Simulated Annealing
# ---------------------------------------------------------------------------


def _sa_hyperparams(
    n: int,
    weights: list[float],
    n_total_trans: int,
    T_min: float,
) -> tuple[float, float, float, int]:
    """
    Derive SA hyper-parameters from problem dimensions.

    Returns
    -------
    lambda_balance   Penalty scaling factor (1 / n_total_trans).
    T_start          Initial temperature (≥ 1, covers max diversity delta).
    alpha            Geometric cooling factor per step.
    stagnation_limit Steps without a global-best improvement before restart.
    """
    lambda_balance = 1.0 / max(1, n_total_trans)
    T_start = max(1.0, sum(weights))
    restart_budget = max(10_000, 100 * n)
    cooling_steps = max(1, int(0.8 * restart_budget))
    alpha = (T_min / T_start) ** (1.0 / cooling_steps)
    stagnation_limit = max(restart_budget // 2, 1_000)
    return lambda_balance, T_start, alpha, stagnation_limit


def _build_restart_tracker(
    prefix_last: Optional[int],
    prefix_tracker: Optional["TransitionTracker"],
    order: list[int],
    groups: list[tuple[str, ...]],
    n_weights: int,
) -> "TransitionTracker":
    """
    Build a fresh TransitionTracker for one SA restart.

    Copies *prefix_tracker* (so the caller's tracker is never mutated) then
    layers the transitions of the current *order* on top.
    """
    tracker = (
        prefix_tracker.copy()
        if prefix_tracker is not None
        else TransitionTracker(n_weights)
    )
    if prefix_last is not None:
        tracker.add(groups[prefix_last], groups[order[0]])
    for ki in range(len(order) - 1):
        tracker.add(groups[order[ki]], groups[order[ki + 1]])
    return tracker


def _run_single_restart(
    rng: random.Random,
    order: list[int],
    groups: list[tuple[str, ...]],
    weights: list[float],
    prefix_last: Optional[int],
    prefix_tracker: Optional["TransitionTracker"],
    lambda_balance: float,
    T_start: float,
    alpha: float,
    T_min: float,
    stagnation_limit: int,
    budget: int,
    global_best_score: float,
    lambda_time: float = 0.0,
    position_offset: int = 0,
    n_total: Optional[int] = None,
) -> tuple[Optional[list[int]], float, int]:
    """
    Run one SA restart starting from *order* (modified in place).

    Stagnation is measured against *global_best_score* so the restart
    terminates early if no improvement over the overall best is found.
    If the initial shuffle already beats *global_best_score*, it is
    recorded immediately — matching the original pre-loop check.

    Returns
    -------
    best_order_if_improved   Best ordering found if it beats *global_best_score*;
                             None if no improvement was made.
    new_best_score           Highest score seen; equals *global_best_score* if
                             no improvement was found.
    iters_used               SA steps executed in this restart.

    Additional keyword arguments
    ----------------------------
    lambda_time      Scaling factor for the spread_bonus.  0.0 (default)
                     disables spread tracking entirely.
    position_offset  Global run index of the first item in *order* (default 0).
    n_total          Total items across the full run for normalization;
                     defaults to len(*order*).
    """
    tracker = _build_restart_tracker(
        prefix_last, prefix_tracker, order, groups, len(weights)
    )
    effective_n_total = n_total if n_total is not None else len(order)
    spread_tracker = (
        SpreadTracker.from_order(
            order, groups, len(weights), position_offset, effective_n_total
        )
        if lambda_time != 0.0
        else None
    )
    diversity = score_sequence(order, groups, weights, prefix_last)
    current_score = (
        diversity
        - lambda_balance * tracker.balance_penalty(weights)
        + (
            lambda_time * spread_tracker.spread_score(weights)
            if spread_tracker is not None
            else 0.0
        )
    )

    current_best = global_best_score
    best_order_found: Optional[list[int]] = None

    # Check whether the initial shuffle already beats the global best (mirrors
    # the pre-loop check in the original monolithic SA).
    if current_score > current_best:
        current_best = current_score
        best_order_found = list(order)

    T = T_start
    stagnation_counter = 0
    n = len(order)

    for iters_used in range(1, budget + 1):
        # Propose a swap of two distinct positions.
        i = rng.randrange(n)
        j = rng.randrange(n - 1)
        if j >= i:
            j += 1
        if i > j:
            i, j = j, i

        old_trans, new_trans = get_affected_transitions(
            order, groups, i, j, prefix_last
        )
        delta_div = sum(
            score_transition(a, b, weights) for a, b in new_trans
        ) - sum(score_transition(a, b, weights) for a, b in old_trans)
        delta_bal = tracker.delta_balance_penalty(old_trans, new_trans, weights)
        delta_spread = (
            spread_tracker.delta_spread_score(i, j, order, groups, weights)
            if spread_tracker is not None
            else 0.0
        )
        delta = (
            delta_div - lambda_balance * delta_bal + lambda_time * delta_spread
        )

        # Accept if improvement, or probabilistically while temperature is warm.
        accept = delta > 0 or (T > T_min and rng.random() < math.exp(delta / T))

        found_new_best = False
        if accept:
            if spread_tracker is not None:
                spread_tracker.apply_swap(i, j, order, groups)
            order[i], order[j] = order[j], order[i]
            current_score += delta
            for ga, gb in old_trans:
                tracker.remove(ga, gb)
            for ga, gb in new_trans:
                tracker.add(ga, gb)
            if current_score > current_best + 1e-9:
                current_best = current_score
                best_order_found = list(order)
                stagnation_counter = 0
                found_new_best = True

        if not found_new_best:
            stagnation_counter += 1
            if stagnation_counter >= stagnation_limit:
                return best_order_found, current_best, iters_used  # stagnated

        T = max(T * alpha, T_min)

    return best_order_found, current_best, budget


def simulated_annealing(
    indices: list[int],
    groups: list[tuple[str, ...]],
    weights: list[float],
    seed: int,
    max_iter: int,
    prefix_last: Optional[int] = None,
    prefix_tracker: Optional["TransitionTracker"] = None,
    n_prefix_transitions: int = 0,
    T_min: float = 1e-4,
    priority: str = "carryover",
    position_offset: int = 0,
    n_total: Optional[int] = None,
) -> tuple[list[int], float]:
    """
    Find a high-scoring ordering of *indices* using Simulated Annealing.

    Optimises:  diversity_score  -  lambda_bal  * balance_penalty
                                 +  lambda_time * spread_bonus    (higher = better).
    Restarts from a fresh random ordering whenever stagnation is detected;
    the global *max_iter* budget is shared across all restarts.

    Parameters
    ----------
    indices              Items to order (indices into *groups* / *items*).
    groups               Full group-tuple list for all items.
    weights              Per-group scoring weights.
    seed                 RNG seed (ensures reproducibility).
    max_iter             Total iteration budget across all restarts.
    prefix_last          Index of the last item in the already-fixed prefix.
    prefix_tracker       TransitionTracker pre-loaded with prefix transitions.
    n_prefix_transitions Transitions already in *prefix_tracker*.
    T_min                Temperature floor; below this only improvements accepted.
    priority             'carryover' (default): spread is a secondary tie-breaker.
                         'time': spread is the primary objective; balance is the
                         secondary tie-breaker.
    position_offset      Global run index of the first item in *indices*
                         (0 when not using --fix-sort; supplied automatically
                         by _optimize_row_groups for each row_group).
    n_total              Total number of items across all row_groups (for spread
                         normalization).  Defaults to len(*indices*).

    Returns
    -------
    best_order  Indices in the best ordering found.
    best_score  Combined score of *best_order*.
    """
    rng = random.Random(seed)
    n = len(indices)

    if n == 0:
        return [], 0.0
    if n == 1:
        return list(indices), score_sequence(
            list(indices), groups, weights, prefix_last
        )

    n_cross = 1 if prefix_last is not None else 0
    n_total_trans = n_prefix_transitions + n_cross + (n - 1)
    lambda_balance, T_start, alpha, stagnation_limit = _sa_hyperparams(
        n, weights, n_total_trans, T_min
    )
    effective_n_total = n_total if n_total is not None else n
    if priority == "time":
        lambda_time = lambda_balance
        lambda_balance = lambda_balance * 1e-4
    else:  # "carryover" (default)
        lambda_time = lambda_balance * 1e-4

    best_order: list[int] = []
    best_score = -math.inf
    total_iters = 0
    n_restarts = 0

    while total_iters < max_iter:
        order = list(indices)
        rng.shuffle(order)
        improved_order, new_best_score, iters_used = _run_single_restart(
            rng,
            order,
            groups,
            weights,
            prefix_last,
            prefix_tracker,
            lambda_balance,
            T_start,
            alpha,
            T_min,
            stagnation_limit,
            budget=max_iter - total_iters,
            global_best_score=best_score,
            lambda_time=lambda_time,
            position_offset=position_offset,
            n_total=effective_n_total,
        )
        total_iters += iters_used
        n_restarts += 1
        if improved_order is not None:
            best_score = new_best_score
            best_order = improved_order

    print(
        f"#   SA: {n_restarts} restart(s), {total_iters} total iteration(s)",
        file=sys.stderr,
    )
    return best_order, best_score


# ---------------------------------------------------------------------------
# Row-group construction
# ---------------------------------------------------------------------------


def build_row_groups(
    groups: list[tuple[str, ...]],
    fix_indices: list[int],
) -> tuple[list[list[int]], list[tuple[str, ...]]]:
    """
    Partition item indices into row_groups based on *fix_indices*.

    Each unique combination of group values at the fixed group indices forms
    one row_group.  Row_group keys are ordered by first occurrence in the
    original input file.

    Returns
    -------
    row_groups  List of index buckets, one per unique key.
    key_order   Corresponding unique keys (in first-occurrence order).
    """
    key_order: list[tuple[str, ...]] = []
    key_seen: set[tuple[str, ...]] = set()
    buckets: dict[tuple[str, ...], list[int]] = {}

    for idx, g in enumerate(groups):
        key = tuple(g[fi] for fi in fix_indices)
        if key not in key_seen:
            key_seen.add(key)
            key_order.append(key)
            buckets[key] = []
        buckets[key].append(idx)

    return [buckets[k] for k in key_order], key_order


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _parse_weights(weight_str: str, n_groups: int) -> list[float]:
    """
    Parse a comma-separated weight string into a validated list of floats.
    Exits with an error if the format is invalid or the count mismatches.
    """
    try:
        weights = [float(w) for w in weight_str.split(",")]
    except ValueError:
        sys.exit(
            "Error: --weight must be comma-separated numbers (e.g. '1,2,1')."
        )
    if len(weights) != n_groups:
        sys.exit(
            f"Error: --weight has {len(weights)} value(s); "
            f"expected {n_groups} (one per group)."
        )
    return weights


def _parse_fix_sort(fix_sort_str: str, n_groups: int) -> list[int]:
    """
    Parse a comma-separated index string into a validated list of group indices.
    Exits with an error if the format is invalid or any index is out of range.
    """
    try:
        fix_indices = [int(x) for x in fix_sort_str.strip().split(",")]
    except ValueError:
        sys.exit(
            "Error: --fix-sort must be comma-separated integers (e.g. '2' or '2,0')."
        )
    for fi in fix_indices:
        if fi < 0 or fi >= n_groups:
            sys.exit(
                f"Error: --fix-sort index {fi} is out of range "
                f"(groups are numbered 0–{n_groups - 1})."
            )
    return fix_indices


def _optimize_row_groups(
    groups: list[tuple[str, ...]],
    weights: list[float],
    row_groups: list[list[int]],
    rg_keys: list[tuple[str, ...]],
    unique_per_group: list[list[str]],
    fix_indices: list[int],
    seed: int,
    max_iter_arg: Optional[int],
    priority: str = "carryover",
) -> list[int]:
    """
    Run Simulated Annealing sequentially over each row_group and return the
    concatenated final ordering of all item indices.

    Each row_group's SA receives the accumulated TransitionTracker from all
    previous row_groups as a frozen background, steering it away from
    transition pairs that are already over-represented.
    """
    final_order: list[int] = []
    prefix_last: Optional[int] = None
    global_tracker = TransitionTracker(len(weights))
    global_trans_count = 0

    for k, rg_indices in enumerate(row_groups):
        n_rg = len(rg_indices)
        max_iter = (
            max_iter_arg if max_iter_arg is not None else 50_000 * max(n_rg, 1)
        )
        # Each row_group gets a deterministic but distinct seed derived from the
        # global seed so results are independent across groups yet fully reproducible.
        seed_k = seed + k
        rg_prefix_last = prefix_last  # capture before advancing

        best_order, best_score = simulated_annealing(
            rg_indices,
            groups,
            weights,
            seed=seed_k,
            max_iter=max_iter,
            prefix_last=prefix_last,
            prefix_tracker=global_tracker,
            n_prefix_transitions=global_trans_count,
            priority=priority,
            position_offset=len(final_order),
            n_total=len(groups),
        )

        # Record this row_group's transitions so the next row_group's SA sees
        # the full history.
        if prefix_last is not None and best_order:
            global_tracker.add(groups[prefix_last], groups[best_order[0]])
            global_trans_count += 1
        for m in range(len(best_order) - 1):
            global_tracker.add(groups[best_order[m]], groups[best_order[m + 1]])
        global_trans_count += max(len(best_order) - 1, 0)

        print(
            f"# Row_group [{k}] key={rg_keys[k]}  "
            f"score={best_score:.4f}  size={n_rg}  max_iter={max_iter}",
            file=sys.stderr,
        )
        if len(row_groups) > 1:
            print_transition_stats(
                best_order,
                groups,
                unique_per_group,
                rg_prefix_last,
                f"Row_group [{k}] key={rg_keys[k]}",
                skip_groups=fix_indices,
            )

        final_order.extend(best_order)
        prefix_last = best_order[-1] if best_order else prefix_last

    return final_order


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="randomize_samples_for_lcmsms.py",
        description=(
            "Randomize a sample list for LC-MS/MS, maximizing the diversity of\n"
            "consecutive condition transitions so that each unique condition value\n"
            "is followed by every other value with approximately equal frequency.\n\n"
            "Input : positional input file (default: 'to_randomise.txt' in the current\n"
            "         working directory; only the first whitespace-separated token per\n"
            "         line is used)\n"
            "Output: randomized list, one item per line, to stdout.\n"
            "Info  : diagnostic / progress output to stderr.\n\n"
            "Algorithm: Simulated Annealing with O(1) incremental score updates.\n"
            "SA outperforms Monte Carlo random restarts by making directed\n"
            "improvements and escaping local optima via temperature acceptance."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--fix-sort",
        metavar="GROUPS",
        default=None,
        help=(
            "Comma-separated group indices used to pre-partition the list into "
            "row_groups before randomizing (e.g. '2' or '2,0'). "
            "SA is run sequentially per row_group: each group is optimized with "
            "the tail of the previous fixed group as cross-boundary context. "
            "Row_groups preserve the first-occurrence order from the input file."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="INT",
        help="Random seed (required — ensures full reproducibility).",
    )
    p.add_argument(
        "--weight",
        metavar="WEIGHTS",
        default=None,
        help=(
            "Comma-separated per-group weights, e.g. '1,2,1'. "
            "Higher weight means that group's transitions contribute more to the score. "
            "Default: 1.0 for every group."
        ),
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "Total SA iteration budget per row_group across all restarts "
            "(default: 50 000 × row_group size). "
            "When stagnation is detected the run restarts from a new random ordering; "
            "the budget counts all iterations across restarts."
        ),
    )
    p.add_argument(
        "input",
        type=Path,
        metavar="FILE",
        help=(
            "Path to the input file containing sample names (positional). "
            "Default: 'to_randomise.txt' in the current directory."
        ),
    )
    p.add_argument(
        "--no-warn",
        action="store_true",
        help="Suppress the large-list (> 20 items) warning.",
    )
    p.add_argument(
        "--priority",
        choices=["carryover", "time"],
        default="carryover",
        metavar="{carryover,time}",
        help=(
            "Optimization priority (default: carryover). "
            "'carryover': minimize run-order carryover; temporal spread of "
            "each condition is a secondary tie-breaker. "
            "'time': maximize temporal spread of each condition across the run "
            "to reduce instrument-drift confounding; transition balance is the "
            "secondary tie-breaker."
        ),
    )
    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

LARGE_LIST_THRESHOLD = 20


def main() -> None:
    args = build_parser().parse_args()

    # ── Read & parse input ─────────────────────────────────────────────────
    input_path = args.input
    if not input_path.exists():
        sys.exit(f"Error: '{input_path.resolve()}' not found.")

    items = read_items(input_path)
    if not items:
        sys.exit(f"Error: no items found in '{input_path}'.")

    groups = parse_groups(items)
    n_groups = len(groups[0])

    print(f"# Seed: {args.seed}", file=sys.stderr)
    unique_per_group = [
        unique_ordered([g[gi] for g in groups]) for gi in range(n_groups)
    ]
    print(f"# {len(items)} items, {n_groups} group(s):", file=sys.stderr)
    for gi, u in enumerate(unique_per_group):
        print(f"#   Group {gi}: {u}", file=sys.stderr)

    # ── Parse weights & fix-sort ───────────────────────────────────────────
    weights = (
        _parse_weights(args.weight, n_groups)
        if args.weight is not None
        else [1.0] * n_groups
    )
    fix_indices = (
        _parse_fix_sort(args.fix_sort, n_groups)
        if args.fix_sort is not None
        else []
    )

    # ── Large-list warning ─────────────────────────────────────────────────
    if (
        not args.no_warn
        and len(items) > LARGE_LIST_THRESHOLD
        and not fix_indices
    ):
        print(
            f"Warning: {len(items)} items with no --fix-sort specified. "
            "Using --fix-sort decomposes the problem into smaller subproblems, "
            "dramatically improving both speed and solution quality. "
            "Example: --fix-sort 2  (use --no-warn to suppress this message)",
            file=sys.stderr,
        )

    # ── Build row groups ───────────────────────────────────────────────────
    if fix_indices:
        row_groups, rg_keys = build_row_groups(groups, fix_indices)
        print(
            f"# --fix-sort {args.fix_sort!r}: {len(row_groups)} row_group(s):",
            file=sys.stderr,
        )
        for k, (key, rg) in enumerate(zip(rg_keys, row_groups)):
            print(f"#   [{k}] key={key}  ({len(rg)} items)", file=sys.stderr)
    else:
        row_groups = [list(range(len(items)))]
        rg_keys = [("all",)]

    # ── Simulated Annealing per row group ──────────────────────────────────
    final_order = _optimize_row_groups(
        groups,
        weights,
        row_groups,
        rg_keys,
        unique_per_group,
        fix_indices,
        args.seed,
        args.max_iter,
        args.priority,
    )

    # ── Overall transition stats ───────────────────────────────────────────
    print_transition_stats(
        final_order,
        groups,
        unique_per_group,
        prefix_last=None,
        label="Overall",
        skip_groups=fix_indices,
    )

    # ── Output ─────────────────────────────────────────────────────────────
    for idx in final_order:
        print(items[idx])


if __name__ == "__main__":
    main()
