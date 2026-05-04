#!/usr/bin/env python3
"""
randomize_samples_for_lcmsms.py

Randomize a sample list for LC-MS/MS runs, maximizing the diversity of
consecutive condition transitions so that each unique condition value is
followed by every other value with approximately equal frequency.

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
Combined score = diversity_score - lambda * balance_penalty  (higher is better).

  diversity_score   For each consecutive pair (A, B): sum of weights[g] for
                    all groups g where A[g] != B[g].  Rewards items that
                    differ in as many condition dimensions as possible.

  balance_penalty   For each group g: sum_{a != b} T_g[a,b]^2  where
                    T_g[a,b] counts how many times value a was immediately
                    followed by value b in group g.  Penalises over-represented
                    specific (a -> b) pairs, pushing all directed transitions
                    toward equal frequency.

  lambda            Set automatically to 1 / (n_prefix_transitions +
                    n_cross_boundary + row_group_size - 1), i.e. the total
                    number of transitions in the full sequence up to and
                    including the current row_group.  This keeps both terms
                    on the same numerical scale as the sequence grows.

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


def delta_score_swap(
    order: list[int],
    groups: list[tuple[str, ...]],
    weights: list[float],
    i: int,
    j: int,
    prefix_last: Optional[int] = None,
) -> float:
    """O(1) diversity-score change from swapping positions *i* and *j*."""
    if i > j:
        i, j = j, i
    old, new = get_affected_transitions(order, groups, i, j, prefix_last)
    return sum(score_transition(a, b, weights) for a, b in new) - sum(
        score_transition(a, b, weights) for a, b in old
    )


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
# Simulated Annealing
# ---------------------------------------------------------------------------


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
) -> tuple[list[int], float]:
    """
    Find a high-scoring ordering of *indices* using Simulated Annealing.

    Optimises the combined score:

        diversity_score  -  lambda * balance_penalty

    where lambda = 1 / max(1, n_total_transitions) and n_total_transitions
    includes both the already-fixed prefix and the current row_group.

    Parameters
    ----------
    indices              Items to order (indices into *groups* / *items*).
    groups               Full group-tuple list for all items.
    weights              Per-group scoring weights.
    seed                 RNG seed (ensures reproducibility).
    max_iter             Maximum number of swap proposals.
    prefix_last          Index of the last item in the already-fixed prefix,
                         or None.  Used so the cross-boundary transition is
                         included in scoring.
    prefix_tracker       TransitionTracker pre-loaded with all transitions from
                         already-fixed row_groups.  A copy is taken so the
                         caller's tracker is never mutated.
    n_prefix_transitions Number of transitions already recorded in
                         *prefix_tracker* (used to scale lambda correctly).
    T_min                Temperature floor; below this only improvements are
                         accepted.

    Returns
    -------
    best_order  Indices in the best ordering found.
    best_score  Combined score of best_order.
    """
    rng = random.Random(seed)
    n = len(indices)

    if n == 0:
        return [], 0.0
    if n == 1:
        return list(indices), score_sequence(
            list(indices), groups, weights, prefix_last
        )

    # Total transitions = prefix + cross-boundary (if any) + within current group.
    n_cross = 1 if prefix_last is not None else 0
    n_total_trans = n_prefix_transitions + n_cross + (n - 1)

    # Lambda scales the balance penalty relative to the full sequence length so
    # that lambda * balance_penalty stays on the same order as diversity_score.
    lambda_balance = 1.0 / max(1, n_total_trans)

    # T_start covers typical diversity deltas; balance term is same scale by design.
    T_start = max(1.0, sum(weights))

    # Compute alpha so T decays from T_start to T_min over 80 % of max_iter.
    cooling_steps = max(1, int(0.8 * max_iter))
    alpha = (T_min / T_start) ** (1.0 / cooling_steps)

    # Random initial ordering.
    order = list(indices)
    rng.shuffle(order)

    # Initialise tracker: start from a copy of the prefix tracker (so prefix
    # transition counts act as background for the balance penalty), then add
    # the current row_group's transitions on top.
    tracker = (
        prefix_tracker.copy()
        if prefix_tracker is not None
        else TransitionTracker(len(weights))
    )
    if prefix_last is not None:
        tracker.add(groups[prefix_last], groups[order[0]])
    for k in range(n - 1):
        tracker.add(groups[order[k]], groups[order[k + 1]])

    diversity = score_sequence(order, groups, weights, prefix_last)
    current_score = diversity - lambda_balance * tracker.balance_penalty(
        weights
    )
    best_order = list(order)
    best_score = current_score

    T = T_start
    stagnation_limit = max(2000, 20 * n)
    stagnation_counter = 0

    for _ in range(max_iter):
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
        delta = delta_div - lambda_balance * delta_bal

        # Accept if improvement, or probabilistically while temperature is warm.
        accept = delta > 0 or (T > T_min and rng.random() < math.exp(delta / T))

        found_new_best = False
        if accept:
            order[i], order[j] = order[j], order[i]
            current_score += delta
            # Update tracker: remove transitions that no longer exist, add new ones.
            for ga, gb in old_trans:
                tracker.remove(ga, gb)
            for ga, gb in new_trans:
                tracker.add(ga, gb)
            if current_score > best_score + 1e-9:
                best_score = current_score
                best_order = list(order)
                stagnation_counter = 0
                found_new_best = True

        if not found_new_best:
            stagnation_counter += 1
            if stagnation_counter >= stagnation_limit:
                break

        T = max(T * alpha, T_min)

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
            "Max SA iterations per row_group (default: 50 000 × row_group size). "
            "Stagnation-based early stopping applies regardless of this limit."
        ),
    )
    p.add_argument(
        "input",
        nargs="?",
        metavar="FILE",
        default="to_randomise.txt",
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
    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

LARGE_LIST_THRESHOLD = 20


def main() -> None:
    args = build_parser().parse_args()

    # ── Read & parse input ─────────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Error: '{input_path.resolve()}' not found.")

    items = read_items(input_path)
    if not items:
        sys.exit(f"Error: no items found in '{input_path}'.")

    groups = parse_groups(items)
    n_groups = len(groups[0])

    # Print seed information to stderr for reproducibility.
    print(f"# Seed: {args.seed}", file=sys.stderr)

    # Print group summary to stderr.
    unique_per_group = [
        unique_ordered([g[gi] for g in groups]) for gi in range(n_groups)
    ]
    print(f"# {len(items)} items, {n_groups} group(s):", file=sys.stderr)
    for gi, u in enumerate(unique_per_group):
        print(f"#   Group {gi}: {u}", file=sys.stderr)

    # ── Parse weights ──────────────────────────────────────────────────────
    if args.weight is not None:
        try:
            weights = [float(w) for w in args.weight.split(",")]
        except ValueError:
            sys.exit(
                "Error: --weight must be comma-separated numbers (e.g. '1,2,1')."
            )
        if len(weights) != n_groups:
            sys.exit(
                f"Error: --weight has {len(weights)} value(s); "
                f"expected {n_groups} (one per group)."
            )
    else:
        weights = [1.0] * n_groups

    # ── Parse --fix-sort ───────────────────────────────────────────────────
    fix_indices: list[int] = []
    if args.fix_sort is not None:
        try:
            fix_indices = [int(x) for x in args.fix_sort.strip().split(",")]
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
    final_order: list[int] = []
    prefix_last: Optional[int] = None
    global_tracker = TransitionTracker(
        len(weights)
    )  # accumulates all fixed transitions
    global_trans_count = 0  # number of transitions recorded in global_tracker

    for k, rg_indices in enumerate(row_groups):
        n_rg = len(rg_indices)
        max_iter = (
            args.max_iter
            if args.max_iter is not None
            else 50_000 * max(n_rg, 1)
        )

        # Each row_group gets a deterministic but distinct seed derived from the
        # global seed so results are independent across groups yet fully reproducible.
        seed_k = args.seed + k

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
        )

        # Record this row_group's transitions into the global tracker so the
        # next row_group's SA sees the full history.
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
