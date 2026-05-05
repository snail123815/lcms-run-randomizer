"""Tests for SA hyper-parameters, restart helpers, and simulated_annealing."""

import math
import random

import pytest

from randomize_samples_for_lcmsms import (
    TransitionTracker,
    _build_restart_tracker,
    _run_single_restart,
    _sa_hyperparams,
    simulated_annealing,
)

# ---------------------------------------------------------------------------
# _sa_hyperparams
# ---------------------------------------------------------------------------


class TestSaHyperparams:
    def test_alpha_in_unit_interval(self):
        lam, T_start, alpha, stag = _sa_hyperparams(10, [1.0] * 4, 40, 1e-4)
        assert 0.0 < alpha < 1.0

    def test_lambda_balance_positive(self):
        lam, *_ = _sa_hyperparams(10, [1.0], 20, 1e-4)
        assert lam > 0.0

    def test_lambda_scales_with_n_total_trans(self):
        lam10, *_ = _sa_hyperparams(5, [1.0], 10, 1e-4)
        lam20, *_ = _sa_hyperparams(5, [1.0], 20, 1e-4)
        assert lam10 > lam20  # fewer transitions → larger lambda

    def test_lambda_n_total_trans_zero(self):
        # Should not raise ZeroDivisionError
        lam, *_ = _sa_hyperparams(5, [1.0], 0, 1e-4)
        assert lam == pytest.approx(1.0)

    def test_T_start_at_least_one(self):
        _, T_start, _, _ = _sa_hyperparams(10, [0.0001], 20, 1e-4)
        assert T_start >= 1.0

    def test_T_start_equals_sum_weights_when_large(self):
        weights = [3.0, 2.0]
        _, T_start, _, _ = _sa_hyperparams(10, weights, 20, 1e-4)
        assert T_start == pytest.approx(5.0)

    def test_stagnation_limit_at_least_1000(self):
        *_, stag = _sa_hyperparams(1, [1.0], 5, 1e-4)
        assert stag >= 1_000

    def test_stagnation_limit_scales_with_n(self):
        *_, stag_small = _sa_hyperparams(10, [1.0], 20, 1e-4)
        *_, stag_large = _sa_hyperparams(1000, [1.0], 20, 1e-4)
        assert stag_large >= stag_small


# ---------------------------------------------------------------------------
# _build_restart_tracker
# ---------------------------------------------------------------------------


class TestBuildRestartTracker:
    GROUPS = [("A",), ("B",), ("C",), ("D",)]

    def test_no_prefix_no_prefix_tracker(self):
        order = [0, 1, 2]
        t = _build_restart_tracker(None, None, order, self.GROUPS, 1)
        # Transitions: A→B, B→C
        assert t.T[0].get(("A", "B"), 0) == 1
        assert t.T[0].get(("B", "C"), 0) == 1
        assert t.T[0].get(("C", "D"), 0) == 0  # D not in order

    def test_with_prefix_last_adds_cross_boundary(self):
        order = [1, 2]  # B, C
        # prefix_last=0 (A) → adds A→B cross-boundary transition
        t = _build_restart_tracker(0, None, order, self.GROUPS, 1)
        assert t.T[0].get(("A", "B"), 0) == 1
        assert t.T[0].get(("B", "C"), 0) == 1

    def test_prefix_tracker_is_copied_not_mutated(self):
        prefix_t = TransitionTracker(1)
        prefix_t.add(("A",), ("B",))
        original_sum_sq = list(prefix_t.sum_sq)

        order = [2, 3]  # C, D
        _build_restart_tracker(None, prefix_t, order, self.GROUPS, 1)

        # Original tracker must be unchanged
        assert prefix_t.sum_sq == original_sum_sq

    def test_prefix_tracker_state_included(self):
        prefix_t = TransitionTracker(1)
        prefix_t.add(("A",), ("B",))  # background: A→B already happened

        order = [2, 3]  # C, D
        t = _build_restart_tracker(None, prefix_t, order, self.GROUPS, 1)

        assert t.T[0].get(("A", "B"), 0) == 1  # from prefix
        assert t.T[0].get(("C", "D"), 0) == 1  # from order

    def test_single_item_order(self):
        # Single item: no internal transitions to add
        t = _build_restart_tracker(None, None, [0], self.GROUPS, 1)
        assert sum(t.sum_sq) == 0.0


# ---------------------------------------------------------------------------
# _run_single_restart
# ---------------------------------------------------------------------------


class TestRunSingleRestart:
    # 4-item sequence: A, B, C, D with one group each
    GROUPS = [("A",), ("B",), ("C",), ("D",)]
    WEIGHTS = [1.0]
    INDICES = [0, 1, 2, 3]

    def _make_rng(self, seed=0):
        return random.Random(seed)

    def _run(self, order, seed=0, budget=50_000, global_best=-math.inf):
        from randomize_samples_for_lcmsms import _sa_hyperparams

        lam, T_start, alpha, stag = _sa_hyperparams(
            len(order), self.WEIGHTS, len(order) - 1, 1e-4
        )
        rng = self._make_rng(seed)
        return _run_single_restart(
            rng,
            order,
            self.GROUPS,
            self.WEIGHTS,
            prefix_last=None,
            prefix_tracker=None,
            lambda_balance=lam,
            T_start=T_start,
            alpha=alpha,
            T_min=1e-4,
            stagnation_limit=stag,
            budget=budget,
            global_best_score=global_best,
        )

    def test_respects_budget(self):
        order = list(self.INDICES)
        _, _, iters = self._run(order, budget=100)
        assert iters <= 100

    def test_impossible_global_best_returns_none_order(self):
        # With global_best=+inf, no improvement is possible
        order = list(self.INDICES)
        best_order, returned_score, _ = self._run(order, global_best=math.inf)
        assert best_order is None
        assert returned_score == math.inf

    def test_returns_improvement_when_initial_beats_global(self):
        # global_best=-inf → any order beats it → best_order is not None
        order = list(self.INDICES)
        best_order, score, _ = self._run(order, global_best=-math.inf)
        assert best_order is not None
        assert len(best_order) == len(self.INDICES)

    def test_output_is_permutation_of_input(self):
        order = list(self.INDICES)
        best_order, _, _ = self._run(order)
        if best_order is not None:
            assert sorted(best_order) == sorted(self.INDICES)

    def test_stagnation_triggers_early_exit(self):
        # Very tight stagnation limit with a large budget → exits early
        from randomize_samples_for_lcmsms import _sa_hyperparams

        lam, T_start, alpha, _ = _sa_hyperparams(
            len(self.INDICES), self.WEIGHTS, 3, 1e-4
        )
        rng = self._make_rng(42)
        order = list(self.INDICES)
        _, _, iters = _run_single_restart(
            rng,
            order,
            self.GROUPS,
            self.WEIGHTS,
            prefix_last=None,
            prefix_tracker=None,
            lambda_balance=lam,
            T_start=T_start,
            alpha=alpha,
            T_min=1e-4,
            stagnation_limit=1,  # trigger after every non-improving step
            budget=1_000_000,
            global_best_score=-math.inf,
        )
        assert iters < 1_000_000


# ---------------------------------------------------------------------------
# simulated_annealing
# ---------------------------------------------------------------------------


class TestSimulatedAnnealing:
    GROUPS_SMALL = [("A",), ("B",), ("C",), ("D",), ("E",), ("F",)]
    WEIGHTS = [1.0]

    def test_empty_indices(self):
        order, score = simulated_annealing(
            [], self.GROUPS_SMALL, self.WEIGHTS, 0, 100
        )
        assert order == []
        assert score == 0.0

    def test_single_index(self):
        order, score = simulated_annealing(
            [3], self.GROUPS_SMALL, self.WEIGHTS, 0, 100
        )
        assert order == [3]
        assert score == 0.0

    def test_deterministic_same_seed(self):
        indices = list(range(6))
        r1, s1 = simulated_annealing(
            indices, self.GROUPS_SMALL, self.WEIGHTS, seed=7, max_iter=5_000
        )
        r2, s2 = simulated_annealing(
            indices, self.GROUPS_SMALL, self.WEIGHTS, seed=7, max_iter=5_000
        )
        assert r1 == r2
        assert s1 == pytest.approx(s2)

    def test_different_seeds_may_differ(self):
        # Not guaranteed to differ every time, but with 6 diverse items it's
        # extremely unlikely that seed 0 and seed 99 produce the same order.
        indices = list(range(6))
        r1, _ = simulated_annealing(
            indices, self.GROUPS_SMALL, self.WEIGHTS, seed=0, max_iter=5_000
        )
        r2, _ = simulated_annealing(
            indices, self.GROUPS_SMALL, self.WEIGHTS, seed=99, max_iter=5_000
        )
        # They *might* agree by chance, but we can at least check both are valid
        assert sorted(r1) == indices
        assert sorted(r2) == indices

    def test_output_is_permutation(self):
        indices = list(range(6))
        order, _ = simulated_annealing(
            indices, self.GROUPS_SMALL, self.WEIGHTS, seed=42, max_iter=5_000
        )
        assert sorted(order) == indices

    def test_score_positive_for_diverse_items(self):
        # All items differ → any ordering has score > 0 (each transition = 1)
        indices = list(range(6))
        _, score = simulated_annealing(
            indices, self.GROUPS_SMALL, self.WEIGHTS, seed=1, max_iter=5_000
        )
        assert score > 0.0

    def test_prefix_last_used(self):
        # With a prefix, the score should include the cross-boundary transition
        indices = [1, 2, 3, 4, 5]
        groups = [("X",), ("A",), ("B",), ("C",), ("D",), ("E",)]
        order_no_prefix, score_no_prefix = simulated_annealing(
            indices, groups, [1.0], seed=0, max_iter=2_000
        )
        order_prefix, score_prefix = simulated_annealing(
            indices, groups, [1.0], seed=0, max_iter=2_000, prefix_last=0
        )
        # Both should be valid permutations of indices
        assert sorted(order_no_prefix) == indices
        assert sorted(order_prefix) == indices

    def test_carryover_priority_is_default(self):
        # Explicit priority="carryover" must give the same result as the default.
        indices = list(range(6))
        r_default, s_default = simulated_annealing(
            indices, self.GROUPS_SMALL, self.WEIGHTS, seed=5, max_iter=5_000
        )
        r_explicit, s_explicit = simulated_annealing(
            indices,
            self.GROUPS_SMALL,
            self.WEIGHTS,
            seed=5,
            max_iter=5_000,
            priority="carryover",
        )
        assert r_default == r_explicit
        assert s_default == pytest.approx(s_explicit)

    def test_time_priority_output_is_permutation(self):
        indices = list(range(6))
        order, _ = simulated_annealing(
            indices,
            self.GROUPS_SMALL,
            self.WEIGHTS,
            seed=3,
            max_iter=5_000,
            priority="time",
        )
        assert sorted(order) == indices

    def test_time_priority_deterministic(self):
        indices = list(range(6))
        r1, s1 = simulated_annealing(
            indices,
            self.GROUPS_SMALL,
            self.WEIGHTS,
            seed=11,
            max_iter=5_000,
            priority="time",
        )
        r2, s2 = simulated_annealing(
            indices,
            self.GROUPS_SMALL,
            self.WEIGHTS,
            seed=11,
            max_iter=5_000,
            priority="time",
        )
        assert r1 == r2
        assert s1 == pytest.approx(s2)
