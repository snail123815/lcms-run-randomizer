"""Tests for pure scoring functions and TransitionTracker."""

import pytest

from randomize_samples_for_lcmsms import (
    TransitionTracker,
    get_affected_transitions,
    score_sequence,
    score_transition,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

W1 = [1.0]
W3 = [1.0, 1.0, 1.0]
W3_UNEVEN = [2.0, 1.0, 0.5]

# Three single-group items: A, B, C
GA = ("A",)
GB = ("B",)
GC = ("C",)

# Three two-group items
G_AB = ("A", "X")
G_BB = ("B", "X")  # same group-1
G_BC = ("B", "Y")


# ---------------------------------------------------------------------------
# score_transition
# ---------------------------------------------------------------------------


class TestScoreTransition:
    def test_all_differ(self):
        assert score_transition(("A", "B", "C"), ("X", "Y", "Z"), W3) == 3.0

    def test_none_differ(self):
        assert score_transition(("A", "B"), ("A", "B"), [1.0, 1.0]) == 0.0

    def test_partial_differ_unit_weights(self):
        # Only group 1 differs
        assert score_transition(("A", "B"), ("A", "Z"), [1.0, 1.0]) == 1.0

    def test_partial_differ_uneven_weights(self):
        # Groups 0 and 2 differ, group 1 same
        result = score_transition(("A", "X", "C"), ("B", "X", "D"), W3_UNEVEN)
        assert result == pytest.approx(2.0 + 0.5)

    def test_single_group(self):
        assert score_transition(GA, GB, W1) == 1.0
        assert score_transition(GA, GA, W1) == 0.0


# ---------------------------------------------------------------------------
# score_sequence
# ---------------------------------------------------------------------------


class TestScoreSequence:
    GROUPS = [GA, GB, GC]  # indices 0=A, 1=B, 2=C

    def test_empty_order(self):
        assert score_sequence([], self.GROUPS, W1) == 0.0

    def test_single_item_no_prefix(self):
        assert score_sequence([0], self.GROUPS, W1) == 0.0

    def test_two_items(self):
        # A→B: differ → 1.0
        assert score_sequence([0, 1], self.GROUPS, W1) == 1.0

    def test_three_items_no_prefix(self):
        # A→B + B→C = 2.0
        assert score_sequence([0, 1, 2], self.GROUPS, W1) == 2.0

    def test_prefix_last_included(self):
        # prefix=C, order=[A, B]  →  C→A + A→B = 2.0
        assert score_sequence([0, 1], self.GROUPS, W1, prefix_last=2) == 2.0

    def test_prefix_last_same_as_first(self):
        # prefix=A, order=[A, B]  →  A→A (0) + A→B (1) = 1.0
        assert score_sequence([0, 1], self.GROUPS, W1, prefix_last=0) == 1.0

    def test_prefix_last_empty_order(self):
        # prefix is irrelevant when order is empty
        assert score_sequence([], self.GROUPS, W1, prefix_last=0) == 0.0


# ---------------------------------------------------------------------------
# get_affected_transitions
# ---------------------------------------------------------------------------


class TestGetAffectedTransitions:
    # 4-item sequence: indices 0,1,2,3 mapping to groups A,B,C,D
    GROUPS = [("A",), ("B",), ("C",), ("D",)]

    def _check_symmetric_pair_count(self, old, new, expected_count):
        assert len(old) == expected_count
        assert len(new) == expected_count

    # ── Adjacent swap (j == i+1) ─────────────────────────────────────────

    def test_adjacent_middle_no_prefix(self):
        # order=[0,1,2,3], swap positions 1 and 2 → affects (0→1), (1→2), (2→3)
        old, new = get_affected_transitions(
            [0, 1, 2, 3], self.GROUPS, 1, 2, None
        )
        self._check_symmetric_pair_count(old, new, 3)
        assert (("A",), ("B",)) in old  # 0→1
        assert (("B",), ("C",)) in old  # 1→2
        assert (("C",), ("D",)) in old  # 2→3
        # new: (0→2), (2→1), (1→3)
        assert (("A",), ("C",)) in new
        assert (("C",), ("B",)) in new
        assert (("B",), ("D",)) in new

    def test_adjacent_at_start_no_prefix(self):
        # swap positions 0 and 1 → only (0→1) and (1→2); no left neighbor
        old, new = get_affected_transitions(
            [0, 1, 2, 3], self.GROUPS, 0, 1, None
        )
        self._check_symmetric_pair_count(old, new, 2)

    def test_adjacent_at_end(self):
        # swap positions 2 and 3 → only (1→2) and (2→3); no right neighbor
        old, new = get_affected_transitions(
            [0, 1, 2, 3], self.GROUPS, 2, 3, None
        )
        self._check_symmetric_pair_count(old, new, 2)

    def test_adjacent_with_prefix(self):
        # prefix=groups[4]=(E,), order=[0,1,2], swap 0 and 1
        groups_ext = self.GROUPS + [("E",)]
        old, new = get_affected_transitions(
            [0, 1, 2], groups_ext, 0, 1, prefix_last=4
        )
        self._check_symmetric_pair_count(old, new, 3)
        # prefix→0 should appear in old
        assert (("E",), ("A",)) in old
        # prefix→1 should appear in new
        assert (("E",), ("B",)) in new

    # ── Non-adjacent swap ────────────────────────────────────────────────

    def test_non_adjacent_middle(self):
        # order=[0,1,2,3], swap positions 0 and 2 → 4 transitions each
        old, new = get_affected_transitions(
            [0, 1, 2, 3], self.GROUPS, 0, 2, None
        )
        # position 0 has no left neighbor → 3 not 4
        self._check_symmetric_pair_count(old, new, 3)

    def test_non_adjacent_inner(self):
        # 5-item sequence, swap positions 1 and 3 → full 4 transitions each
        groups5 = [("A",), ("B",), ("C",), ("D",), ("E",)]
        old, new = get_affected_transitions(
            [0, 1, 2, 3, 4], groups5, 1, 3, None
        )
        self._check_symmetric_pair_count(old, new, 4)


# ---------------------------------------------------------------------------
# TransitionTracker
# ---------------------------------------------------------------------------


class TestTransitionTrackerBasic:
    def test_add_single_transition(self):
        t = TransitionTracker(1)
        t.add(("A",), ("B",))
        assert t.T[0][("A", "B")] == 1
        assert t.sum_sq[0] == 1.0

    def test_add_same_transition_twice(self):
        t = TransitionTracker(1)
        t.add(("A",), ("B",))
        t.add(("A",), ("B",))
        assert t.T[0][("A", "B")] == 2
        assert t.sum_sq[0] == 4.0  # 2² = 4

    def test_add_two_different_transitions(self):
        t = TransitionTracker(1)
        t.add(("A",), ("B",))  # T[A→B]=1
        t.add(("B",), ("A",))  # T[B→A]=1
        assert t.sum_sq[0] == 2.0  # 1² + 1² = 2

    def test_same_value_pair_ignored(self):
        t = TransitionTracker(1)
        t.add(("A",), ("A",))
        assert t.T[0] == {}
        assert t.sum_sq[0] == 0.0

    def test_remove_restores_state(self):
        t = TransitionTracker(1)
        t.add(("A",), ("B",))
        t.add(("A",), ("B",))
        t.remove(("A",), ("B",))
        assert t.T[0][("A", "B")] == 1
        assert t.sum_sq[0] == 1.0

    def test_remove_to_zero_cleans_entry(self):
        t = TransitionTracker(1)
        t.add(("A",), ("B",))
        t.remove(("A",), ("B",))
        assert ("A", "B") not in t.T[0]
        assert t.sum_sq[0] == 0.0

    def test_multi_group(self):
        t = TransitionTracker(2)
        t.add(("A", "X"), ("B", "Y"))
        assert t.T[0][("A", "B")] == 1
        assert t.T[1][("X", "Y")] == 1
        assert t.sum_sq[0] == 1.0
        assert t.sum_sq[1] == 1.0


class TestTransitionTrackerCopy:
    def test_copy_is_independent(self):
        t = TransitionTracker(1)
        t.add(("A",), ("B",))
        c = t.copy()
        c.add(("A",), ("B",))
        # Original should still have count 1
        assert t.T[0][("A", "B")] == 1
        assert c.T[0][("A", "B")] == 2

    def test_copy_preserves_state(self):
        t = TransitionTracker(1)
        t.add(("A",), ("B",))
        t.add(("B",), ("C",))
        c = t.copy()
        assert c.T[0] == t.T[0]
        assert c.sum_sq == t.sum_sq


class TestTransitionTrackerBalancePenalty:
    def test_zero_when_empty(self):
        t = TransitionTracker(2)
        assert t.balance_penalty([1.0, 1.0]) == 0.0

    def test_unit_weights(self):
        t = TransitionTracker(1)
        t.add(("A",), ("B",))
        t.add(("A",), ("B",))  # T[A→B]=2 → sum_sq=4
        assert t.balance_penalty([1.0]) == 4.0

    def test_non_unit_weights(self):
        t = TransitionTracker(2)
        t.add(("A", "X"), ("B", "Y"))  # sum_sq=[1,1]
        assert t.balance_penalty([3.0, 2.0]) == pytest.approx(5.0)

    def test_zero_weight_skipped(self):
        t = TransitionTracker(2)
        t.add(("A", "X"), ("B", "Y"))
        assert t.balance_penalty([0.0, 1.0]) == 1.0


class TestTransitionTrackerDeltaBalancePenalty:
    """delta_balance_penalty must be consistent with direct add/remove."""

    def _make_tracker_with_transitions(self, transitions):
        t = TransitionTracker(1)
        for ga, gb in transitions:
            t.add(ga, gb)
        return t

    def _consistency_check(self, tracker, old_trans, new_trans, weights):
        old_bp = tracker.balance_penalty(weights)
        delta = tracker.delta_balance_penalty(old_trans, new_trans, weights)
        for ga, gb in old_trans:
            tracker.remove(ga, gb)
        for ga, gb in new_trans:
            tracker.add(ga, gb)
        new_bp = tracker.balance_penalty(weights)
        assert new_bp == pytest.approx(old_bp + delta, abs=1e-10)

    def test_add_one_remove_one(self):
        t = self._make_tracker_with_transitions([(("A",), ("B",))])
        old = [(("A",), ("B",))]
        new = [(("B",), ("A",))]
        self._consistency_check(t, old, new, [1.0])

    def test_same_pair_in_old_and_new_no_change(self):
        t = TransitionTracker(1)
        t.add(("A",), ("B",))
        old = [(("A",), ("B",))]
        new = [(("A",), ("B",))]
        delta = t.delta_balance_penalty(old, new, [1.0])
        assert delta == pytest.approx(0.0)

    def test_increase_count_increases_penalty(self):
        t = TransitionTracker(1)
        t.add(("A",), ("B",))  # T[A→B]=1
        # Adding another A→B: old=[], new=[A→B] → delta = 2²-1² = 3
        delta = t.delta_balance_penalty([], [(("A",), ("B",))], [1.0])
        assert delta == pytest.approx(3.0)

    def test_multi_group_consistency(self):
        t = TransitionTracker(2)
        t.add(("A", "X"), ("B", "Y"))
        t.add(("A", "X"), ("B", "Y"))
        old = [(("A", "X"), ("B", "Y"))]
        new = [(("B", "X"), ("A", "Y"))]
        self._consistency_check(t, old, new, [1.0, 1.0])

    def test_does_not_mutate_tracker(self):
        t = TransitionTracker(1)
        t.add(("A",), ("B",))
        before_sq = list(t.sum_sq)
        before_T = {k: v for k, v in t.T[0].items()}
        t.delta_balance_penalty([(("A",), ("B",))], [(("B",), ("C",))], [1.0])
        assert t.sum_sq == before_sq
        assert t.T[0] == before_T
