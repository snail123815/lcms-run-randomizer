"""Tests for SpreadTracker: O(1) position-variance tracking for spread_bonus."""

import pytest

from randomize_samples_for_lcmsms import SpreadTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_st(order, groups, n_groups=None, offset=0, n_total=None):
    if n_groups is None:
        n_groups = len(groups[0]) if groups else 1
    if n_total is None:
        n_total = len(order)
    return SpreadTracker.from_order(order, groups, n_groups, offset, n_total)


# ---------------------------------------------------------------------------
# TestSpreadTrackerCreation
# ---------------------------------------------------------------------------


class TestSpreadTrackerCreation:
    # items: A at indices 0,2; B at index 1; C at index 3
    GROUPS = [("A",), ("B",), ("A",), ("C",)]

    def test_count_populated(self):
        st = _make_st([0, 1, 2, 3], self.GROUPS)
        assert st.count[0]["A"] == 2
        assert st.count[0]["B"] == 1
        assert st.count[0]["C"] == 1

    def test_sum_p_correct(self):
        # A at local positions 0 and 2 → sum_p = 0+2 = 2.0
        st = _make_st([0, 1, 2, 3], self.GROUPS)
        assert st.sum_p[0]["A"] == pytest.approx(2.0)
        assert st.sum_p[0]["B"] == pytest.approx(1.0)

    def test_sum_p2_correct(self):
        # A at positions 0,2 → sum_p2 = 0+4 = 4.0
        st = _make_st([0, 1, 2, 3], self.GROUPS)
        assert st.sum_p2[0]["A"] == pytest.approx(4.0)

    def test_position_offset_applied(self):
        groups = [("A",), ("B",)]
        st = _make_st([0, 1], groups, n_groups=1, offset=10, n_total=20)
        # A at global pos 10, B at global pos 11
        assert st.sum_p[0]["A"] == pytest.approx(10.0)
        assert st.sum_p[0]["B"] == pytest.approx(11.0)

    def test_single_item_spread_score_zero(self):
        # count=1 for every value → var=0 for all → spread_score=0
        groups = [("A",)]
        st = _make_st([0], groups)
        assert st.spread_score([1.0]) == pytest.approx(0.0)

    def test_multi_group(self):
        # Two groups: group0 = strain, group1 = fraction
        groups = [("Wt", "Cell"), ("FraA", "Ext"), ("Wt", "Cell")]
        st = _make_st([0, 1, 2], groups, n_groups=2)
        assert st.count[0]["Wt"] == 2
        assert st.count[0]["FraA"] == 1
        assert st.count[1]["Cell"] == 2
        assert st.count[1]["Ext"] == 1


# ---------------------------------------------------------------------------
# TestSpreadTrackerSpreadScore
# ---------------------------------------------------------------------------


class TestSpreadTrackerSpreadScore:
    def test_spread_higher_than_clustered(self):
        # 4 items: 2 × "A", 2 × "B"
        # Spread:   A B A B  →  A at pos 0,2  B at pos 1,3  (higher variance)
        # Clustered: A A B B → A at pos 0,1  B at pos 2,3  (lower variance)
        groups_spread = [("A",), ("B",), ("A",), ("B",)]
        groups_clust = [("A",), ("A",), ("B",), ("B",)]
        order = [0, 1, 2, 3]
        n_total = 4
        st_spread = _make_st(order, groups_spread, n_total=n_total)
        st_clust = _make_st(order, groups_clust, n_total=n_total)
        assert st_spread.spread_score([1.0]) > st_clust.spread_score([1.0])

    def test_all_distinct_values_score_zero(self):
        # Each value appears exactly once → count=1 → var=0
        groups = [("A",), ("B",), ("C",), ("D",)]
        st = _make_st([0, 1, 2, 3], groups, n_total=4)
        assert st.spread_score([1.0]) == pytest.approx(0.0)

    def test_zero_weight_contributes_nothing(self):
        groups = [("A",), ("A",)]
        st = _make_st([0, 1], groups)
        assert st.spread_score([0.0]) == pytest.approx(0.0)

    def test_weight_scales_score(self):
        groups = [("A",), ("B",), ("A",), ("B",)]
        st = _make_st([0, 1, 2, 3], groups, n_total=4)
        s1 = st.spread_score([1.0])
        s2 = st.spread_score([2.0])
        assert s2 == pytest.approx(2.0 * s1)

    def test_score_nonnegative(self):
        import random as _rng

        _r = _rng.Random(0)
        groups = [("A" if _r.random() < 0.5 else "B",) for _ in range(10)]
        st = _make_st(list(range(10)), groups, n_total=10)
        assert st.spread_score([1.0]) >= 0.0

    def test_manually_computed_variance(self):
        # A at positions 0,2: mean=1, var = (0+4)/2 - 1^2 = 2 - 1 = 1.0
        # B at positions 1,3: mean=2, var = (1+9)/2 - 2^2 = 5 - 4 = 1.0
        # norm = 4/16 = 0.25 → spread_score = 0.25*(1+1) = 0.5
        groups = [("A",), ("B",), ("A",), ("B",)]
        st = _make_st([0, 1, 2, 3], groups, n_total=4)
        assert st.spread_score([1.0]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# TestSpreadTrackerDelta
# ---------------------------------------------------------------------------


class TestSpreadTrackerDelta:
    # A appears at indices 0,2,4; B at 1,3 — all in a single group
    GROUPS5 = [("A",), ("B",), ("A",), ("B",), ("A",)]

    def _check_invariant(self, groups, order, i, j, weights, n_total=None):
        """Assert delta + old_score == new_score after apply_swap + order swap."""
        if n_total is None:
            n_total = len(order)
        st = _make_st(list(order), groups, n_total=n_total)
        old_score = st.spread_score(weights)
        delta = st.delta_spread_score(i, j, order, groups, weights)
        st.apply_swap(i, j, order, groups)
        order[i], order[j] = order[j], order[i]
        new_score = st.spread_score(weights)
        assert new_score == pytest.approx(old_score + delta, abs=1e-12)

    def test_non_adjacent_swap(self):
        order = [0, 1, 2, 3, 4]
        self._check_invariant(self.GROUPS5, order, 0, 4, [1.0])

    def test_adjacent_swap(self):
        order = [0, 1, 2, 3, 4]
        self._check_invariant(self.GROUPS5, order, 1, 2, [1.0])

    def test_swap_at_boundary(self):
        order = [0, 1, 2, 3, 4]
        self._check_invariant(self.GROUPS5, order, 0, 1, [1.0])

    def test_same_value_delta_is_zero(self):
        # Positions 0 and 2 both carry value "A" → no change
        order = [0, 1, 2, 3, 4]
        st = _make_st(list(order), self.GROUPS5)
        delta = st.delta_spread_score(0, 2, order, self.GROUPS5, [1.0])
        assert delta == pytest.approx(0.0)

    def test_delta_does_not_mutate_tracker(self):
        order = [0, 1, 2, 3, 4]
        st = _make_st(list(order), self.GROUPS5)
        score_before = st.spread_score([1.0])
        st.delta_spread_score(0, 3, order, self.GROUPS5, [1.0])
        assert st.spread_score([1.0]) == pytest.approx(score_before)

    def test_multi_group_invariant(self):
        # Two groups
        groups = [
            ("Wt", "Cell"),
            ("FraA", "Ext"),
            ("Wt", "Ext"),
            ("FraA", "Cell"),
            ("Wt", "Cell"),
        ]
        order = [0, 1, 2, 3, 4]
        self._check_invariant(groups, order, 1, 3, [1.0, 1.0], n_total=5)

    def test_with_position_offset(self):
        # Verify invariant when tracker has non-zero offset
        groups = [("A",), ("B",), ("A",), ("B",)]
        order = [0, 1, 2, 3]
        n_total = 8
        st = SpreadTracker.from_order(
            order, groups, 1, position_offset=4, n_total=n_total
        )
        old_score = st.spread_score([1.0])
        delta = st.delta_spread_score(0, 3, order, groups, [1.0])
        st.apply_swap(0, 3, order, groups)
        order[0], order[3] = order[3], order[0]
        new_score = st.spread_score([1.0])
        assert new_score == pytest.approx(old_score + delta, abs=1e-12)


# ---------------------------------------------------------------------------
# TestSpreadTrackerApplySwap
# ---------------------------------------------------------------------------


class TestSpreadTrackerApplySwap:
    def test_updates_sum_p(self):
        # A at pos 0, B at pos 1 → swap → A moves to pos 1, B moves to pos 0
        groups = [("A",), ("B",)]
        order = [0, 1]
        st = _make_st(order, groups, n_total=4)
        st.apply_swap(0, 1, order, groups)
        assert st.sum_p[0]["A"] == pytest.approx(1.0)
        assert st.sum_p[0]["B"] == pytest.approx(0.0)

    def test_updates_sum_p2(self):
        groups = [("A",), ("B",)]
        order = [0, 1]
        st = _make_st(order, groups, n_total=4)
        # Before: A.sum_p2=0, B.sum_p2=1; after swap: A.sum_p2=1, B.sum_p2=0
        st.apply_swap(0, 1, order, groups)
        assert st.sum_p2[0]["A"] == pytest.approx(1.0)
        assert st.sum_p2[0]["B"] == pytest.approx(0.0)

    def test_count_unchanged_after_swap(self):
        groups = [("A",), ("B",), ("A",)]
        order = [0, 1, 2]
        st = _make_st(order, groups, n_total=3)
        st.apply_swap(0, 2, order, groups)
        # A appears twice regardless of where
        assert st.count[0]["A"] == 2

    def test_same_value_no_change(self):
        groups = [("A",), ("A",)]
        order = [0, 1]
        st = _make_st(order, groups, n_total=4)
        sum_p_before = st.sum_p[0].get("A", 0.0)
        st.apply_swap(0, 1, order, groups)
        assert st.sum_p[0].get("A", 0.0) == pytest.approx(sum_p_before)

    def test_double_swap_restores_state(self):
        groups = [("A",), ("B",), ("A",), ("B",)]
        order = [0, 1, 2, 3]
        st = _make_st(list(order), groups, n_total=4)
        original_score = st.spread_score([1.0])
        # Swap forward
        st.apply_swap(0, 1, order, groups)
        order[0], order[1] = order[1], order[0]
        # Swap back
        st.apply_swap(0, 1, order, groups)
        order[0], order[1] = order[1], order[0]
        assert st.spread_score([1.0]) == pytest.approx(
            original_score, abs=1e-12
        )
