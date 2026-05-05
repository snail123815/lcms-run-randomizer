"""Tests for I/O helpers and row-group construction."""

import pytest

from randomize_samples_for_lcmsms import (
    build_row_groups,
    parse_groups,
    read_items,
    unique_ordered,
)

# ---------------------------------------------------------------------------
# read_items
# ---------------------------------------------------------------------------


class TestReadItems:
    def test_basic(self, tmp_path):
        f = tmp_path / "samples.txt"
        f.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        assert read_items(f) == ["alpha", "beta", "gamma"]

    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / "samples.txt"
        f.write_text("alpha\n\n   \nbeta\n", encoding="utf-8")
        assert read_items(f) == ["alpha", "beta"]

    def test_uses_first_token_only(self, tmp_path):
        f = tmp_path / "samples.txt"
        f.write_text("alpha extra ignored\nbeta\n", encoding="utf-8")
        assert read_items(f) == ["alpha", "beta"]

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        assert read_items(f) == []

    def test_single_line_no_newline(self, tmp_path):
        f = tmp_path / "single.txt"
        f.write_text("only", encoding="utf-8")
        assert read_items(f) == ["only"]


# ---------------------------------------------------------------------------
# parse_groups
# ---------------------------------------------------------------------------


class TestParseGroups:
    def test_three_groups(self):
        items = ["A_B_1", "C_D_2"]
        groups = parse_groups(items)
        assert groups == [("A", "B", "1"), ("C", "D", "2")]

    def test_single_group_no_underscore(self):
        items = ["sample1", "sample2"]
        groups = parse_groups(items)
        assert groups == [("sample1",), ("sample2",)]

    def test_inconsistent_group_count_raises(self):
        items = ["A_B", "C_D_E"]
        with pytest.raises(SystemExit):
            parse_groups(items)

    def test_all_same_structure(self):
        items = ["Wt_Mannitol_Cellular_1", "FraB_GA_Extracellular_3"]
        groups = parse_groups(items)
        assert len(groups) == 2
        for g in groups:
            assert len(g) == 4


# ---------------------------------------------------------------------------
# unique_ordered
# ---------------------------------------------------------------------------


class TestUniqueOrdered:
    def test_duplicates_removed_preserves_order(self):
        assert unique_ordered(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]

    def test_already_unique_unchanged(self):
        assert unique_ordered(["x", "y", "z"]) == ["x", "y", "z"]

    def test_all_same(self):
        assert unique_ordered(["a", "a", "a"]) == ["a"]

    def test_empty(self):
        assert unique_ordered([]) == []

    def test_single_element(self):
        assert unique_ordered(["q"]) == ["q"]


# ---------------------------------------------------------------------------
# build_row_groups
# ---------------------------------------------------------------------------


class TestBuildRowGroups:
    # Items: strain_carbon_fraction_rep
    ITEMS = [
        "Wt_Man_Cell_1",
        "Wt_Man_Ext_1",
        "FraA_Man_Cell_1",
        "FraA_Man_Ext_1",
        "Wt_Man_Cell_2",
        "FraA_Man_Ext_2",
    ]

    @pytest.fixture
    def groups(self):
        return parse_groups(self.ITEMS)

    def test_single_fix_index_partition(self, groups):
        # fix_index=2 (fraction): "Cell" vs "Ext"
        row_groups, key_order = build_row_groups(groups, fix_indices=[2])
        assert len(row_groups) == 2
        # First-occurrence order: Cell appears first
        assert key_order[0] == ("Cell",)
        assert key_order[1] == ("Ext",)

    def test_bucket_contents_correct(self, groups):
        row_groups, key_order = build_row_groups(groups, fix_indices=[2])
        cell_bucket = row_groups[0]
        ext_bucket = row_groups[1]
        # Indices 0, 2, 4 are Cell; 1, 3, 5 are Ext
        assert sorted(cell_bucket) == [0, 2, 4]
        assert sorted(ext_bucket) == [1, 3, 5]

    def test_key_order_is_first_occurrence_not_alphabetical(self, groups):
        # fix_index=0 (strain): Wt appears before FraA in input
        _, key_order = build_row_groups(groups, fix_indices=[0])
        assert key_order[0] == ("Wt",)
        assert key_order[1] == ("FraA",)

    def test_multi_fix_indices_composite_key(self, groups):
        # fix_indices=[0,2]: (strain, fraction) → (Wt,Cell), (Wt,Ext), (FraA,Cell), (FraA,Ext)
        row_groups, key_order = build_row_groups(groups, fix_indices=[0, 2])
        assert len(row_groups) == 4
        assert key_order[0] == ("Wt", "Cell")
        assert key_order[1] == ("Wt", "Ext")
        assert key_order[2] == ("FraA", "Cell")
        assert key_order[3] == ("FraA", "Ext")

    def test_all_items_accounted_for(self, groups):
        row_groups, _ = build_row_groups(groups, fix_indices=[2])
        all_indices = [i for rg in row_groups for i in rg]
        assert sorted(all_indices) == list(range(len(self.ITEMS)))

    def test_single_group_value(self, groups):
        # fix_index=1 (carbon source): only "Man" → one group containing all
        row_groups, key_order = build_row_groups(groups, fix_indices=[1])
        assert len(row_groups) == 1
        assert key_order == [("Man",)]
        assert sorted(row_groups[0]) == list(range(len(self.ITEMS)))
