"""Tests for CLI argument-parsing helpers and end-to-end integration."""

import subprocess
import sys
from pathlib import Path

import pytest

from randomize_samples_for_lcmsms import _parse_fix_sort, _parse_weights

# Path to the script and the sample input file
REPO_ROOT = Path(__file__).parent.parent
SCRIPT = REPO_ROOT / "randomize_samples_for_lcmsms.py"
SAMPLE_INPUT = REPO_ROOT / "to_randomise.txt"

# Keep integration tests fast: cap iterations to a small value.
FAST_ITER = "1000"


# ---------------------------------------------------------------------------
# _parse_weights
# ---------------------------------------------------------------------------


class TestParseWeights:
    def test_valid_three_groups(self):
        assert _parse_weights("1,2,3", 3) == [1.0, 2.0, 3.0]

    def test_single_group(self):
        assert _parse_weights("5", 1) == [5.0]

    def test_float_values(self):
        result = _parse_weights("0.5,1.5", 2)
        assert result == pytest.approx([0.5, 1.5])

    def test_wrong_count_raises(self):
        with pytest.raises(SystemExit):
            _parse_weights("1,2", 3)

    def test_non_numeric_raises(self):
        with pytest.raises(SystemExit):
            _parse_weights("1,abc,3", 3)

    def test_trailing_comma_raises(self):
        # "1,2," splits to ["1","2",""] — float("") fails
        with pytest.raises(SystemExit):
            _parse_weights("1,2,", 3)


# ---------------------------------------------------------------------------
# _parse_fix_sort
# ---------------------------------------------------------------------------


class TestParseFixSort:
    def test_valid_single(self):
        assert _parse_fix_sort("2", 4) == [2]

    def test_valid_multiple(self):
        assert _parse_fix_sort("0,2", 4) == [0, 2]

    def test_leading_trailing_whitespace(self):
        assert _parse_fix_sort(" 1 ", 3) == [1]

    def test_out_of_range_raises(self):
        with pytest.raises(SystemExit):
            _parse_fix_sort("4", 4)  # valid range 0–3

    def test_negative_index_raises(self):
        with pytest.raises(SystemExit):
            _parse_fix_sort("-1", 4)

    def test_float_string_raises(self):
        with pytest.raises(SystemExit):
            _parse_fix_sort("1.5", 4)

    def test_non_numeric_raises(self):
        with pytest.raises(SystemExit):
            _parse_fix_sort("a", 4)

    def test_exactly_last_valid_index(self):
        assert _parse_fix_sort("3", 4) == [3]


# ---------------------------------------------------------------------------
# Integration tests (subprocess)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_input_lines():
    """Return non-empty lines from to_randomise.txt, or skip if absent."""
    if not SAMPLE_INPUT.exists():
        pytest.skip("to_randomise.txt not found — skipping integration tests")
    lines = [ln.split()[0] for ln in SAMPLE_INPUT.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return lines


def _run_script(*extra_args):
    """Run the script and return stdout lines (stripped)."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--no-warn", "--max-iter", FAST_ITER, *extra_args, str(SAMPLE_INPUT)],
        capture_output=True,
        text=True,
        check=True,
    )
    return [ln for ln in result.stdout.splitlines() if ln.strip()]


class TestIntegration:
    def test_output_line_count_matches_input(self, sample_input_lines):
        out = _run_script("--seed", "42")
        assert len(out) == len(sample_input_lines)

    def test_output_is_permutation_of_input(self, sample_input_lines):
        out = _run_script("--seed", "42")
        assert sorted(out) == sorted(sample_input_lines)

    def test_deterministic_same_seed(self, sample_input_lines):
        out1 = _run_script("--seed", "7")
        out2 = _run_script("--seed", "7")
        assert out1 == out2

    def test_different_seeds_differ(self, sample_input_lines):
        out1 = _run_script("--seed", "1")
        out2 = _run_script("--seed", "999")
        # Both valid permutations; almost certainly not the same order
        assert sorted(out1) == sorted(sample_input_lines)
        assert sorted(out2) == sorted(sample_input_lines)
        # Warn rather than fail — theoretically identical by coincidence
        if out1 == out2:
            pytest.skip("Seeds 1 and 999 produced the same order (unlikely but possible)")

    def test_fix_sort_output_is_permutation(self, sample_input_lines):
        out = _run_script("--seed", "42", "--fix-sort", "2")
        assert sorted(out) == sorted(sample_input_lines)

    def test_fix_sort_deterministic(self, sample_input_lines):
        out1 = _run_script("--seed", "5", "--fix-sort", "2")
        out2 = _run_script("--seed", "5", "--fix-sort", "2")
        assert out1 == out2

    def test_weight_flag_accepted(self, sample_input_lines):
        # Provide 4 weights matching the 4 groups in to_randomise.txt
        out = _run_script("--seed", "42", "--weight", "2,1,1,1")
        assert sorted(out) == sorted(sample_input_lines)

    def test_missing_file_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "nonexistent_file.txt"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "Error" in result.stderr
