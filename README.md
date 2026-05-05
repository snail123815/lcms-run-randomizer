# lcms-run-randomizer

Randomize a sample list for LC-MS/MS runs, maximizing the diversity of consecutive condition transitions so that each unique condition value is followed by every other value with **approximately equal frequency**.

---

## Background

In LC-MS/MS proteomics experiments, systematic run-order bias can confound quantitative results, especially for low-abundance proteins. Simply shuffling samples avoids block effects but does not guarantee that every condition-to-condition transition is represented equally.

---

## Goal

Uses **Simulated Annealing (SA)** to find a run order where:

Across the full sequence — each directed pair (condition A → condition B) occurs as often as every other pair, for every condition dimension independently.

When grouping samples by one or more fixed condition(s) (e.g. growth phase), the same balance is achieved **within each group** (e.g. within each fraction), and the balance penalty is computed globally across the full sequence, so the algorithm steers each group away from over-represented transitions in previous groups, producing a globally balanced run order.

---

## Requirements

- Python ≥ 3.10 (stdlib only: `argparse`, `math`, `random`, `sys`, `pathlib`)

---

## Input format

A plain-text file with one sample name per line. Only the first whitespace-separated token on each line is used. Each token is split on `_` to obtain per-group condition values:

```
Wt_Mannitol_Cellular_1
Wt_Mannitol_Cellular_2
FraB_Galacturonic-acid_Extracellular_1
FraB_Galacturonic-acid_Extracellular_2
...
```

Here group 0 = strain, group 1 = carbon source, group 2 = fraction, group 3 = replicate.

All tokens must have the same number of `_`-delimited parts.

---

## Usage

```
python randomize_samples_for_lcmsms.py [options] FILE
```

Output — the randomized sample list — is written to **stdout**.  
Diagnostics (group summaries, per-group transition statistics, scores) go to **stderr**.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `FILE` (positional) | `to_randomise.txt` | Input file path |
| `--seed INT` | `42` | Random seed — change to get a different randomization |
| `--fix-sort GROUPS` | *(none)* | Comma-separated group indices used to pre-partition the list into sub-problems before randomizing (e.g. `2` or `1,2`) |
| `--weight WEIGHTS` | `1,1,...` | Comma-separated per-group weights; higher weight = that group contributes more to the score |
| `--max-iter INT` | `50000 × n` | Maximum SA iterations per sub-problem |
| `--priority {carryover,time}` | `carryover` | Optimization priority: `carryover` maximizes Diversity (consecutive-condition transitions); `time` maximizes Spread (each value evenly distributed across run positions to reduce instrument-drift confounding) |
| `--no-warn` | | Suppress the large-list warning |
| `--plot` | | Print SA score-convergence plots to stderr (requires `pip install plotext`) |

### Typical workflow

**bash / zsh**

```bash
# Simple randomization (≤ 20 samples)
python randomize_samples_for_lcmsms.py --seed 7 > run_order.txt

# Large list: fix group 2 (fraction) to create sub-problems, then randomize within each
python randomize_samples_for_lcmsms.py --fix-sort 2 --seed 7 > run_order.txt

# Redirect diagnostics to a log file
python randomize_samples_for_lcmsms.py --fix-sort 2 --seed 7 > run_order.txt 2> run_order.log

# Weight strain transitions twice as heavily as others (group 0)
python randomize_samples_for_lcmsms.py --fix-sort 2 --seed 7 --weight 2,1,1,1 > run_order.txt

# Prioritize even temporal spread (instrument-drift mitigation) over carryover reduction
python randomize_samples_for_lcmsms.py --fix-sort 2 --seed 7 --priority time > run_order.txt
```

**PowerShell**

```powershell
# Simple randomization (≤ 20 samples)
python randomize_samples_for_lcmsms.py --seed 7 | Set-Content run_order.txt

# Large list: fix group 2 (fraction) to create sub-problems, then randomize within each
python randomize_samples_for_lcmsms.py --fix-sort 2 --seed 7 | Set-Content run_order.txt

# Redirect diagnostics to a log file
# PowerShell merges stderr into the success stream with *>&1, so split them first
python randomize_samples_for_lcmsms.py --fix-sort 2 --seed 7 2>run_order.log | Set-Content run_order.txt

# Weight strain transitions twice as heavily as others (group 0)
python randomize_samples_for_lcmsms.py --fix-sort 2 --seed 7 --weight 2,1,1,1 | Set-Content run_order.txt

# Prioritize even temporal spread (instrument-drift mitigation) over carryover reduction
python randomize_samples_for_lcmsms.py --fix-sort 2 --seed 7 --priority time | Set-Content run_order.txt
```

---

## Algorithm

### Simulated Annealing with O(1) incremental scoring

At each iteration the algorithm proposes a swap of two positions and evaluates
the change in a combined score:

```
combined = diversity_score  −  λ_bal  · balance_penalty
                            +  λ_time · spread_bonus
```

**`diversity_score`** — for each consecutive pair (A, B): sum of `weights[g]`
for all groups g where A\[g\] ≠ B\[g\].  Rewards runs that differ in as many
condition dimensions as possible.

**`balance_penalty`** — for each group g: Σ_{a,b} T_g\[a,b\]², where
T_g\[a,b\] counts how many times value a was immediately followed by value b.
Penalises over-represented directed transitions, driving all pairs toward equal
frequency.

**`spread_bonus`** — for each group g and unique value v: normalized position
variance `4 · Var(positions of v) / n²`.  Rewards samples with the same
condition value appearing spread evenly across the run, mitigating
instrument-drift confounding.

**λ_bal / λ_time** — controlled by `--priority`:
- `carryover` (default): `λ_bal = 1 / n_trans`; `λ_time = λ_bal × 10⁻⁴` — spread breaks ties only.
- `time`: `λ_time = 1 / n_trans`; `λ_bal = λ_time × 10⁻⁴` — spread is the primary objective.

All three deltas (diversity, balance, spread) are computed in **O(1)**
per swap proposal (only the ≤ 4 affected transitions and 2 affected positions
need to be evaluated), making the algorithm fast even for large lists.

Temperature decays exponentially from `T_start = max(1, Σ weights)` to
`T_min = 1e-4` over 80 % of `max_iter`, with stagnation-based early stopping.

### `--fix-sort` and global transition history

For large lists (> 20 items), `--fix-sort` partitions the list by the unique
values of one or more groups and runs SA independently on each partition
(row_group).  Each row_group is optimized **sequentially**: the
`TransitionTracker` accumulated from all previously completed row_groups is
passed into the next SA call as a frozen background, so the balance penalty
steers each new row_group away from transition pairs already over-represented
in earlier row_groups.  This produces a globally even distribution across the
full run sequence.

---

## Output example (stderr diagnostics)

```
# ── Input ─────────────────────────────────────────────────────────────
# Seed: 7
# 64 items, 4 group(s):
#   Group 0: ['Wt', 'FraA', 'FraB', 'SdeA']
#   Group 1: ['Mannitol', 'Galacturonic-acid']
#   Group 2: ['Cellular', 'Extracellular']
#   Group 3: ['1', '2', '3', '4']
# --fix-sort '2': 2 row_group(s):
#   [0] key=('Cellular',)  (32 items)
#   [1] key=('Extracellular',)  (32 items)
#
# ── Optimisation ──────────────────────────────────────────────────────────
# Row_group [0] key=('Cellular',)  size=32  max_iter=1600000
#   Transitions  (31 transition(s))
#     (Group 2 omitted — fixed by --fix-sort)
#     Group 0:
#       Wt              -> FraA            :  3/31 =   9.7%
#       ...
#   Quality  (31 transition(s)):
#     Diversity :  31.0 / 31.0  (100.0%)
#     Balance   :  Group 0: 98.2%   Group 1: 97.5%   Group 3: 96.8%
#     Spread    :  Group 0:  Wt=85%  FraA=87%  FraB=82%  SdeA=84%
#
# Row_group [1] key=('Extracellular',)  size=32  max_iter=1600000
#   ...
#
# Overall  (63 transition(s))
#   Group 0:
#     Wt              -> FraA            :  6/63 =   9.5%
#     ...
# Overall quality  (63 transition(s)):
#   Diversity :  61.0 / 62.0  (98.4%)
#   Balance   :  Group 0: 97.8%   Group 1: 96.9%   Group 3: 95.4%
#   Spread    :  Group 0:  Wt=88%  FraA=90%  FraB=85%  SdeA=86%
#
# ── Output: randomized sample list follows ────────────────────────────────
```

The three quality metrics are:
- **Diversity** — actual vs. theoretical-maximum diversity score (100 % = every consecutive pair differs in every non-fixed group).
- **Balance** — Cauchy–Schwarz ratio of ideal to actual transition-count sum-of-squares (100 % = all directed A→B pairs equally frequent).
- **Spread** — per-value mean-absolute-deviation quality (100 % = occurrences land exactly at ideal evenly-spaced positions).

---

## Score-convergence plots (`--plot`)

Add `--plot` to print SA score-convergence charts to stderr.  Requires:

```
pip install plotext
```

**≤ 5 restarts** — a single chart showing the mean score per bin across all iterations.

**> 5 restarts** — two side-by-side charts:

| Left — *restart 1* | Right — *all-restart max* |
|---|---|
| Score trace of the first SA restart, trimmed at the point the global peak is first reached (the long flat stagnation tail is hidden). | Running maximum score per bin across every restart, showing the best solution found as the full iteration budget is spent. |

Example (19 restarts, 100 000 iterations):

```
# SA score: row_group [0] key=('Mannitol', 'Pellets')
#             restart 1                 all-restart max
#      ┌────────────────────────┐       ┌────────────────────────┐
# 27.20┤        ············   │  27.20┤····················    │
#      │       ·                │       │                        │
# 25.94┤     ··                 │  27.17│                        │
#      │     ·                  │       │                        │
# 24.68┤                        │  27.13│                        │
#      │   ··                   │       │                        │
# 23.42┤ ··                     │  27.10│            ·   ·       │
#      │·                       │       │                        │
# 22.16┤·                       │  27.07│            ·   ·       │
#      └┬────┬────┬────┬────────┘       └┬────┬──────────┬──────┘
#       66  1670 3275 4880             1000 25500       99000
# score       iteration             best score   iteration (all)
```

The left chart ends early because the score reached its peak at iteration ~220 and the remainder was flat stagnation.  The right chart shows that the optimum (27.20) was found in the very first restart and no subsequent restart improved on it.

---

## Roadmap

- [x] Check for weighted diversity effectiveness
- [x] Add `--priority` flag with `SpreadTracker` for temporal-spread optimization
- [x] Interpretable quality report (Diversity / Balance / Spread metrics) in stderr
- [x] Plotting: SA score-convergence curves via `--plot` (requires `plotext`)
- [ ] ~~Extract core functions into submodules (`scoring`, `io`, `annealing`)~~
- [ ] Support for Excel / CSV input

---

## License

GPL-3.0