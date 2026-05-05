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
| `--input FILE` | `to_randomise.txt` | Input file path |
| `--seed INT` | `42` | Random seed for full reproducibility |
| `--fix-sort GROUPS` | *(none)* | Comma-separated group indices used to pre-partition the list into sub-problems before randomizing (e.g. `2` or `1,2`) |
| `--weight WEIGHTS` | `1,1,...` | Comma-separated per-group weights; higher weight = that group contributes more to the score |
| `--max-iter INT` | `50000 × n` | Maximum SA iterations per sub-problem |
| `--no-warn` | | Suppress the large-list warning |

### Typical workflow

```bash
# Simple randomization (≤ 20 samples)
python randomize_samples_for_lcmsms.py --seed 7 > run_order.txt

# Large list: fix group 2 (fraction) to create sub-problems, then randomize within each
python randomize_samples_for_lcmsms.py --fix-sort 2 --seed 7 > run_order.txt

# Redirect diagnostics to a log file
python randomize_samples_for_lcmsms.py --fix-sort 2 --seed 7 > run_order.txt 2> run_order.log

# Weight strain transitions twice as heavily as others (group 0)
python randomize_samples_for_lcmsms.py --fix-sort 2 --seed 7 --weight 2,1,1,1 > run_order.txt
```

---

## Algorithm

### Simulated Annealing with O(1) incremental scoring

At each iteration the algorithm proposes a swap of two positions and evaluates
the change in a combined score:

```
combined = diversity_score  −  λ · balance_penalty
```

**`diversity_score`** — for each consecutive pair (A, B): sum of `weights[g]`
for all groups g where A\[g\] ≠ B\[g\].  Rewards runs that differ in as many
condition dimensions as possible.

**`balance_penalty`** — for each group g: Σ_{a,b} T_g\[a,b\]², where
T_g\[a,b\] counts how many times value a was immediately followed by value b.
Penalises over-represented directed transitions, driving all pairs toward equal
frequency.

**λ** — set automatically to `1 / (total transitions in sequence so far)`,
keeping both terms on the same numerical scale as the sequence grows.

Both the diversity delta and the balance-penalty delta are computed in **O(1)**
per swap proposal (only the ≤ 4 affected transitions need to be evaluated),
making the algorithm fast even for large lists.

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
# Seed: 7
# 64 items, 4 group(s):
#   Group 0: ['Wt', 'FraA', 'FraB', 'SdeA']
#   Group 1: ['Mannitol', 'Galacturonic-acid']
#   Group 2: ['Cellular', 'Extracellular']
#   Group 3: ['1', '2', '3', '4']
# --fix-sort '2': 2 row_group(s):
#   [0] key=('Cellular',)  (32 items)
#   [1] key=('Extracellular',)  (32 items)
# Row_group [0] key=('Cellular',)  score=65.87  size=32  max_iter=1600000
# Row_group [0] key=('Cellular',)  (31 transition(s))
#   (Group 2 omitted — fixed by --fix-sort)
#   Group 0:
#     Wt              -> FraA            :  3/31 =   9.7%
#     ...
# Overall  (63 transition(s))
#   Group 0:
#     Wt              -> FraA            :  6/63 =   9.5%
#     ...
```

---

## Roadmap

- [x] Check for weighted diversity effectiveness
- [ ] Extract core functions into submodules (`scoring`, `io`, `annealing`)
- [ ] Plotting: transition heatmaps, score convergence curves
- [ ] Support for Excel / CSV input

---

## License

GPL-3.0