# Data

## Contents

`processed/` contains the three curated Rubisco kinetics datasets used for training the **general** models (trained across all 9 phyla):

| File | Target property | Species (rows) |
|---|---|---|
| `Specificity.csv` | Sc/o (specificity factor) | 208 |
| `kcatC.csv` | kcat CO2 (catalytic turnover rate) | 241 |
| `KC.csv` | Km CO2 (Michaelis constant) | 221 |


Each file has the following columns: `ID`, `Name`, `Species`, `Phylum`, `Sequence`, and the target kinetic value.

## `Name` column encoding

The `Name` column encodes which of the three kinetic properties were available for that row in the original literature-compiled dataset, via a prefix + sequential number:

| Prefix | Meaning |
|---|---|
| `S` | Specificity only |
| `SKC` | Specificity + Km |
| `SKcat` | Specificity + kcat |
| `SKcatKC` | All three properties available |
| `Kcat`, `KcatKC`, `KC` | (analogous, without Specificity) |

`Name` is **unique within each file** and is the join key used to match rows against ESM embedding CSVs during training.