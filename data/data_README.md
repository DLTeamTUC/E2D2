# Data preparation

The E2D2 notebook expects five preprocessed CSV files placed under three subfolders at the repository root:

```
.
├── e2d2.ipynb
└── data preparation outputs go in:
    ├── CICDDoS2019/
    │   ├── DrDoS_NTP_reduced_balanced.csv
    │   ├── Portmap_reduced_balanced.csv
    │   └── Syn_reduced_balanced.csv
    ├── CICIoT2023/
    │   └── merged_CICIoT2023_balanced.csv
    └── 5GNIDD/
        └── merged_5GNIDD.csv
```

This document explains where to obtain the raw datasets and how to produce these specific files. The intermediate files are not redistributed in this repository because of their size and because some are subject to the original datasets' redistribution terms.

## 1. Download the source datasets

| Dataset       | Source                                                                              | Description                                                |
| ------------- | ----------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| CIC-DDoS2019  | https://www.unb.ca/cic/datasets/ddos-2019.html                                      | Realistic DDoS attack traces; multiple per-attack CSVs     |
| CICIoT2023    | https://www.unb.ca/cic/datasets/iotdataset-2023.html                                | Large-scale IoT attack benchmark; many per-attack CSVs     |
| 5GNIDD        | https://etsin.fairdata.fi/dataset/9d13ef28-2ca7-44b0-9950-225359afac65               | 5G network intrusion detection dataset; train/test/val splits |

Each source provides flow-level CSVs already extracted with [CICFlowMeter](https://www.unb.ca/cic/research/applications.html), so no PCAP-to-flow conversion is needed on the user's side.

After downloading, extract each archive into its own working directory. The instructions below assume the raw files are placed under:

```
raw/
├── CICDDoS2019/        # contains DrDoS_NTP.csv, Portmap.csv, Syn.csv, ...
├── CICIoT2023/         # contains all per-attack flow CSVs
└── 5GNIDD/             # contains 5GNIDD-train.csv, 5GNIDD-val.csv, 5GNIDD-test.csv
```

## 2. Preprocessing per dataset

The three datasets need different preparation. The notebook itself handles column dropping, NaN/Inf removal, zero-variance removal, MinMax scaling, and feature selection. The steps below produce the CSV files the notebook reads.

### 2.1 CIC-DDoS2019 — three balanced 50/50 subsets

For each of the three attack types used in the paper (NTP, Portmap, Syn), the original CSV from the CIC-DDoS2019 release is downsampled to **200,000 rows total**, split **50/50 between benign and attack samples**.

Concretely, for each attack file:

1. Read the original CSV.
2. Identify benign rows (`" Label" == "BENIGN"`) and attack rows (everything else).
3. Randomly sample 100,000 benign rows and 100,000 attack rows without replacement, with a fixed seed for reproducibility.
4. Concatenate, shuffle, and write the result.

A reference Python implementation:

```python
import pandas as pd
from pathlib import Path

RAW = Path("raw/CICDDoS2019")
OUT = Path("CICDDoS2019")
OUT.mkdir(exist_ok=True)

PAIRS = [
    ("DrDoS_NTP.csv", "DrDoS_NTP_reduced_balanced.csv"),
    ("Portmap.csv",   "Portmap_reduced_balanced.csv"),
    ("Syn.csv",       "Syn_reduced_balanced.csv"),
]
N_PER_CLASS = 100_000
SEED = 42

for src_name, dst_name in PAIRS:
    df = pd.read_csv(RAW / src_name, low_memory=False)
    benign = df[df[" Label"] == "BENIGN"]
    attack = df[df[" Label"] != "BENIGN"]

    benign_sample = benign.sample(
        n=min(N_PER_CLASS, len(benign)), random_state=SEED
    )
    attack_sample = attack.sample(
        n=min(N_PER_CLASS, len(attack)), random_state=SEED
    )

    out = (
        pd.concat([benign_sample, attack_sample], ignore_index=True)
          .sample(frac=1.0, random_state=SEED)
          .reset_index(drop=True)
    )
    out.to_csv(OUT / dst_name, index=False)
    print(f"  wrote {dst_name}: {len(out):,} rows "
          f"({(out[' Label'] == 'BENIGN').sum():,} benign / "
          f"{(out[' Label'] != 'BENIGN').sum():,} attack)")
```

A small attack class may yield fewer than 100,000 attack rows; in that case the `min(...)` clause above caps the per-class size to whatever is available, and the final size will be smaller than 200k. This does not affect the notebook because it derives its own benign-reference / stream split from whatever rows are present.

### 2.2 CICIoT2023 — concatenated and balanced

The CICIoT2023 release distributes attack traffic across many per-class CSV files (UDP flood, TCP flood, ICMP flood, brute-force, recon, etc.). For E2D2, all files are concatenated into a single CSV.

```python
import pandas as pd
from pathlib import Path

RAW = Path("raw/CICIoT2023")
OUT = Path("CICIoT2023")
OUT.mkdir(exist_ok=True)

frames = []
for csv_path in sorted(RAW.glob("*.csv")):
    df = pd.read_csv(csv_path, low_memory=False)
    frames.append(df)
    print(f"  {csv_path.name}: {len(df):,} rows")

merged = pd.concat(frames, ignore_index=True)
merged = merged.sample(frac=1.0, random_state=42).reset_index(drop=True)
merged.to_csv(OUT / "merged_CICIoT2023_balanced.csv", index=False)

print(f"  Total: {len(merged):,} rows")
print(f"  Benign: {(merged['Label'] == 'BENIGN').sum():,}")
print(f"  Attack: {(merged['Label'] != 'BENIGN').sum():,}")
```

Note that `Label` here has no leading space (unlike CIC-DDoS2019). The "balanced" suffix in the output filename is historical; the merged file is class-balanced enough for the notebook because of the variety of source files, and the notebook only uses the benign portion for VAE training, with the rest reserved for the streaming evaluation.

### 2.3 5GNIDD — concatenated splits

5GNIDD ships as three separate splits (train, validation, test). The notebook treats the dataset as a single stream and produces its own benign-reference / stream split internally, so the three splits are concatenated:

```python
import pandas as pd
from pathlib import Path

RAW = Path("raw/5GNIDD")
OUT = Path("5GNIDD")
OUT.mkdir(exist_ok=True)

# Adjust filenames to match what the 5GNIDD release uses
SPLITS = ["5GNIDD-train.csv", "5GNIDD-val.csv", "5GNIDD-test.csv"]

frames = [pd.read_csv(RAW / s, low_memory=False) for s in SPLITS]
merged = pd.concat(frames, ignore_index=True)
merged = merged.sample(frac=1.0, random_state=42).reset_index(drop=True)
merged.to_csv(OUT / "merged_5GNIDD.csv", index=False)

print(f"  Total: {len(merged):,} rows")
print(f"  Benign: {(merged['Attack Type'] == 'Benign').sum():,}")
print(f"  Attack: {(merged['Attack Type'] != 'Benign').sum():,}")
```

Note that 5GNIDD uses `Attack Type` as the label column and `Benign` (capitalized differently from the other datasets) as the benign value.

## 3. Expected file properties

After running the steps above, the notebook expects:

| File                                        | Approx. rows | Label column   | Benign value  |
| ------------------------------------------- | ------------ | -------------- | ------------- |
| `CICDDoS2019/DrDoS_NTP_reduced_balanced.csv`| ≤ 200,000    | `' Label'`     | `'BENIGN'`    |
| `CICDDoS2019/Portmap_reduced_balanced.csv`  | ≤ 200,000    | `' Label'`     | `'BENIGN'`    |
| `CICDDoS2019/Syn_reduced_balanced.csv`      | ≤ 200,000    | `' Label'`     | `'BENIGN'`    |
| `CICIoT2023/merged_CICIoT2023_balanced.csv` | varies       | `'Label'`      | `'BENIGN'`    |
| `5GNIDD/merged_5GNIDD.csv`                  | varies       | `'Attack Type'`| `'Benign'`    |

Cell 0.1 of the notebook verifies that each file is present at startup and prints a `[OK]` or `[MISSING]` marker; the experiment matrix in Cell 5.1 then loads them.

The exact row counts you obtain may differ slightly from those used in the paper if the random seeds differ or if the source dataset has been updated since the paper run. Small numerical deviations from the paper's reported metrics are expected and documented in the main `README.md`; the qualitative findings are robust to such variation.

## 4. Storage and citation

The five preprocessed CSVs together occupy roughly several gigabytes on disk. They are not redistributed in this repository — please cite the original dataset releases:

* CIC-DDoS2019 — Sharafaldin et al., *ICCST 2019*.
* CICIoT2023 — Neto et al., *Sensors 23(13):5941, 2023*.
* 5GNIDD — Samarakoon et al., *arXiv:2212.01298, 2022*.

The full bibliographic entries are listed as references [21]–[23] in the E2D2 paper.
