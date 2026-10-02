# Data plan: one clean training dataset

*Private working notes (kept out of git). Written 2 Oct 2026; final deadline 22 Oct, team-merge deadline 15 Oct.*

**Goal:** gather all the data, clean it and pack it into one folder (`knee-data/`) that every modelling path (A–F in `rsna-knee-approach-paths.md`) can train from. Only then build models.

**Status:** Phase 0 in progress.

---

## Open decisions (needed before Phase 3)

| Decision | Options | Suggestion |
|---|---|---|
| Image size on disk | All slices at 256 px ≈ **54 GB** (+ ~6 GB MRNet), or 224 px ≈ **41 GB** | 256 px if the SSD has room |
| MRNet (Stanford) | Include now as its own marked part, or wait for the host's ruling on external data | Include now, marked `source = mrnet`, and keep one final submission without it |

---

## What runs where

- **Your PC (VS Code + git):** all code (you commit it), label building, folds, checks, training on the RTX 5070, and the finished dataset in one folder on an SSD.
- **Kaggle:** the 570 GB of DICOMs never leave Kaggle. Kaggle runs repo code for jobs that must read them (one-time CPU runs), the submission notebooks, and GPU training only if the PC is not enough.
- **The link:** the Kaggle command-line tool in the VS Code terminal.
  - `python scripts/push_code.py "message"` uploads `kneemri/` as your private dataset `kneemri-code`.
  - `python scripts/kaggle_run.py notebooks/<name>` runs a notebook folder on Kaggle and downloads its output to `runs/<name>/`.

---

## Phase 0 — Connect the two environments (½ day)

1. **Files:** unzip the Phase 0 files into the repo folder (next to `docs/`).
   ```
   knee_mri/
     .gitignore          data, weights, secrets and this file stay out of git
     requirements.txt    Python libraries
     pyproject.toml      lets `pytest` find the kneemri package
     configs/project.json   your Kaggle username + where knee-data/ lives
     kneemri/            the shared code (runs on your PC and on Kaggle)
     scripts/            push_code.py, kaggle_run.py
     notebooks/00_hello/ first Kaggle notebook
     tests/              unit tests
     docs/               your notes
   ```
2. **Python environment** (VS Code terminal, in the repo folder):
   ```
   python -m venv .venv
   .venv\Scripts\Activate.ps1        (Mac/Linux: source .venv/bin/activate)
   pip install -r requirements.txt
   pytest                            -> 5 passed
   ```
   Then in VS Code: Ctrl+Shift+P → "Python: Select Interpreter" → `.venv`.
3. **Kaggle login:** kaggle.com → Settings → API → **Generate New Token**. Save the token in a file called `access_token` inside a `.kaggle` folder in your home folder (Windows: `C:\Users\<you>\.kaggle\access_token`), or run `kaggle auth login`.
   Test: `kaggle competitions files rsna-knee-abnormality-detection` lists the competition files.
4. **Settings:** put your Kaggle username (and where `knee-data/` should live) in `configs/project.json`.
5. **Commit** (you): `git add .` → `git status` (check: no data, no token) → `git commit -m "Phase 0: project skeleton and Kaggle link"` → `git push`.
6. **Upload the code:** `python scripts/push_code.py "phase 0"` → private dataset `<you>/kneemri-code`.
7. **First Kaggle run:** `python scripts/kaggle_run.py notebooks/00_hello` → output in `runs/00_hello/`.
8. **PyTorch for the RTX 5070** (needed for training later): `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`, then
   `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.

**Done when:** tests pass, `runs/00_hello/hello.json` shows the competition files and the code version you pushed, and the GPU is visible.
The hello run also shows which DICOM compression formats occur and whether Kaggle can decode them with internet off (matters for the submission notebook).

---

## Phase 1 — Inventory everything (Kaggle, one CPU run, ~1–2 h)

1. Read the header (not the pixels) of all ~820k DICOM files into one table, one row per file: study, series, position, orientation, pixel spacing, slice thickness, echo, scanner make/model/software, compression, and every other tag present.
2. List which of the 86 allowed tags actually exist, so nothing useful is missed.
3. MRNet: list files, array shapes, label files and counts.
4. Download the table and the small CSVs (`train.csv` with reports, `train_series.csv`, test files, public label sets) to the PC.

## Phase 2 — Clean the metadata (PC, fast, iterate freely)

1. Inside each series: drop stray localiser slices, keep one slice per position (multi-echo), order by position, flag gaps.
2. Each series: plane from orientation vs `train_series.csv`; fluid / fat-sat flags vs echo and repetition times (if those tags exist); mark scouts (< 5 slices) and missing pixel spacing.
3. Each study: missing key series; knee side (left/right) from the left-right position in scanner coordinates, cross-checked with "right/left" in the report (5 of 12 labels depend on the side); bilateral studies (series of both knees — keep the relevant knee).
4. Pick the series for each slot (sagittal fluid, sagittal other, coronal fluid, coronal other, axial) from the real metadata, and record why.
5. Scanner fingerprint per study (to group folds).
6. Write every rule, and how many rows it touched, to `docs/cleaning_rules.md`.

## Phase 3 — Build the image store (Kaggle, a few one-time CPU runs)

1. Every kept slice: decode, rotate into the standard orientation, resample to a fixed scale of 0.625 mm per pixel on a 256×256 canvas (160 mm). Models crop 224 px = 140 mm, the setting in the notes, with margin for augmentation.
2. Keep **all** slices of all kept series, so each path can choose its own slices without going back to the DICOMs.
3. Store as one big uint8 array of slices + an index (study, series, slot, position, knee side). ~54 GB in 4 parts (a Kaggle run saves at most ~20 GB).
4. Fingerprint each series' middle slice to find exact duplicate scans across studies.
5. MRNet (one run): same format; sagittal → sagittal-fluid slot, coronal T1 → coronal-other, axial → axial; marked `source = mrnet`. It has no pixel spacing, so it is resized, not cropped by mm.
6. Download everything to `knee-data/`.

## Phase 4 — Labels and splits (PC)

1. `gold.csv` (58), public label sets as downloaded, `labels_v2.csv` + its empty-cell mask; later your own LLM labels (Path B1) and pseudo-labels (B4) as **new files, never overwrites**.
2. Two extra targets shared with MRNet: "any meniscus tear" and "abnormal".
3. 5 folds: grouped by scanner, balanced by label, gold studies ~11–12 per fold, duplicates in the same fold, MRNet in training only.
4. Report text + language table (for making labels only — the test set has no reports, so they are never a model input).

## Phase 5 — Pack and freeze "knee-data v1" (PC)

1. One folder, a `manifest.json` (code version, settings, counts, checksums) and `docs/DATA_CARD.md` (every file and column explained).
2. One data loader in `kneemri/` used by every path. Paths differ only in its settings: slices per slot, 2.5D vs single-slice ×3 (RadImageNet), with/without MRNet, mirror left knees or not.

```
knee-data/
  manifest.json
  meta/     slices.parquet, series.parquet, studies.parquet, slots.csv
  images/   rsna_part0-3.npy, mrnet.npy, index.parquet
  labels/   gold.csv, sources/, labels_v2.csv, mrnet.csv
  splits/   folds.csv
  reports/  reports.parquet
```

---

## Tests: check the dataset and find what we missed

| # | Check | Catches | Passes when |
|---|---|---|---|
| T1 | Unit tests on fake DICOMs (local, before every Kaggle build) | orientation, ordering, localiser, echo, scale, compression bugs | all pass |
| T2 | Completeness | lost studies / series / files | every file is kept or dropped with a reason; counts add up exactly |
| T3 | Consistency | broken links between files | index matches arrays; images, labels and folds use the same IDs; gold equals `train.csv`; each study in exactly one fold |
| T4 | Outliers | blank, dark, saturated or odd-sized images | every outlier reviewed: kept, fixed or dropped |
| T5 | Visual review | wrong orientation, crop or knee side; MRNet looking different | 50 random studies + every flagged one look right |
| T6 | Training/test parity | the model seeing different images at test time | 20 training studies + the 3 test examples, processed by the inference code, match the store |
| T7 | Image–label alignment (no model) | images attached to the wrong labels | bright-fluid amount in fluid slots ranks Effusion positives clearly higher |
| T8 | Smoke training | loader bugs, slow loading | 1 epoch runs, loss falls, a tiny model memorises 32 studies, GPU isn't waiting for data |
| T9 | Kaggle dry run | submission failures | internet-off notebook processes the 3 test studies; time × 1,322 well under 9 h |
| T10 | Review vs notes and rules | anything forgotten | all tags reviewed, bilateral studies, languages, no report text as input, MRNet licence, data private |

Result: a QC report (counts, what was dropped and why, test results) next to the dataset.

---

## Timeline

About 4 days for Phases 0–5 and the tests → dataset ready around **6 Oct** (about 2 days later than the first submission planned in the notes). Path A training can start as soon as the Phase 3 download finishes, while T4–T5 continue.

## Rules and licences to keep in mind

- The dataset is derived from competition data: keep it private, never share it outside the team.
- MRNet: keep the Kaggle copy private (its research-use agreement forbids sharing the data). External data has no host ruling yet (thread 743416), so keep a final submission that does not use it.
- Report text is used only to make labels, never as a model input.
