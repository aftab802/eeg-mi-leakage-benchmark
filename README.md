# A Systematic Benchmark of Motor Imagery Classifiers Under Leakage-Safe Evaluation

[![tests](https://github.com/aftab802/eeg-mi-leakage-benchmark/actions/workflows/tests.yml/badge.svg)](https://github.com/aftab802/eeg-mi-leakage-benchmark/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Leakage-safe (L0) benchmark of CSP+CNN, EEGNet, 1D CNN and a four-class CNN on PhysioNet EEGMMIDB, with the L0–L3 leakage taxonomy and a matched-protocol ablation.

Code accompanying the manuscript **“A Systematic Benchmark of Motor Imagery Classifiers Under Leakage-Safe Evaluation”** by Muhammad Aftab ur Rehman, Muhammad Awais ur Rehman, Kiran Asghar, Farwa Zahra, Laiba Aamir, Fatima Maqbool (submitted, 2026).
Corresponding author: Muhammad Aftab ur Rehman, CYBEX – School of IT Professionals (Pvt) Limited, Faisalabad, Pakistan.

## Installation

```bash
git clone https://github.com/aftab802/eeg-mi-leakage-benchmark.git
cd eeg-mi-leakage-benchmark
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Python 3.10–3.13 (tested on 3.13 with TensorFlow 2.20). A GPU is optional but recommended for the deep-learning experiments.

## Data

PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB), 109 subjects, 64 channels, 160 Hz —
https://physionet.org/content/eegmmidb/1.0.0/ (DOI [10.13026/C28G6P](https://doi.org/10.13026/C28G6P)).
Nothing needs to be downloaded by hand: the loaders fetch the EDF files through `mne.datasets.eegbci`
on first use and cache them under `~/mne_data` (about 3.4 GB for all subjects; pass `--data-path` to change it).

## Reproducing the paper

```bash
# all four architectures, each under its own paper configuration
python scripts/run_benchmark.py
# matched-protocol ablation: 1D CNN under L0, L2 and the as-executed protocol
python scripts/run_benchmark.py --models conv1d --ablation
# four-class CNN under a subject-wise split instead of the original trial-level split
python scripts/run_benchmark.py --models fourclass --fourclass-protocol L0
```
Each architecture also has its own script (`run_group_aware_eegnet.py`, `run_conv1d_case_study.py`,
`run_csp_cnn.py`, `run_fourclass_cnn.py`).

Every run writes a JSON file to `results/` with the test metrics, confusion matrix, split sizes and, for subject
data, `train_test_subject_overlap` (0 under the leakage-safe protocol).

## Reproduction status

| Result in the paper | Value | Source of the number | Re-run with this repository? |
|---|---|---|---|
| EEGNet, L0 | 75.16 % | Original notebook, output preserved in `original_notebooks/` | **Re-run: 74.68 %** (Colab T4, TF 2.21) |
| Four-class CNN | 66 % | Original notebook, output preserved (66.48 % at 100 epochs) | **Re-run: 64.63 % trial-level, 58.33 % subject-wise (L0); 63.31 ± 5.05 % over six subject-wise splits** |
| CSP+CNN, L0 | 59.39 % | Re-run of this code (Colab T4); original 62.35 % run used a split that could not be recovered | **Re-run: 59.39 % (L0), 61.33 % (L2)** |
| 1D CNN, L0 | 64.65 % | Re-run of this code (Colab T4) | **Re-run: 64.65 % (L0)** |
| Ablation, 1D CNN L2 / as executed | 68.31 % / 69.53 % | Re-run of this code (Colab T4); replaces the unsupported 80.02 % | **Re-run** |

## Notes on the measured results

* Measured on Google Colab (NVIDIA T4, TensorFlow 2.21) with `notebooks/reproduce_mi_results.ipynb`, 10–11 Nov 2026.
* The 71.46 % 1D CNN figure of the companion case study is a **record-level** result with the test set reused for
  validation. The benchmark's Tables 5 and 7 now use the L0 (64.65 %) and L2 (68.31 %) re-runs.
* The four-class CNN was trained on a **trial-level** split (one permutation of all trials, 90/10) and has
  ~15.0 M parameters; its classes are left fist, right fist, both feet and **rest** (R01 baseline windows).
* The CSP+CNN head has 9,473 trainable parameters (see `tests/test_models.py`).

## Repository layout

```
cybex_eeg/        shared, leakage-aware library (splits, normalisation, features, models)
scripts/          one command per experiment in the paper; results are written to results/*.json
tests/            pytest suite run on synthetic data (no download needed)
original_notebooks/  the original Colab notebooks, with their printed outputs, for provenance
```

## Tests

```bash
pytest -q
```
Tests check the split/leakage guards, parameter counts against the paper, the PhysioNet four-class extraction on synthetic EDF files, and end-to-end runs of every model on synthetic EEG.

## Citation

See [`CITATION.cff`](CITATION.cff). Until the article is published, please cite the manuscript title above and
this repository.

## Related repositories

* [eeg-dementia-integrity-benchmark](https://github.com/aftab802/eeg-dementia-integrity-benchmark) — Data Integrity and Feature Engineering in Small-Cohort EEG Dementia Classification
* [eeg-mi-protocol-leakage-case-study](https://github.com/aftab802/eeg-mi-protocol-leakage-case-study) — A Case Study of Evaluation-Protocol Leakage in Motor Imagery EEG Classification
* [eeg-mi-four-class-cnn](https://github.com/aftab802/eeg-mi-four-class-cnn) — Four-Class Motor Imagery Classification Using a Residual Convolutional Neural Network
* [eeg-mi-fbcsp-cnn](https://github.com/aftab802/eeg-mi-fbcsp-cnn) — CSP+CNN: Filter-Bank Common Spatial Patterns with a Convolutional Classifier for Motor Imagery Classification from EEG
* [group-aware-eegnet](https://github.com/aftab802/group-aware-eegnet) — Group-Aware EEGNet: Subject-Independent Motor Imagery Classification from EEG
* [eeg-dementia-label-correction](https://github.com/aftab802/eeg-dementia-label-correction) — Label Correction and Spectral Features for Multiclass Dementia Detection from EEG
* [lead-eeg](https://github.com/aftab802/lead-eeg) — LEAD-EEG: A Lightweight Ensemble Framework for Dementia Detection from EEG

## License

MIT — see [LICENSE](LICENSE). The datasets are distributed by their owners under their own terms.

## Repeated subject-wise splits

Six random splits (seed 42 plus seeds 1–5, `.github/workflows/repeated-splits.yml`); mean ± SD of test accuracy. Raw values: `results_reference/repeated_splits.json`.

| Model | Subject-wise (L0) | Trial-level (L2) | As executed |
|---|---|---|---|
| EEGNet | 74.67 ± 2.80% | – | – |
| 1D CNN | 65.62 ± 2.12% | 68.35 ± 2.33% | 67.60 ± 6.90% |
| CSP+CNN | 61.29 ± 2.05% | 59.95 ± 1.63% | – |

The paired L2 − L0 difference is +2.73 ± 3.98 points for the 1D CNN (positive in four of six splits) and −1.3 ± 3.5 points for CSP+CNN.

Four-class CNN, subject-wise split, 100 epochs: 63.31 ± 5.05% over six splits (58.33, 67.86, 68.18, 64.94, 55.95, 64.61), against 64.63% for a single matched trial-level re-run.

Seeds 3 and 4 report identical trial-level accuracies; `seed-check.yml` confirms for CSP+CNN that the test sets and confusion matrices differ and only the totals coincide.
