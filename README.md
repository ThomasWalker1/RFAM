# Recursive Feature Alignment Machines (RFAMs)

RFAMs are an improvement on Recursive Feature Machines (RFMs) using the principles of normal alignment details in "Normal Alignment: The Geometric Structure of Models Learning Sparse Data."

This code has been developed from https://github.com/aradha/recursive_feature_machines/tree/pip_install

We test RFAMs on the tabular datasets of https://github.com/LeoYu/neural-tangent-kernel-UCI

## Requirements

```bash
pip install numpy scipy scikit-learn
```

## Steps

### 1. Download data

```bash
bash setup.sh
```

### 2. Run training

```bash
python train.py -mode full -datadir data
```

Runs a grid search over regularization and alpha hyperparameters, trains on k-fold splits, and writes results to `outputs/results_rfam_full.log`.

To also run the alpha=0 (RFM) baseline:

```bash
python train.py -mode alpha0 -datadir data
```

Writes results to `outputs/results_rfam_alpha0.log`.

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `-datadir` | `data` | Path to the dataset directory |
| `-outdir` | `outputs` | Path to the output directory |
| `-mode` | `full` | `full` searches all alpha values; `alpha0` fixes alpha=0 |

### 3. Evaluate results

```bash
python eval.py
```

Reads `outputs/results_rfam_alpha0.log` and `outputs/results_rfam_full.log` and prints summary statistics: test accuracy, normal alignment, normalized effective rank, and attack success rate per epsilon.

## Run full experiment

To run everything end-to-end:

```bash
bash run_experiment.sh
```

See [run_experiment.sh](run_experiment.sh) for details.
