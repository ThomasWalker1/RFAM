#!/usr/bin/env bash
set -euo pipefail

echo "==> Downloading data..."
bash setup.sh

echo "==> Training (alpha=0 baseline)..."
python train.py -mode alpha0 -dir data

echo "==> Training (full hyperparameter search)..."
python train.py -mode full -dir data

echo "==> Evaluating results..."
python eval.py
