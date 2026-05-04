#!/usr/bin/env bash
set -euo pipefail

echo "==> Downloading data..."
bash setup.sh

echo "==> Training (alpha=1 baseline)..."
python train.py -mode alpha1 -datadir data

echo "==> Training (full)..."
python train.py -mode alpha0 -datadir data

echo "==> Evaluating results..."
python eval.py
