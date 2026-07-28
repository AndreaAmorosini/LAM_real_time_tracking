#!/usr/bin/env bash
set -Eeuo pipefail

python -m pip install -U pip setuptools wheel
python -m pip install "numpy==1.23.0" scipy six python-dotenv
python -m pip install --no-build-isolation git+https://github.com/mattloper/chumpy.git
python -m pip install -r requirements-lam-base.txt
