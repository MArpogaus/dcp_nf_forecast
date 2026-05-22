#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=3
source /app/.venv/bin/activate
exec dvc repro -f train@dataset0-normal_baseline train@dataset0-lognormal_baseline train@dataset0-bernstein_nf train@dataset0-bernstein_nf_lognormal train@dataset0-bernstein_nf_scale train@dataset0-bernstein_nf_scale_lognormal train@dataset0-spline_nf train@dataset0-spline_nf_lognormal train@dataset0-spline_nf_scale train@dataset0-spline_nf_scale_lognormal
