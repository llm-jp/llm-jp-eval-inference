#!/bin/bash

set -eu

SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
LOG=logs/eval-`uuidgen`

mkdir -p logs

source ${SCRIPT_DIR}/venv/bin/activate
python ${SCRIPT_DIR}/offline_inference_transformers.py "$@" | tee $LOG
rm $LOG
deactivate
