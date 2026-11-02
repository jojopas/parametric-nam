#!/bin/zsh
# Train a parametric capture from a finished session, then export it as a ParametricWaveNet .nam.
# usage: PANAMA_DIR=/path/to/PANAMA scripts/train.sh <session_dir> [epochs=30] [batch=16]
# PANAMA_DIR is a PANAMA checkout installed into its own venv (pip install -e .).
# PY overrides the Python to use (default: $PANAMA_DIR/.venv/bin/python).
# LEARNING overrides the trainer config (default: configs/learning_mps.json, Apple GPU).
set -eo pipefail
S=${1:?session dir}; EPOCHS=${2:-30}; BATCH=${3:-16}
PANAMA=${PANAMA_DIR:?set PANAMA_DIR to a PANAMA checkout}
PY=${PY:-$PANAMA/.venv/bin/python}
ROOT=${0:a:h:h}
LEARNING=${LEARNING:-$ROOT/configs/learning_mps.json}
[ -f $S/plan.json ] || { echo "no $S/plan.json"; exit 1 }
export PYTHONPATH=$ROOT:$PANAMA
$PY -m parametric_nam.capture_plan data-config --dir $S
[ -f $S/model.json ] || $PY -m parametric_nam.capture_plan model-config --dir $S --template $ROOT/configs/model_wide.json
NAME=$($PY -c "import json;print(json.load(open('$S/plan.json'))['name'])")
(cd $PANAMA && $PY custom_train_full.py --base-outdir $S/train --data-config $S/data.json \
  --model-config $S/model.json --learning-config $LEARNING \
  --epochs $EPOCHS --batch-size $BATCH --check-val-every-n-epoch 2 2>&1 | tee $S/train.log)
# the checkpoint with the lowest validation ESR
setopt null_glob
CKPT=$(ls $S/train/*/lightning_logs/version_*/checkpoints/epoch=*ESR=*.ckpt 2>/dev/null \
  | awk -F'ESR=' '{split($2,a,"_"); print a[1], $0}' | sort -g | head -1 | cut -d' ' -f2-)
[ -n "$CKPT" ] || { echo "no checkpoint written: see $S/train.log"; exit 1 }
echo "checkpoint: $CKPT"
$PY -m parametric_nam.convert_panama --ckpt "$CKPT" --model-config $S/model.json --plan $S/plan.json --out $S/$NAME.nam
echo "model: $S/$NAME.nam"
