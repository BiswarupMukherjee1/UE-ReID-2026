#!/bin/bash
set -e
WORK_DIR="/media/DiscoLocal/IPCV/UE-ReID/Part-Aware-Transformer-main"
EXP_DIR="/media/DiscoLocal/IPCV/UE-ReID/experiments"
mkdir -p "$WORK_DIR/submissions"
cd "$WORK_DIR"

echo "=============================================="
echo " exp04 (REA, 80 epochs) — started: $(date)"
echo "=============================================="
python train.py --config_file "$EXP_DIR/exp04_rea/config/train.yml" 2>&1 | tee "$EXP_DIR/exp04_rea/train_80ep.log"
if [ ! -f "$WORK_DIR/models/exp04_rea/part_attention_vit_80.pth" ]; then
    echo "ERROR: exp04 checkpoint not found. Stopping."; exit 1
fi
python update.py --config_file "$EXP_DIR/exp04_rea/config/test.yml" --track "$WORK_DIR/submissions/exp04_80ep_submission.txt"
echo "exp04 done: $(date)"

echo "=============================================="
echo " exp05 (low LR=0.0005, 80 epochs) — started: $(date)"
echo "=============================================="
python train.py --config_file "$EXP_DIR/exp05_lowlr/config/train.yml" 2>&1 | tee "$EXP_DIR/exp05_lowlr/train_80ep.log"
if [ ! -f "$WORK_DIR/models/exp05_lowlr/part_attention_vit_80.pth" ]; then
    echo "ERROR: exp05 checkpoint not found. Stopping."; exit 1
fi
python update.py --config_file "$EXP_DIR/exp05_lowlr/config/test.yml" --track "$WORK_DIR/submissions/exp05_80ep_submission.txt"
echo "exp05 done: $(date)"

echo "ALL DONE: $(date)"
