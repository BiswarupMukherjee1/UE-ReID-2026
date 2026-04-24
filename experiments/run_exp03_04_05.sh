#!/bin/bash
set -e

WORK_DIR="/media/DiscoLocal/IPCV/UE-ReID/Part-Aware-Transformer-main"
EXP_DIR="/media/DiscoLocal/IPCV/UE-ReID/experiments"

mkdir -p "$WORK_DIR/submissions"
cd "$WORK_DIR"

echo "=============================================="
echo " exp03: re-ranking on exp02 checkpoint"
echo " Started: $(date)"
echo "=============================================="

python update.py \
    --config_file "$EXP_DIR/exp03_reranking/config/test.yml" \
    --track "$WORK_DIR/submissions/exp03_submission.txt" \
    TEST.WEIGHT models/exp02_combined/part_attention_vit_60.pth

echo "exp03 done: $(date)"

echo ""
echo "=============================================="
echo " exp04: combined + REA (40 epochs)"
echo " Started: $(date)"
echo "=============================================="

python train.py \
    --config_file "$EXP_DIR/exp04_rea/config/train.yml" \
    2>&1 | tee "$EXP_DIR/exp04_rea/train.log"

python update.py \
    --config_file "$EXP_DIR/exp04_rea/config/test.yml" \
    --track "$WORK_DIR/submissions/exp04_submission.txt"

echo "exp04 done: $(date)"

echo ""
echo "=============================================="
echo " exp05: combined + low LR=0.0005 (40 epochs)"
echo " Started: $(date)"
echo "=============================================="

python train.py \
    --config_file "$EXP_DIR/exp05_lowlr/config/train.yml" \
    2>&1 | tee "$EXP_DIR/exp05_lowlr/train.log"

python update.py \
    --config_file "$EXP_DIR/exp05_lowlr/config/test.yml" \
    --track "$WORK_DIR/submissions/exp05_submission.txt"

echo "exp05 done: $(date)"

echo ""
echo "=============================================="
echo " ALL DONE: $(date)"
echo " exp03: submissions/exp03_submission_submission.csv"
echo " exp04: submissions/exp04_submission_submission.csv"
echo " exp05: submissions/exp05_submission_submission.csv"
echo "=============================================="
