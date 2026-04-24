#!/bin/bash
# =============================================================================
# exp01_sequential — Train Stage 1 → Stage 2 → Inference → Local Eval
#
# Usage:
#   bash /media/DiscoLocal/IPCV/UE-ReID/experiments/exp01_sequential/run_exp01.sh
# =============================================================================

set -e

WORK_DIR="/media/DiscoLocal/IPCV/UE-ReID/Part-Aware-Transformer-main"
EXP_DIR="/media/DiscoLocal/IPCV/UE-ReID/experiments/exp01_sequential"

STAGE1_CFG="$EXP_DIR/stage1_urbam/config/train.yml"
STAGE2_CFG="$EXP_DIR/stage2_competition/config/train.yml"
TEST_CFG="$EXP_DIR/stage2_competition/config/test.yml"

STAGE1_LOG="$EXP_DIR/stage1_urbam/run.log"
STAGE2_LOG="$EXP_DIR/stage2_competition/run.log"

STAGE1_CKPT="$WORK_DIR/models/exp01_stage1/part_attention_vit_60.pth"
STAGE2_CKPT="$WORK_DIR/models/exp01_stage2/part_attention_vit_60.pth"

SUBMISSION_TXT="$WORK_DIR/submissions/exp01_submission.txt"
URBAM_EVAL_DIR="$WORK_DIR/urbam_eval_exp01"

mkdir -p "$WORK_DIR/submissions"

cd "$WORK_DIR"

# ── Stage 1: UrbAM training ──────────────────────────────────────────────────
echo "=============================================="
echo " Stage 1: UrbAM training (60 epochs)"
echo " Started: $(date)"
echo " Log: $STAGE1_LOG"
echo "=============================================="

python train.py --config_file "$STAGE1_CFG" 2>&1 | tee "$STAGE1_LOG"

if [ ! -f "$STAGE1_CKPT" ]; then
    echo "ERROR: Stage 1 checkpoint not found: $STAGE1_CKPT"
    echo "Stage 2 will NOT run. Check $STAGE1_LOG"
    exit 1
fi

echo ""
echo "Stage 1 done: $(date) — checkpoint OK"

# ── Stage 2: Competition fine-tune ───────────────────────────────────────────
echo ""
echo "=============================================="
echo " Stage 2: Competition fine-tune (60 epochs)"
echo " Started: $(date)"
echo " Log: $STAGE2_LOG"
echo "=============================================="

python train.py --config_file "$STAGE2_CFG" 2>&1 | tee "$STAGE2_LOG"

if [ ! -f "$STAGE2_CKPT" ]; then
    echo "ERROR: Stage 2 checkpoint not found: $STAGE2_CKPT"
    echo "Inference will NOT run. Check $STAGE2_LOG"
    exit 1
fi

echo ""
echo "Stage 2 done: $(date) — checkpoint OK"

# ── Inference (Kaggle submission CSV) ────────────────────────────────────────
echo ""
echo "=============================================="
echo " Inference: generating Kaggle submission"
echo " Started: $(date)"
echo "=============================================="

python update.py \
    --config_file "$TEST_CFG" \
    --track "$SUBMISSION_TXT"

echo "Submission saved: ${SUBMISSION_TXT%.txt}_submission.csv"

# ── Local eval on UrbAM inv split ────────────────────────────────────────────
echo ""
echo "=============================================="
echo " Local eval: UrbAM inv split (mAP & CMC)"
echo " Started: $(date)"
echo "=============================================="

python evaluate_urbam.py \
    --config_file "$TEST_CFG" \
    --out_dir "$URBAM_EVAL_DIR" \
    --k1 15 --k2 8 --lambda_value 0.1

echo ""
echo "=============================================="
echo " exp01_sequential COMPLETE: $(date)"
echo " Kaggle submission : ${SUBMISSION_TXT%.txt}_submission.csv"
echo " UrbAM local eval  : $URBAM_EVAL_DIR/submission.csv"
echo " Stage 1 log       : $STAGE1_LOG"
echo " Stage 2 log       : $STAGE2_LOG"
echo "=============================================="
