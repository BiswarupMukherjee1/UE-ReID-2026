#!/bin/bash
# =============================================================================
# exp02_combined — Train on UrbAM + Competition simultaneously, then infer
#
# Usage:
#   bash /media/DiscoLocal/IPCV/UE-ReID/experiments/exp02_combined/run_exp02.sh
# =============================================================================

set -e

WORK_DIR="/media/DiscoLocal/IPCV/UE-ReID/Part-Aware-Transformer-main"
EXP_DIR="/media/DiscoLocal/IPCV/UE-ReID/experiments/exp02_combined"

TRAIN_CFG="$EXP_DIR/config/train.yml"
TEST_CFG="$EXP_DIR/config/test.yml"
TRAIN_LOG="$EXP_DIR/train.log"
CKPT="$WORK_DIR/models/exp02_combined/part_attention_vit_60.pth"
SUBMISSION_TXT="$WORK_DIR/submissions/exp02_submission.txt"
URBAM_EVAL_DIR="$WORK_DIR/urbam_eval_exp02"

mkdir -p "$WORK_DIR/submissions"

cd "$WORK_DIR"

# ── Training ─────────────────────────────────────────────────────────────────
echo "=============================================="
echo " exp02_combined: UrbAM + Competition (60 epochs)"
echo " Started: $(date)"
echo " Log: $TRAIN_LOG"
echo "=============================================="

python train.py --config_file "$TRAIN_CFG" 2>&1 | tee "$TRAIN_LOG"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT"
    exit 1
fi

echo "Training done: $(date)"

# ── Inference ─────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo " Inference: generating Kaggle submission"
echo " Started: $(date)"
echo "=============================================="

python update.py \
    --config_file "$TEST_CFG" \
    --track "$SUBMISSION_TXT"

echo "Submission: ${SUBMISSION_TXT%.txt}_submission.csv"

# ── Local eval ────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo " Local eval: UrbAM inv split"
echo " Started: $(date)"
echo "=============================================="

python evaluate_urbam.py \
    --config_file "$TEST_CFG" \
    --out_dir "$URBAM_EVAL_DIR" \
    --k1 15 --k2 8 --lambda_value 0.1

echo ""
echo "=============================================="
echo " exp02_combined COMPLETE: $(date)"
echo " Kaggle submission : ${SUBMISSION_TXT%.txt}_submission.csv"
echo " UrbAM local eval  : $URBAM_EVAL_DIR/"
echo " Train log         : $TRAIN_LOG"
echo "=============================================="
