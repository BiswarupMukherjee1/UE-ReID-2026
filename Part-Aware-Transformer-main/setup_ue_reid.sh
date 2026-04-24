#!/usr/bin/env bash
# =============================================================================
# UE-ReID 2026 — session setup
# Usage:  source /media/DiscoLocal/IPCV/UE-ReID/Part-Aware-Transformer-main/setup_ue_reid.sh
# =============================================================================

ROOT="/media/DiscoLocal/IPCV/UE-ReID"
PROJECT="$ROOT/Part-Aware-Transformer-main"
EXPERIMENTS="$ROOT/experiments"

# Best configs as of 2026-04-21
CKPT_BASELINE="$PROJECT/model/part_attention_vit_40.pth"
CKPT_BEST="$PROJECT/models/exp02_combined/part_attention_vit_60.pth"
TRAIN_CFG_BEST="$EXPERIMENTS/exp02_combined/config/train.yml"
TEST_CFG_BEST="$EXPERIMENTS/exp02_combined/config/test.yml"

# --- activate conda env -------------------------------------------------------
if command -v conda &>/dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate pat
    echo "[env] conda env 'pat' activated"
else
    echo "[warn] conda not found — activate 'pat' manually"
fi

# --- fix: ensure correct torch/cuda is loaded ---------------------------------
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'" 2>/dev/null \
    && echo "[gpu] CUDA OK" \
    || echo "[warn] CUDA not available — check torch version with: python -c 'import torch; print(torch.__version__)'"

# --- cd into project ----------------------------------------------------------
cd "$PROJECT" || { echo "[error] Project dir not found: $PROJECT"; return 1; }
echo "[dir] $(pwd)"

# =============================================================================
# STATUS REPORT
# =============================================================================
echo ""
echo "========================================"
echo "  UE-ReID 2026 — session status"
echo "========================================"

# --- checkpoint availability --------------------------------------------------
echo ""
echo "[ checkpoints ]"
for f in \
    "$CKPT_BASELINE" \
    "$CKPT_BEST" \
    "$PROJECT/models/exp04_rea/part_attention_vit_40.pth" \
    "$PROJECT/models/exp05_lowlr/part_attention_vit_40.pth"; do
    if [ -f "$f" ]; then
        printf "  [OK]      %s\n" "${f#$ROOT/}"
    else
        printf "  [MISSING] %s\n" "${f#$ROOT/}"
    fi
done

# --- latest submission CSVs ---------------------------------------------------
echo ""
echo "[ latest submissions ]"
find "$PROJECT/submissions" -name "*_submission.csv" \
    -printf "  %TY-%Tm-%Td %TH:%TM  %f\n" 2>/dev/null | sort | tail -8

# --- Kaggle scores ------------------------------------------------------------
echo ""
echo "[ Kaggle leaderboard (public mAP) ]"
echo "  0.10183  baseline          part_attention_vit_40.pth"
echo "  0.04498  exp01_sequential  UrbAM→competition (catastrophic forgetting)"
echo "  0.12052  exp02_combined    UrbAM+competition simultaneous  *** BEST ***"
echo "  0.12034  exp03_reranking   exp02 + re-ranking (marginal loss)"
echo "  0.10386  exp04_rea         combined + REA 40 epochs"
echo "  0.10056  exp05_lowlr       combined + LR=0.0005 40 epochs"

echo ""
echo "========================================"
echo ""

# =============================================================================
# CONVENIENCE ALIASES
# =============================================================================

# Run inference with best checkpoint → generates submission CSV
alias run_infer="python update.py \
    --config_file $TEST_CFG_BEST \
    --track $PROJECT/submissions/latest_submission.txt"

# Quick re-run of best experiment training
alias run_best="bash $EXPERIMENTS/exp02_combined/run_exp02.sh"

# Run all pending experiments
alias run_all="nohup bash $EXPERIMENTS/run_exp03_04_05.sh \
    > $EXPERIMENTS/run_all.log 2>&1 & echo \$!"

# Watch training log live
alias watchlog="tail -f $EXPERIMENTS/run_all.log"

# Check GPU
alias gpu="nvidia-smi"

# Scores reminder
alias scores='echo "
  0.10183  baseline
  0.04498  exp01_sequential  (bad — forgetting)
  0.12052  exp02_combined    *** BEST ***
  0.12034  exp03_reranking
  0.10386  exp04_rea
  0.10056  exp05_lowlr
"'

echo "[ aliases ready ]"
echo "  run_infer  — inference with best checkpoint (exp02)"
echo "  run_best   — retrain exp02 (combined, 60 epochs)"
echo "  run_all    — run exp03/04/05 in background"
echo "  watchlog   — tail the run_all log"
echo "  gpu        — nvidia-smi"
echo "  scores     — print mAP leaderboard"
echo ""

# =============================================================================
# NEXT STEPS REMINDER
# =============================================================================
echo "[ next steps ]"
echo "  1. Combined training 60 epochs + REA (exp04 failed at 40 epochs)"
echo "  2. Class filtering post-processing on exp02 checkpoint"
echo "  3. Fine-tune from exp02 checkpoint instead of ImageNet"
echo ""
