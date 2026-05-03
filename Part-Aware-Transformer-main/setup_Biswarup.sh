#!/usr/bin/env bash
# =============================================================================
# UE-ReID 2026 — session setup
# Usage: source /media/DiscoLocal/IPCV/UE-ReID/Part-Aware-Transformer-main/setup_ue_reid.sh
# =============================================================================

ROOT="/media/DiscoLocal/IPCV/UE-ReID"
PROJECT="$ROOT/Part-Aware-Transformer-main"

# --- activate conda env -------------------------------------------------------
if command -v conda &>/dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate pat
    echo "[env] conda env 'pat' activated"
else
    echo "[warn] conda not found — activate 'pat' manually"
fi

# --- fix numpy version (must be 1.26.4 for torch compatibility) ---------------
NUMPY_VER=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null)
if [ "$NUMPY_VER" != "1.26.4" ]; then
    echo "[fix] numpy is $NUMPY_VER — downgrading to 1.26.4..."
    pip install numpy==1.26.4 --break-system-packages -q
    echo "[fix] numpy downgraded"
else
    echo "[env] numpy $NUMPY_VER OK"
fi

# --- verify torch + cuda ------------------------------------------------------
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null \
    && echo "[gpu] CUDA OK — torch $(python -c 'import torch; print(torch.__version__)')" \
    || echo "[warn] CUDA not available"

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
    "$PROJECT/models/exp13_vitlarge_uam_unified/part_attention_vit_40.pth" \
    "$PROJECT/models/exp13_vitlarge_uam_unified/part_attention_vit_35.pth" \
    "$PROJECT/models/exp13_continued/part_attention_vit_40.pth"; do
    if [ -f "$f" ]; then
        printf "  [OK]      %s\n" "${f#$ROOT/}"
    else
        printf "  [MISSING] %s\n" "${f#$ROOT/}"
    fi
done

# --- key scripts --------------------------------------------------------------
echo ""
echo "[ key scripts ]"
for f in \
    "$PROJECT/update.py" \
    "$PROJECT/class_filter_rerank.py" \
    "$PROJECT/eval_cls_concat.py"; do
    if [ -f "$f" ]; then
        printf "  [OK]      %s\n" "$(basename $f)"
    else
        printf "  [MISSING] %s\n" "$(basename $f)"
    fi
done

# --- latest submissions -------------------------------------------------------
echo ""
echo "[ latest submissions (last 5) ]"
ls -lt "$PROJECT/submissions/"*.csv 2>/dev/null | head -5 | awk '{print "  "$6,$7,$8,$9}' | xargs -I{} basename {} 2>/dev/null
ls -lt "$PROJECT/submissions/"*.csv 2>/dev/null | head -5 | awk '{print "  "$6,$7, $9}'

# --- scores -------------------------------------------------------------------
echo ""
echo "[ Kaggle scores — public mAP ]"
echo "  0.13267  exp13_ep40_classfilter_mergedbins   *** BEST ***"
echo "  0.13133  exp13_ep40_classfilter"
echo "  0.12998  exp13_ep40 (standard)"
echo "  0.12836  exp13_ep40_clsconcat_rerank_v3"
echo "  0.12434  exp13_ep35"
echo "  0.12127  exp13_continued_ep40 (hurt — LR restart)"
echo "  0.12052  exp02_combined (ViT-Base)"
echo "  0.11167  exp13_ep40_clsconcat_norerank"
echo "  0.10183  baseline"

echo ""
echo "========================================"

# =============================================================================
# CONVENIENCE ALIASES
# =============================================================================
BEST_CKPT="$PROJECT/models/exp13_vitlarge_uam_unified/part_attention_vit_40.pth"
BEST_CFG="/media/DiscoLocal/IPCV/UE-ReID/experiments/exp13_vitlarge_uam_unified/config/test.yml"

# Standard inference — regenerates qf.npy gf.npy
alias run_infer="python update.py \
    --config_file $BEST_CFG \
    --track $PROJECT/submissions/exp13_ep40_fresh.txt \
    TEST.WEIGHT $BEST_CKPT"

# Class filtering with merged bins (current best pipeline)
alias run_classfilter="python class_filter_rerank.py \
    --merge_bins \
    --output $PROJECT/submissions/classfilter_mergedbins_latest.csv"

# GPU status
alias gpu="nvidia-smi"

# Scores reminder
alias scores='echo "
  0.13267  classfilter_mergedbins  *** BEST ***
  0.13133  classfilter
  0.12998  exp13_ep40_standard
  0.12052  exp02_combined
  0.10183  baseline
"'

echo ""
echo "[ aliases ready ]"
echo "  run_infer      — inference with best checkpoint → qf.npy gf.npy"
echo "  run_classfilter — class filter + merged bins → submission CSV"
echo "  gpu            — nvidia-smi"
echo "  scores         — print mAP leaderboard"
echo ""

echo "[ next steps ]"
echo "  1. Averaged CLS (last 2 layers) + class filter + merged bins"
echo "  2. exp14 — AdamW, resolution change, fresh training"
echo "  3. DANN — camera invariant features"
echo ""