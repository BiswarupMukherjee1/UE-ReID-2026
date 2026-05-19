"""
submit_exp16.py — CORRECT competition feature extraction for exp16.

Loads actual competition data:
  - image_query/ + query.csv  → 928 images, all c004 → view_id=1 (backward)
  - image_test/  + test.csv   → 2844 images, c001/c002/c003 → view_id=0 (forward)

Produces qf.npy [928, 1024] and gf.npy [2844, 1024] for CAJ reranking.

Usage:
    cd /media/DiscoLocal/IPCV/UE-ReID/Part-Aware-Transformer-main

    python /media/DiscoLocal/IPCV/UE-ReID/experiments/exp16_viewemb_equalized/submit_exp16.py \
        --weight models/./exp16_viewemb_equalized/part_attention_vit_40.pth

    # Then run CAJ:
    python caj_filter_rerank.py \
        --output submission_ep40.csv \
        --merge_bins --k1 15 --k2 4 \
        --qf qf.npy --gf gf.npy
"""

import os
import csv
import sys
import torch
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms as T

EXP16_DIR = os.path.dirname(os.path.abspath(__file__))
PAT_DIR   = '/media/DiscoLocal/IPCV/UE-ReID/Part-Aware-Transformer-main'
DATA_ROOT = '/media/DiscoLocal/IPCV/UE-ReID/urban-elements-re-id-challenge-2026/Urban2026'

sys.path.insert(0, EXP16_DIR)
sys.path.insert(0, PAT_DIR)
os.chdir(PAT_DIR)

from config import cfg
from model import make_model
from utils.logger import setup_logger


# ------------------------------------------------------------------ #
# Image transform (same as training: normalize to [-1, 1])
# ------------------------------------------------------------------ #
def build_test_transform(size=(256, 128)):
    return T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


# ------------------------------------------------------------------ #
# Load items from CSV
# ------------------------------------------------------------------ #
def load_items_from_csv(csv_path, img_dir):
    """
    Returns list of (img_path, camid_int) in CSV row order.
    Order is critical — must match query.csv / test.csv row order
    because submission indexes and CAJ camera labels depend on it.
    """
    items = []
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            img_path = os.path.join(img_dir, row['imageName'])
            camid    = int(row['cameraID'][1:])   # 'c004' → 4
            items.append((img_path, camid))
    return items


# ------------------------------------------------------------------ #
# Feature extraction
# ------------------------------------------------------------------ #
def extract_features(model, items, transform, batch_size=64, desc=""):
    """
    Extract L2-normalized features for a list of (img_path, camid) items.
    Applies view_ids: camid==4 → 1 (backward), else → 0 (forward).
    """
    model.eval()
    all_features = []
    n = len(items)

    for start in range(0, n, batch_size):
        batch = items[start : start + batch_size]

        imgs = []
        camids = []
        for img_path, camid in batch:
            img = Image.open(img_path).convert('RGB')
            imgs.append(transform(img))
            camids.append(camid)

        imgs     = torch.stack(imgs).cuda()
        cam_t    = torch.tensor(camids, dtype=torch.long)
        view_ids = (cam_t == 4).long().cuda()

        with torch.no_grad():
            feats = model(imgs, view_ids=view_ids).float()

        # L2 normalize
        fnorm = torch.norm(feats, p=2, dim=1, keepdim=True)
        feats = feats.div(fnorm.expand_as(feats))
        all_features.append(feats.cpu())

        if (start // batch_size) % 5 == 0:
            print(f"  {desc}: {min(start + batch_size, n)}/{n}")

    return torch.cat(all_features, 0)


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Competition inference exp16")
    parser.add_argument("--weight",    required=True, type=str,
                        help="Path to checkpoint, e.g. models/.../part_attention_vit_40.pth")
    parser.add_argument("--qf_out",   default="./qf.npy",  type=str)
    parser.add_argument("--gf_out",   default="./gf.npy",  type=str)
    parser.add_argument("--batch",    default=64, type=int)
    args = parser.parse_args()

    test_yml = os.path.join(EXP16_DIR, 'config/test.yml')
    cfg.merge_from_file(test_yml)
    cfg.merge_from_list(["TEST.WEIGHT", args.weight])
    cfg.freeze()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    logger = setup_logger("PAT", "models/./exp16_viewemb_equalized", if_train=False)
    logger.info(f"Loading checkpoint: {args.weight}")

    # ---------------------------------------------------------------- #
    # Build model
    # ---------------------------------------------------------------- #
    model = make_model(cfg, "part_attention_vit", 0, 0, 0)
    model.load_param(args.weight)
    model = model.cuda()
    model.eval()

    # Confirm view_embedding
    if hasattr(model.base, 'view_embedding'):
        ve = model.base.view_embedding
        logger.info(f"view_embedding loaded: norm_fwd={ve[0].norm():.4f}, "
                    f"norm_bwd={ve[1].norm():.4f}")
        if ve[0].norm().item() < 0.01:
            logger.warning("view_embedding norms are near zero — "
                           "did you load a trained checkpoint?")
    else:
        logger.warning("view_embedding not found!")

    transform = build_test_transform(size=(256, 128))

    # ---------------------------------------------------------------- #
    # Load competition query (928 images, all c004)
    # ---------------------------------------------------------------- #
    query_csv = os.path.join(DATA_ROOT, 'query.csv')
    query_dir = os.path.join(DATA_ROOT, 'image_query')
    query_items = load_items_from_csv(query_csv, query_dir)
    logger.info(f"Query: {len(query_items)} images")

    q_cams = [c for _, c in query_items]
    assert all(c == 4 for c in q_cams), "Expected all query camids = 4"
    logger.info(f"All query camids = 4 ✓ → view_id=1 (backward)")

    print("Extracting query features...")
    qf = extract_features(model, query_items, transform, args.batch, "query")
    logger.info(f"qf shape: {qf.shape}")   # expected [928, 1024]

    # ---------------------------------------------------------------- #
    # Load competition gallery (2844 images, c001/c002/c003)
    # ---------------------------------------------------------------- #
    test_csv = os.path.join(DATA_ROOT, 'test.csv')
    test_dir = os.path.join(DATA_ROOT, 'image_test')
    gallery_items = load_items_from_csv(test_csv, test_dir)
    logger.info(f"Gallery: {len(gallery_items)} images")

    g_cams = [c for _, c in gallery_items]
    assert all(c != 4 for c in g_cams), "Expected no c004 in gallery"
    logger.info(f"Gallery camids: all in {{1,2,3}} ✓ → view_id=0 (forward)")

    print("Extracting gallery features...")
    gf = extract_features(model, gallery_items, transform, args.batch, "gallery")
    logger.info(f"gf shape: {gf.shape}")   # expected [2844, 1024]

    # ---------------------------------------------------------------- #
    # Save
    # ---------------------------------------------------------------- #
    qf_np = qf.numpy()
    gf_np = gf.numpy()
    np.save(args.qf_out, qf_np)
    np.save(args.gf_out, gf_np)
    logger.info(f"Saved: {args.qf_out} {qf_np.shape}")
    logger.info(f"Saved: {args.gf_out} {gf_np.shape}")

    # ---------------------------------------------------------------- #
    # Sanity: cosine similarity between first query and all gallery
    # ---------------------------------------------------------------- #
    sim = float(np.dot(qf_np[0], gf_np.T).max())
    logger.info(f"Sanity check — max cosine sim of query[0] vs gallery: {sim:.4f} "
                f"(>0.3 = reasonable)")

    print()
    print("=" * 60)
    print("Features saved. Now run CAJ reranking:")
    print()
    print("python caj_filter_rerank.py \\")
    print(f"    --output submission_ep40_caj.csv \\")
    print("    --merge_bins --k1 15 --k2 4 \\")
    print(f"    --qf {args.qf_out} --gf {args.gf_out}")
    print("=" * 60)