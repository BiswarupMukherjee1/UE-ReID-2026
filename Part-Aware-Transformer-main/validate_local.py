"""
Local validation script for Urban Elements ReID.

Simulates the Kaggle challenge structure using the train split:
  - Gallery: all train images from c001, c002, c003
  - Query:   train images whose identity also appears in gallery
             (uses a held-out camera or random per-identity sample)

This lets us tune k1, k2, lambda_value locally without burning Kaggle submissions.

Usage:
    # Single run
    python validate_local.py --config_file config/UrbanElementsReID_test.yml

    # Parameter sweep
    python validate_local.py --config_file config/UrbanElementsReID_test.yml --sweep
"""

import os
import csv
import torch
import argparse
import numpy as np
from collections import defaultdict
from config import cfg
from model import make_model
from utils.re_ranking import re_ranking
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TrainSplitDataset(Dataset):
    def __init__(self, image_dir, entries, transform):
        self.image_dir = image_dir
        self.entries = entries  # list of (filename, identity_id, cam_id)
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        fname, pid, cam = self.entries[idx]
        img = Image.open(os.path.join(self.image_dir, fname)).convert('RGB')
        return self.transform(img), pid, cam


def get_transform(size=(256, 128)):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(model, loader):
    model.eval()
    features, pids, camids = [], [], []
    with torch.no_grad():
        for imgs, pid, cam in loader:
            imgs = imgs.cuda()
            ff = model(imgs).float()
            # TTA flip
            ff += model(torch.flip(imgs, [3])).float()
            ff = ff / torch.norm(ff, p=2, dim=1, keepdim=True)
            features.append(ff.cpu().numpy())
            pids.extend(pid.numpy() if hasattr(pid, 'numpy') else pid)
            camids.extend(cam)
    return np.vstack(features), np.array(pids), np.array(camids)


# ---------------------------------------------------------------------------
# mAP / CMC evaluation
# ---------------------------------------------------------------------------

def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    num_q = distmat.shape[0]
    indices = np.argsort(distmat, axis=1)

    aps = []
    cmc_scores = np.zeros(max_rank)

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_cam = q_camids[q_idx]

        order = indices[q_idx]
        # Remove same camera + same identity (junk)
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_cam)
        keep = ~remove

        g_pids_sorted = g_pids[order][keep]
        matches = (g_pids_sorted == q_pid).astype(np.float32)

        if matches.sum() == 0:
            continue

        # CMC
        first_hit = np.where(matches)[0]
        if len(first_hit) > 0 and first_hit[0] < max_rank:
            cmc_scores[first_hit[0]:] += 1

        # AP
        num_rel = matches.sum()
        tmp_cmc = matches.cumsum()
        tmp_cmc = tmp_cmc / (np.arange(len(matches)) + 1) * matches
        ap = tmp_cmc.sum() / num_rel
        aps.append(ap)

    mAP = np.mean(aps)
    cmc_scores /= num_q
    return mAP, cmc_scores


# ---------------------------------------------------------------------------
# Build query/gallery split from train CSV
# ---------------------------------------------------------------------------

def build_splits(train_csv_path):
    """
    Strategy: for each identity, put images from one camera in query,
    the rest in gallery. Mimics c004=query, c001/c002/c003=gallery structure.

    Since train has no c004, we simulate by picking the camera with fewest
    images per identity as the 'query camera'.
    """
    entries = []
    with open(train_csv_path) as f:
        for row in csv.DictReader(f):
            cam = row['cameraID']
            fname = row['imageName']
            pid = int(row['Corresponding Indexes'])
            entries.append((fname, pid, cam))

    # Group by identity
    id_to_entries = defaultdict(list)
    for e in entries:
        id_to_entries[e[1]].append(e)

    # For each identity, pick one camera as query (least frequent)
    query_entries = []
    gallery_entries = []

    for pid, ents in id_to_entries.items():
        cam_groups = defaultdict(list)
        for e in ents:
            cam_groups[e[2]].append(e)

        if len(cam_groups) < 2:
            # Only one camera — put in gallery only, can't use as query
            gallery_entries.extend(ents)
            continue

        # Pick camera with fewest images as query camera
        query_cam = min(cam_groups, key=lambda c: len(cam_groups[c]))
        for cam, cam_ents in cam_groups.items():
            if cam == query_cam:
                query_entries.extend(cam_ents)
            else:
                gallery_entries.extend(cam_ents)

    print(f"Query: {len(query_entries)} images, Gallery: {len(gallery_entries)} images")
    unique_q_ids = len(set(e[1] for e in query_entries))
    print(f"Unique query identities: {unique_q_ids}")
    return query_entries, gallery_entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="config/UrbanElementsReID_test.yml")
    parser.add_argument("--train_csv", default="/media/DiscoLocal/IPCV/UE-ReID/urban-elements-re-id-challenge-2026/Urban2026/train.csv")
    parser.add_argument("--image_dir", default="/media/DiscoLocal/IPCV/UE-ReID/urban-elements-re-id-challenge-2026/Urban2026/image_train")
    parser.add_argument("--k1", default=20, type=int)
    parser.add_argument("--k2", default=6, type=int)
    parser.add_argument("--lambda_value", default=0.3, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # Build splits
    print("Building query/gallery split from train set...")
    query_entries, gallery_entries = build_splits(args.train_csv)

    # Datasets
    size = tuple(cfg.INPUT.SIZE_TEST)
    transform = get_transform(size)
    query_ds = TrainSplitDataset(args.image_dir, query_entries, transform)
    gallery_ds = TrainSplitDataset(args.image_dir, gallery_entries, transform)

    query_loader = DataLoader(query_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=4)
    gallery_loader = DataLoader(gallery_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=4)

    # Model
    print(f"Loading model from {cfg.TEST.WEIGHT}")
    model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
    model.load_param(cfg.TEST.WEIGHT)
    model.cuda()
    model.eval()

    # Extract features
    print("Extracting query features...")
    qf, q_pids, q_cams = extract_features(model, query_loader)
    print("Extracting gallery features...")
    gf, g_pids, g_cams = extract_features(model, gallery_loader)
    print(f"qf: {qf.shape}, gf: {gf.shape}")

    # Save features for sweep reuse
    np.save("./val_qf.npy", qf)
    np.save("./val_gf.npy", gf)
    np.save("./val_q_pids.npy", q_pids)
    np.save("./val_g_pids.npy", g_pids)
    np.save("./val_q_cams.npy", q_cams)
    np.save("./val_g_cams.npy", g_cams)
    print("Saved validation features.")

    def run_eval(k1, k2, lam):
        q_g = 1.0 - np.dot(qf, gf.T)
        q_q = 1.0 - np.dot(qf, qf.T)
        g_g = 1.0 - np.dot(gf, gf.T)
        dist = re_ranking(q_g, q_q, g_g, k1=k1, k2=k2, lambda_value=lam)
        mAP, cmc = evaluate(dist, q_pids, g_pids, q_cams, g_cams)
        return mAP, cmc

    if args.sweep:
        print("\n=== Parameter sweep ===")
        print(f"{'k1':>4} {'k2':>4} {'lambda':>8} {'mAP':>8} {'R@1':>8} {'R@5':>8}")
        best = (0, None)
        for k1 in [10, 15, 20, 25]:
            for k2 in [3, 4, 6, 8]:
                for lam in [0.1, 0.2, 0.3, 0.5]:
                    mAP, cmc = run_eval(k1, k2, lam)
                    print(f"{k1:>4} {k2:>4} {lam:>8.1f} {mAP:>8.4f} {cmc[0]:>8.4f} {cmc[4]:>8.4f}")
                    if mAP > best[0]:
                        best = (mAP, (k1, k2, lam))
        print(f"\nBest: mAP={best[0]:.4f} with k1={best[1][0]}, k2={best[1][1]}, lambda={best[1][2]}")
    else:
        print(f"\nRunning with k1={args.k1}, k2={args.k2}, lambda={args.lambda_value}")
        mAP, cmc = run_eval(args.k1, args.k2, args.lambda_value)
        print(f"\nmAP:  {mAP:.4f}")
        print(f"R@1:  {cmc[0]:.4f}")
        print(f"R@5:  {cmc[4]:.4f}")
        print(f"R@10: {cmc[9]:.4f}")
