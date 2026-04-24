"""
Re-ranking parameter sweep on exp02 features.
Uses the UrbAM inv split as local eval proxy.
Runs entirely on CPU — no GPU needed.

Usage:
    python sweep_reranking.py
"""

import numpy as np
import csv
import os
import sys
sys.path.insert(0, '/media/DiscoLocal/IPCV/UE-ReID/Part-Aware-Transformer-main')
from utils.re_ranking import re_ranking

# ── Load exp02 features ───────────────────────────────────────────────────────
QF_PATH = './qf.npy'
GF_PATH = './gf.npy'
GALLERY_CSV = '/media/DiscoLocal/IPCV/UE-ReID/urban-elements-re-id-challenge-2026/Urban2026/test.csv'
QUERY_CSV   = '/media/DiscoLocal/IPCV/UE-ReID/urban-elements-re-id-challenge-2026/Urban2026/query.csv'
OUT_DIR     = './submissions/sweep'
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading features...")
qf = np.load(QF_PATH)
gf = np.load(GF_PATH)
print(f"qf: {qf.shape}  gf: {gf.shape}")

# Pre-compute distance matrices (expensive, do once)
print("Computing distance matrices...")
q_g_dist = 1.0 - np.dot(qf, gf.T)
q_q_dist = 1.0 - np.dot(qf, qf.T)
g_g_dist = 1.0 - np.dot(gf, gf.T)
print("Done. Starting sweep...\n")

# ── Parameter grid ────────────────────────────────────────────────────────────
k1_values  = [10, 15, 20, 25, 30]
k2_values  = [4,  6,  8,  10]
lam_values = [0.1, 0.2, 0.3, 0.5]

# ── Sweep ─────────────────────────────────────────────────────────────────────
results = []
total = len(k1_values) * len(k2_values) * len(lam_values)
count = 0

for k1 in k1_values:
    for k2 in k2_values:
        for lam in lam_values:
            count += 1
            dist = re_ranking(q_g_dist, q_q_dist, g_g_dist,
                              k1=k1, k2=k2, lambda_value=lam)
            indices = np.argsort(dist, axis=1)[:, :100]

            # Save submission CSV
            fname = f"sweep_k1{k1}_k2{k2}_lam{int(lam*10)}.csv"
            fpath = os.path.join(OUT_DIR, fname)
            lista = ["{:06d}.jpg".format(i) for i in range(1, len(indices)+1)]
            with open(fpath, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['imageName', 'Corresponding Indexes'])
                for name, row in zip(lista, indices):
                    w.writerow([name, ' '.join(map(str, row + 1))])

            results.append((k1, k2, lam, fname))
            print(f"[{count:3d}/{total}] k1={k1:2d} k2={k2:2d} lam={lam:.1f} → {fname}")

print(f"\nDone. {len(results)} CSVs saved to {OUT_DIR}/")
print("\nTo evaluate each one locally against UrbAM, run evaluate_csv.py.")
print("To submit the best to Kaggle, pick the CSV with highest local mAP.")

# Print summary
print("\n--- All parameter combinations ---")
print(f"{'k1':>4} {'k2':>4} {'lam':>5}  {'file'}")
for k1, k2, lam, fname in results:
    print(f"{k1:>4} {k2:>4} {lam:>5.1f}  {fname}")
