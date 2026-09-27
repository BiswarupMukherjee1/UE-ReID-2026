"""Average L2-normalised features from several checkpoints.

Example:
    python ensemble_features.py \
        --qf features/qf_ep55.npy features/qf_ep60.npy \
        --gf features/gf_ep55.npy features/gf_ep60.npy \
        --out features/ens
Writes features/ens_qf.npy and features/ens_gf.npy.
"""

import argparse

import numpy as np


def average(paths):
    feats = [np.load(p) for p in paths]
    mean = np.sum(feats, axis=0)
    return mean / np.linalg.norm(mean, axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qf", nargs="+", required=True, help="query feature files")
    parser.add_argument("--gf", nargs="+", required=True, help="gallery feature files")
    parser.add_argument("--out", required=True, help="output prefix")
    args = parser.parse_args()
    if len(args.qf) != len(args.gf):
        parser.error("--qf and --gf need the same number of files")

    qf, gf = average(args.qf), average(args.gf)
    np.save(f"{args.out}_qf.npy", qf)
    np.save(f"{args.out}_gf.npy", gf)
    print(f"query {qf.shape}, gallery {gf.shape} -> {args.out}_qf.npy, {args.out}_gf.npy")


if __name__ == "__main__":
    main()
