"""Batch-composition statistics for the two identity samplers.

Runs one or more epochs of index generation (no images are loaded, no GPU
needed) and reports, per batch of P identities x K instances:

  * how many backward-camera (c004) images the batch contains,
  * how many cross-view positive pairs it contains, i.e. pairs of images
    of the same identity where one image is c004 and the other is not.
    Only pairs inside the same K-image group are counted, which is what
    the batch-hard triplet loss can use as positives.

Both samplers are imported from the training code (PAT's default sampler and
experiments/exp17_dann_equalized/sampler_exp17.py), so the numbers describe
exactly what the trainer sees.

Example:
    python tools/sampler_stats.py \
        --pat-root pat \
        --competition-root /path/to/Urban2026 \
        --uam-root /path/to/UAM_Unified
"""

import argparse
import csv
import os
import random
import sys

import numpy as np

UAM_PID_OFFSET = 2000   # same offset as data/datasets/UAM_Unified.py
BACKWARD_CAM = 4


def read_competition(root):
    items = []
    with open(os.path.join(root, "train.csv"), newline="") as f:
        for row in csv.DictReader(f):
            pid = int(row["Corresponding Indexes"])
            cam = int(row["cameraID"].strip()[1:])
            items.append((row["imageName"], pid, cam, {}))
    return items


def read_uam(root):
    items = []
    for split in ("train", "query", "test"):
        with open(os.path.join(root, f"{split}.csv"), newline="") as f:
            for row in csv.DictReader(f):
                pid = UAM_PID_OFFSET + int(row["objectID"])
                cam = int(row["cameraID"].strip()[1:])
                items.append((row["imageName"], pid, cam, {}))
    return items


def batch_stats(order, data, batch_size, num_instances):
    n_batches = len(order) // batch_size
    c004_counts, pair_counts = [], []
    for b in range(n_batches):
        batch = order[b * batch_size:(b + 1) * batch_size]
        c004_counts.append(sum(data[i][2] == BACKWARD_CAM for i in batch))
        pairs = 0
        for g in range(0, batch_size, num_instances):
            group = batch[g:g + num_instances]
            n_back = sum(data[i][2] == BACKWARD_CAM for i in group)
            pairs += n_back * (len(group) - n_back)
        pair_counts.append(pairs)
    return n_batches, np.array(c004_counts), np.array(pair_counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pat-root", required=True,
                        help="folder that contains data/samplers/triplet_sampler.py")
    parser.add_argument("--sampler-dir", default=None,
                        help="folder with sampler_exp17.py "
                             "(default: experiments/exp17_dann_equalized)")
    parser.add_argument("--competition-root", required=True)
    parser.add_argument("--uam-root", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-instances", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sampler_dir = args.sampler_dir or os.path.join(
        repo_root, "experiments", "exp17_dann_equalized")
    sys.path.insert(0, os.path.abspath(args.pat_root))
    sys.path.insert(0, os.path.abspath(sampler_dir))
    from data.samplers.triplet_sampler import RandomIdentitySampler
    from sampler_exp17 import CameraEqualizedSampler

    random.seed(args.seed)
    np.random.seed(args.seed)

    data = read_competition(args.competition_root) + read_uam(args.uam_root)
    pids = {d[1] for d in data}
    cross_view_pids = {d[1] for d in data if d[2] == BACKWARD_CAM}
    n_back = sum(d[2] == BACKWARD_CAM for d in data)
    print(f"images: {len(data)}  identities: {len(pids)}  "
          f"c004 images: {n_back} ({100 * n_back / len(data):.1f}%)  "
          f"identities with a c004 image: {len(cross_view_pids)}")
    print(f"batch = {args.batch_size // args.num_instances} identities x "
          f"{args.num_instances} images, {args.epochs} epochs per sampler\n")

    samplers = {
        "default (PAT)": RandomIdentitySampler,
        "camera-balanced": CameraEqualizedSampler,
    }
    header = (f"{'sampler':<22}{'batches/ep':>11}{'c004/batch':>12}"
              f"{'no c004':>10}{'x-view pairs/batch':>20}{'no pair':>10}")
    print(header)
    print("-" * len(header))
    for name, cls in samplers.items():
        sampler = cls(data, args.batch_size, args.num_instances)
        nb, c004, pairs = [], [], []
        for _ in range(args.epochs):
            n, c, p = batch_stats(list(iter(sampler)), data,
                                  args.batch_size, args.num_instances)
            nb.append(n)
            c004.append(c)
            pairs.append(p)
        c004 = np.concatenate(c004)
        pairs = np.concatenate(pairs)
        print(f"{name:<22}{np.mean(nb):>11.0f}{c004.mean():>12.2f}"
              f"{(c004 == 0).mean():>10.1%}{pairs.mean():>20.2f}"
              f"{(pairs == 0).mean():>10.1%}")


if __name__ == "__main__":
    main()
