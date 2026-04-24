"""
UrbAM Local Evaluation Pipeline

Converts UrbAM XML labels to CSV, runs inference, and evaluates with evaluate_csv.py.

This uses the 'inv' split where query=c004 (back-view), gallery=c001/c002/c003 —
directly mirroring the Kaggle challenge structure.

Usage:
    python evaluate_urbam.py \
        --config_file config/UrbanElementsReID_test.yml \
        --split_dir /media/DiscoLocal/IPCV/UE-ReID/UrbAM-ReID/ICIP_UrbAM-ReID/splits/Containers/containers/inv \
        --k1 15 --k2 8 --lambda_value 0.1
"""

import os
import csv
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from config import cfg
from model import make_model
from utils.re_ranking import re_ranking


# ---------------------------------------------------------------------------
# XML → CSV converter
# ---------------------------------------------------------------------------

def xml_to_csv(xml_path, out_csv_path):
    """Convert UrbAM XML label to CSV format expected by evaluate_csv.py."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rows = []
    for item in root.findall('Items/Item'):
        rows.append({
            'cameraID': item.get('cameraID'),
            'imageName': item.get('imageName'),
            'Corresponding Indexes': int(item.get('objectID')),
        })
    with open(out_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['cameraID', 'imageName', 'Corresponding Indexes'])
        writer.writeheader()
        writer.writerows(rows)
    return rows


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UrbAMDataset(Dataset):
    def __init__(self, image_dir, entries, transform):
        self.image_dir = image_dir
        self.entries = entries  # list of dicts with imageName, objectID
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        img = Image.open(os.path.join(self.image_dir, e['imageName'])).convert('RGB')
        return self.transform(img), int(e['Corresponding Indexes']), e['imageName']


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(model, loader):
    model.eval()
    features, pids, fnames = [], [], []
    with torch.no_grad():
        for imgs, pid, fname in loader:
            imgs = imgs.cuda()
            ff = model(imgs).float()
            ff += model(torch.flip(imgs, [3])).float()
            ff = ff / torch.norm(ff, p=2, dim=1, keepdim=True)
            features.append(ff.cpu().numpy())
            pids.extend(pid.numpy() if hasattr(pid, 'numpy') else list(pid))
            fnames.extend(fname)
    return np.vstack(features), np.array(pids), fnames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='config/UrbanElementsReID_test.yml')
    parser.add_argument('--split_dir',
        default='/media/DiscoLocal/IPCV/UE-ReID/UrbAM-ReID/ICIP_UrbAM-ReID/splits/Containers/containers/inv')
    parser.add_argument('--k1', default=15, type=int)
    parser.add_argument('--k2', default=8, type=int)
    parser.add_argument('--lambda_value', default=0.1, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--out_dir', default='./urbam_eval', type=str)
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # --- Convert XML to CSV ---
    print("Converting XML labels to CSV...")
    query_xml = os.path.join(args.split_dir, 'query_label.xml')
    test_xml  = os.path.join(args.split_dir, 'test_label.xml')
    query_csv = os.path.join(args.out_dir, 'query.csv')
    test_csv  = os.path.join(args.out_dir, 'test.csv')

    query_entries = xml_to_csv(query_xml, query_csv)
    test_entries  = xml_to_csv(test_xml,  test_csv)
    print(f"Query: {len(query_entries)} images, Gallery: {len(test_entries)} images")

    q_cams = set(e['cameraID'] for e in query_entries)
    g_cams = set(e['cameraID'] for e in test_entries)
    print(f"Query cameras: {q_cams}")
    print(f"Gallery cameras: {g_cams}")

    # --- Load model ---
    print(f"Loading model from {cfg.TEST.WEIGHT}")
    model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
    model.load_param(cfg.TEST.WEIGHT)
    model.cuda()
    model.eval()

    size = tuple(cfg.INPUT.SIZE_TEST)
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    query_img_dir = os.path.join(args.split_dir, 'image_query')
    test_img_dir  = os.path.join(args.split_dir, 'image_test')

    query_ds = UrbAMDataset(query_img_dir, query_entries, transform)
    test_ds  = UrbAMDataset(test_img_dir,  test_entries,  transform)

    query_loader = DataLoader(query_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4)

    # --- Extract features ---
    print("Extracting query features...")
    qf, q_pids, q_fnames = extract_features(model, query_loader)
    print("Extracting gallery features...")
    gf, g_pids, g_fnames = extract_features(model, test_loader)
    print(f"qf: {qf.shape}, gf: {gf.shape}")

    # --- Re-ranking ---
    print(f"Re-ranking: k1={args.k1}, k2={args.k2}, lambda={args.lambda_value}")
    q_g = 1.0 - np.dot(qf, gf.T)
    q_q = 1.0 - np.dot(qf, qf.T)
    g_g = 1.0 - np.dot(gf, gf.T)
    dist = re_ranking(q_g, q_q, g_g, k1=args.k1, k2=args.k2, lambda_value=args.lambda_value)

    indices = np.argsort(dist, axis=1)[:, :100]

    # --- Build submission CSV (sorted gallery order) ---
    # evaluate_csv.py sorts gallery by filename numerically
    sorted_g_fnames = sorted(g_fnames, key=lambda x: int(x.split('.')[0]))
    fname_to_idx = {fname: i for i, fname in enumerate(sorted_g_fnames)}

    submission_path = os.path.join(args.out_dir, 'submission.csv')
    with open(submission_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['imageName', 'Corresponding Indexes'])
        for i, q_fname in enumerate(q_fnames):
            # indices[i] are positions in gf — map back to sorted gallery positions
            raw_indices = indices[i]
            # gf was built in XML order; map each to its sorted position + 1-based
            sorted_positions = []
            for idx in raw_indices:
                gfname = g_fnames[idx]
                sorted_pos = fname_to_idx[gfname] + 1  # 1-based
                sorted_positions.append(sorted_pos)
            writer.writerow([q_fname, ' '.join(map(str, sorted_positions))])

    print(f"Submission saved to {submission_path}")

    # --- Run official evaluator ---
    print("\nRunning official evaluate_csv.py...")
    eval_cmd = (
        f"python evaluate_csv.py "
        f"--path {args.out_dir} "
        f"--track {submission_path}"
    )
    print(f"CMD: {eval_cmd}")
    os.system(eval_cmd)
