import numpy as np
import csv
import argparse
from argparse import Namespace
from collections import Counter
from caj_rerank_ref import compute_jaccard_distance

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--merge_bins', action='store_true')
parser.add_argument('--k1', type=int, default=20)
parser.add_argument('--k2', type=int, default=6)
parser.add_argument('--k1_intra', type=int, default=20)
parser.add_argument('--k1_inter', type=int, default=20)
parser.add_argument('--k2_intra', type=int, default=3)
parser.add_argument('--k2_inter', type=int, default=3)
parser.add_argument('--lam', type=float, default=0.0,
                    help='Lambda for mixing CAJ with cosine dist. 0=pure CAJ (default)')
parser.add_argument('--qf', type=str, default='qf.npy')
parser.add_argument('--gf', type=str, default='gf.npy')
args = parser.parse_args()

QUERY_CSV   = '/media/DiscoLocal/IPCV/UE-ReID/urban-elements-re-id-challenge-2026/Urban2026/query.csv'
TEST_CSV    = '/media/DiscoLocal/IPCV/UE-ReID/urban-elements-re-id-challenge-2026/Urban2026/test.csv'
QUERY_CLASS = '/media/DiscoLocal/IPCV/UE-ReID/urban-elements-re-id-challenge-2026/Urban2026/query_classes.csv'
TEST_CLASS  = '/media/DiscoLocal/IPCV/UE-ReID/urban-elements-re-id-challenge-2026/Urban2026/test_classes.csv'

qf = np.load(args.qf)
gf = np.load(args.gf)
num_query   = qf.shape[0]
num_gallery = gf.shape[0]
print(f'qf: {qf.shape}, gf: {gf.shape}')

def load_camids(csv_path):
    cams = []
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            cams.append(int(row['cameraID'][1:]))
    return np.array(cams)

q_cams = load_camids(QUERY_CSV)
g_cams = load_camids(TEST_CSV)
print(f'Query cams: {Counter(q_cams.tolist())}')
print(f'Gallery cams: {Counter(g_cams.tolist())}')

assert len(q_cams) == num_query,   f'Mismatch: {len(q_cams)} vs {num_query}'
assert len(g_cams) == num_gallery, f'Mismatch: {len(g_cams)} vs {num_gallery}'

def load_classes(csv_path, merge_bins):
    classes = []
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            c = row['Class'].strip().lower()
            if merge_bins and c in ('rubbishbins', 'container'):
                c = 'bin_container'
            classes.append(c)
    return classes

query_classes   = load_classes(QUERY_CLASS, args.merge_bins)
gallery_classes = load_classes(TEST_CLASS,  args.merge_bins)
print(f'Query class dist:   {Counter(query_classes)}')
print(f'Gallery class dist: {Counter(gallery_classes)}')

assert len(query_classes)   == num_query,   f'Mismatch: {len(query_classes)} vs {num_query}'
assert len(gallery_classes) == num_gallery, f'Mismatch: {len(gallery_classes)} vs {num_gallery}'

features   = np.concatenate([qf, gf], axis=0)
cam_labels = np.concatenate([q_cams, g_cams])
print(f'Combined features: {features.shape}')
print(f'Combined cam_labels: {cam_labels.shape}')

caj_args = Namespace(
    k1=args.k1,
    k2=args.k2,
    ckrnns=True,
    k1_intra=args.k1_intra,
    k1_inter=args.k1_inter,
    clqe=False,
    k2_intra=args.k2_intra,
    k2_inter=args.k2_inter,
)

jaccard_dist = compute_jaccard_distance(
    features=features,
    cam_labels=cam_labels,
    epoch=1,
    args=caj_args,
)
print(f'Full dist matrix: {jaccard_dist.shape}')

dist = jaccard_dist[:num_query, num_query:]
print(f'Q-G dist: {dist.shape}')

if args.lam > 0.0:
    cosine_dist = 1.0 - np.dot(qf, gf.T)
    dist = dist * (1 - args.lam) + cosine_dist * args.lam
    print(f'Applied lambda={args.lam} mixing with cosine distance')

query_arr   = np.array(query_classes)
gallery_arr = np.array(gallery_classes)
mask = query_arr[:, None] != gallery_arr[None, :]
dist[mask] = 1e6

indices = np.argsort(dist, axis=1)[:, :100]
names = ['{:06d}.jpg'.format(i) for i in range(1, len(indices)+1)]
with open(args.output, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['imageName', 'Corresponding Indexes'])
    for name, row in zip(names, indices):
        w.writerow([name, ' '.join(map(str, row+1))])
print(f'Saved: {args.output}')