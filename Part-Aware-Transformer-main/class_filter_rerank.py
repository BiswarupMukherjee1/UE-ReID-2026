import numpy as np
import csv
import argparse
from collections import Counter
from utils.re_ranking import re_ranking

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--merge_bins', action='store_true')
parser.add_argument('--qf', type=str, default='qf.npy')
parser.add_argument('--gf', type=str, default='gf.npy')
args = parser.parse_args()

QUERY_CLASSES = '/media/DiscoLocal/IPCV/UE-ReID/urban-elements-re-id-challenge-2026/Urban2026/query_classes.csv'
TEST_CLASSES  = '/media/DiscoLocal/IPCV/UE-ReID/urban-elements-re-id-challenge-2026/Urban2026/test_classes.csv'

def load_classes(csv_path, merge_bins):
    classes = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            c = row['Class'].strip().lower()
            if merge_bins and c in ('rubbishbins', 'container'):
                c = 'bin_container'
            classes.append(c)
    return classes

qf = np.load(args.qf)
gf = np.load(args.gf)
print(f'qf: {qf.shape}, gf: {gf.shape}')

query_classes   = load_classes(QUERY_CLASSES, args.merge_bins)
gallery_classes = load_classes(TEST_CLASSES,  args.merge_bins)

assert len(query_classes)   == qf.shape[0], f'Query mismatch: {len(query_classes)} vs {qf.shape[0]}'
assert len(gallery_classes) == gf.shape[0], f'Gallery mismatch: {len(gallery_classes)} vs {gf.shape[0]}'

print('Query class dist:',   Counter(query_classes))
print('Gallery class dist:', Counter(gallery_classes))

q_g = np.dot(qf, gf.T)
q_q = np.dot(qf, qf.T)
g_g = np.dot(gf, gf.T)
dist = re_ranking(q_g, q_q, g_g, k1=20, k2=6, lambda_value=0.3)
print(f'Distance matrix: {dist.shape}')

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