import torch
import torch.nn.functional as F
import numpy as np
import csv
import os
import sys
import argparse
sys.path.insert(0, '.')
from config import cfg
from model import make_model
from data.build_DG_dataloader import build_reid_test_loader
from utils.re_ranking import re_ranking

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, required=True)
parser.add_argument('--weight', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--n_blocks', type=int, default=2)
parser.add_argument('--k1', type=int, default=20)
parser.add_argument('--k2', type=int, default=6)
parser.add_argument('--lam', type=float, default=0.3)
parser.add_argument('--no_rerank', action='store_true')
args = parser.parse_args()

cfg.merge_from_file(args.config_file)
cfg.merge_from_list(['TEST.WEIGHT', args.weight])
cfg.freeze()

os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
model.load_param(args.weight)
model.eval().cuda()

num_blocks = len(model.base.blocks)
print(f'ViT-Large blocks: {num_blocks}')
assert num_blocks == 24, f'Expected 24, got {num_blocks}'

val_loader, num_query = build_reid_test_loader(cfg, 'UrbanElementsReID_test')
print(f'Query: {num_query}, Gallery: {len(val_loader.dataset) - num_query}')

features = []
with torch.no_grad():
    for data in val_loader:
        img = data['images'].cuda()
        B = img.shape[0]
        ff = torch.zeros(B, 1024).cuda()
        for _ in range(2):
            layerwise_tokens = model.base(img)
            cls_list = [layerwise_tokens[-(i+1)][:, 0] for i in range(args.n_blocks-1, -1, -1)]
            avg = torch.stack(cls_list, dim=0).mean(dim=0)
            ff = ff + avg
        ff = F.normalize(ff, p=2, dim=1)
        features.append(ff.cpu())

features = torch.cat(features, 0)
print(f'Feature shape: {features.shape}')

qf = features[:num_query].numpy()
gf = features[num_query:].numpy()
print(f'qf: {qf.shape}, gf: {gf.shape}')

np.save('qf_cls_concat.npy', qf)
np.save('gf_cls_concat.npy', gf)

if args.no_rerank:
    print('Cosine distance only')
    dist = 1.0 - np.dot(qf, gf.T)
else:
    print(f'Re-ranking k1={args.k1}, k2={args.k2}, lam={args.lam}')
    q_g = np.dot(qf, gf.T)
    q_q = np.dot(qf, qf.T)
    g_g = np.dot(gf, gf.T)
    dist = re_ranking(q_g, q_q, g_g, k1=args.k1, k2=args.k2, lambda_value=args.lam)

indices = np.argsort(dist, axis=1)[:, :100]
names = ['{:06d}.jpg'.format(i) for i in range(1, len(indices)+1)]
with open(args.output, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['imageName', 'Corresponding Indexes'])
    for name, row in zip(names, indices):
        w.writerow([name, ' '.join(map(str, row+1))])
print(f'Saved: {args.output}')