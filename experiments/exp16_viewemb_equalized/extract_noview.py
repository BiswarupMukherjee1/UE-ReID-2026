"""
extract_noview.py — feature extraction with view_id=0 for ALL images.

Based on finding that asymmetric view embeddings (backward query, forward gallery)
HURT performance (0.13145 vs 0.13555 noview). This script applies view_id=0
to both query and gallery, giving the model's forward-view features consistently.

Usage:
    cd /media/DiscoLocal/IPCV/UE-ReID/Part-Aware-Transformer-main

    python /media/DiscoLocal/IPCV/UE-ReID/experiments/exp16_viewemb_equalized/extract_noview.py \
        --ep 35
    python /media/DiscoLocal/IPCV/UE-ReID/experiments/exp16_viewemb_equalized/extract_noview.py \
        --ep 45
    python /media/DiscoLocal/IPCV/UE-ReID/experiments/exp16_viewemb_equalized/extract_noview.py \
        --ep 50

    # Then CAJ for each:
    python caj_filter_rerank.py --output submission_ep35_noview.csv \
        --merge_bins --k1 15 --k2 4 \
        --qf qf_ep35_noview.npy --gf gf_ep35_noview.npy
"""

import os, sys, csv, argparse
import torch, numpy as np
from PIL import Image
import torchvision.transforms as T

PAT_DIR  = '/media/DiscoLocal/IPCV/UE-ReID/Part-Aware-Transformer-main'
EXP_DIR  = '/media/DiscoLocal/IPCV/UE-ReID/experiments/exp16_viewemb_equalized'
DATA     = '/media/DiscoLocal/IPCV/UE-ReID/urban-elements-re-id-challenge-2026/Urban2026'

sys.path.insert(0, PAT_DIR)
os.chdir(PAT_DIR)

from config import cfg
from model import make_model

def get_items(csv_path, img_dir):
    items = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            items.append(os.path.join(img_dir, row['imageName']))
    return items

def extract(model, paths, view_id_val, batch_size=64):
    tf = T.Compose([
        T.Resize((256, 128), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])
    model.eval()
    feats = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i+batch_size]
        imgs = torch.stack([tf(Image.open(p).convert('RGB')) for p in batch]).cuda()
        view_ids = torch.full((len(batch),), view_id_val, dtype=torch.long).cuda()
        with torch.no_grad():
            f = model(imgs, view_ids=view_ids).float()
        f = f / f.norm(dim=1, keepdim=True)
        feats.append(f.cpu())
        if (i // batch_size) % 5 == 0:
            print(f"  {i+len(batch)}/{len(paths)}")
    return torch.cat(feats).numpy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ep', required=True, type=int, help='Epoch number, e.g. 35')
    args = parser.parse_args()

    weight = f'models/./exp16_viewemb_equalized/part_attention_vit_{args.ep}.pth'
    qf_out = f'qf_ep{args.ep}_noview.npy'
    gf_out = f'gf_ep{args.ep}_noview.npy'

    cfg.merge_from_file(os.path.join(EXP_DIR, 'config/test.yml'))
    cfg.merge_from_list(['TEST.WEIGHT', weight])
    cfg.freeze()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    model = make_model(cfg, 'part_attention_vit', 0, 0, 0)
    model.load_param(weight)
    model = model.cuda()
    print(f'Loaded ep{args.ep}: norm_fwd={model.base.view_embedding[0].norm():.4f}')

    query_paths   = get_items(f'{DATA}/query.csv', f'{DATA}/image_query')
    gallery_paths = get_items(f'{DATA}/test.csv',  f'{DATA}/image_test')

    assert len(query_paths)   == 928,  f'Expected 928, got {len(query_paths)}'
    assert len(gallery_paths) == 2844, f'Expected 2844, got {len(gallery_paths)}'

    print(f'Query ({len(query_paths)}), view_id=0 (no backward embed):')
    qf = extract(model, query_paths, view_id_val=0)

    print(f'Gallery ({len(gallery_paths)}), view_id=0:')
    gf = extract(model, gallery_paths, view_id_val=0)

    np.save(qf_out, qf)
    np.save(gf_out, gf)
    print(f'Saved: {qf_out} {qf.shape}')
    print(f'Saved: {gf_out} {gf.shape}')
    print()
    print('Next:')
    print(f'python caj_filter_rerank.py --output submission_ep{args.ep}_noview.csv \\')
    print(f'    --merge_bins --k1 15 --k2 4 \\')
    print(f'    --qf {qf_out} --gf {gf_out}')