import os
import csv
import torch
import argparse
import numpy as np
from config import cfg
from model import make_model
from utils.logger import setup_logger
from utils.re_ranking import re_ranking
from data.build_DG_dataloader import build_reid_test_loader


def extract_feature(model, dataloaders, num_query):
    features = []

    for data in dataloaders:
        img, a, b, _, _ = data.values()
        n, c, h, w = img.size()
        ff = torch.FloatTensor(n, 768).zero_().cuda()

        # Pass 1: original
        input_img = img.cuda()
        outputs = model(input_img)
        ff += outputs.float()

        # Pass 2: horizontal flip (TTA fix — was broken in update.py)
        input_img_flip = torch.flip(img.cuda(), [3])
        outputs_flip = model(input_img_flip)
        ff += outputs_flip.float()

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features.append(ff)

    features = torch.cat(features, 0)
    qf = features[:num_query]
    gf = features[num_query:]
    return qf, gf


def load_camera_ids(csv_path):
    """Return ordered list of camera IDs from a CSV with 'cameraID' column."""
    cam_ids = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cam_ids.append(row['cameraID'])
    return cam_ids


def compute_caj_weight(q_cams, g_cams, sigma=50.0):
    """
    Camera-Aware Jaccard distance penalty matrix.

    For each (query_i, gallery_j) pair, we compute a weight based on
    how often their camera pair co-occurs across all pairs.

    Rare pairs (like c004 vs gallery) get DOWN-weighted distances
    (i.e. we trust those matches more, since same-ID images will be rare).

    Args:
        q_cams: list of camera IDs for query images
        g_cams: list of camera IDs for gallery images
        sigma:  softness of the penalty (higher = smoother)

    Returns:
        weight matrix of shape (num_query, num_gallery), values in (0, 1]
        Multiply distance matrix by this to apply penalty.
    """
    q_cams = np.array(q_cams)
    g_cams = np.array(g_cams)

    # Count co-occurrence frequency for each camera pair
    unique_q = np.unique(q_cams)
    unique_g = np.unique(g_cams)

    pair_count = {}
    for qc in unique_q:
        for gc in unique_g:
            n_q = np.sum(q_cams == qc)
            n_g = np.sum(g_cams == gc)
            pair_count[(qc, gc)] = n_q * n_g  # proportional to pair frequency

    total = sum(pair_count.values())

    # Build weight matrix: rare pairs get lower weight (boost those matches)
    nq = len(q_cams)
    ng = len(g_cams)
    weight = np.ones((nq, ng), dtype=np.float32)

    for i in range(nq):
        for j in range(ng):
            freq = pair_count.get((q_cams[i], g_cams[j]), 0)
            # Normalised frequency [0, 1]
            norm_freq = freq / (total + 1e-8)
            # Gaussian penalty: common pairs get penalised, rare pairs get boosted
            # weight < 1 means "reduce this distance" (trust it more)
            weight[i, j] = 1.0 - np.exp(-norm_freq * sigma)

    # Clip to avoid collapsing distances to zero
    weight = np.clip(weight, 0.05, 1.0)
    return weight


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Inference with CAJ re-ranking")
    parser.add_argument("--config_file", default="./config/UrbanElementsReID_test.yml", type=str)
    parser.add_argument("--track", default="./submission_caj.txt", type=str)
    parser.add_argument("--query_csv", default="/media/DiscoLocal/IPCV/UE-ReID/urban-elements-re-id-challenge-2026/Urban2026/query.csv", type=str)
    parser.add_argument("--gallery_csv", default="/media/DiscoLocal/IPCV/UE-ReID/urban-elements-re-id-challenge-2026/Urban2026/test.csv", type=str)
    # Re-ranking params (tunable)
    parser.add_argument("--k1", default=20, type=int, help="k-reciprocal neighbours")
    parser.add_argument("--k2", default=6, type=int, help="k-nearest neighbours for query expansion")
    parser.add_argument("--lambda_value", default=0.3, type=float, help="re-ranking lambda")
    # CAJ
    parser.add_argument("--caj_sigma", default=50.0, type=float, help="CAJ penalty softness")
    parser.add_argument("--use_caj", action="store_true", default=True, help="Apply CAJ weighting")
    parser.add_argument("--save_features", action="store_true", default=True)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger("PAT_CAJ", output_dir, if_train=False)
    logger.info(f"Config: {args.config_file}")
    logger.info(f"Checkpoint: {cfg.TEST.WEIGHT}")
    logger.info(f"Re-ranking params: k1={args.k1}, k2={args.k2}, lambda={args.lambda_value}")
    logger.info(f"CAJ: enabled={args.use_caj}, sigma={args.caj_sigma}")

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
    model.load_param(cfg.TEST.WEIGHT)
    model.cuda()
    model.eval()

    for testname in cfg.DATASETS.TEST:
        val_loader, num_query = build_reid_test_loader(cfg, testname)

    logger.info("Extracting features with TTA (flip)...")
    with torch.no_grad():
        qf, gf = extract_feature(model, val_loader, num_query)

    qf = qf.cpu().numpy()
    gf = gf.cpu().numpy()

    if args.save_features:
        np.save("./qf_caj.npy", qf)
        np.save("./gf_caj.npy", gf)
        logger.info("Saved qf_caj.npy and gf_caj.npy")

    logger.info(f"Query features: {qf.shape}, Gallery features: {gf.shape}")

    # --- Distance matrices (cosine, converted to distance) ---
    q_g_dist = 1.0 - np.dot(qf, gf.T)   # shape (nq, ng)
    q_q_dist = 1.0 - np.dot(qf, qf.T)
    g_g_dist = 1.0 - np.dot(gf, gf.T)

    # --- Camera-Aware Jaccard weighting ---
    if args.use_caj:
        logger.info("Loading camera IDs...")
        q_cams = load_camera_ids(args.query_csv)
        g_cams = load_camera_ids(args.gallery_csv)
        logger.info(f"Query cameras: {set(q_cams)}")
        logger.info(f"Gallery cameras: {set(g_cams)}")

        logger.info("Computing CAJ weight matrix...")
        caj_weight = compute_caj_weight(q_cams, g_cams, sigma=args.caj_sigma)
        logger.info(f"CAJ weight range: [{caj_weight.min():.4f}, {caj_weight.max():.4f}]")

        # Apply: reduce distance for rare camera pairs (c004 vs gallery)
        q_g_dist = q_g_dist * caj_weight
        logger.info("CAJ applied to q_g_dist")

    # --- k-reciprocal re-ranking ---
    logger.info(f"Running re-ranking (k1={args.k1}, k2={args.k2}, lambda={args.lambda_value})...")
    re_rank_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist,
                               k1=args.k1, k2=args.k2,
                               lambda_value=args.lambda_value)

    indices = np.argsort(re_rank_dist, axis=1)[:, :100]
    m, n = indices.shape
    logger.info(f"Ranking done: {m} queries, top {n} results each")

    # --- Write .txt track file ---
    with open(args.track, 'wb') as f_w:
        for i in range(m):
            write_line = indices[i] + 1
            write_line = ' '.join(map(str, write_line.tolist())) + '\n'
            f_w.write(write_line.encode())

    # --- Write submission CSV ---
    lista_nombres = ["{:06d}.jpg".format(i) for i in range(1, m + 1)]
    output_path = args.track.replace(".txt", "_submission.csv")

    with open(output_path, 'w', newline='') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(['imageName', 'Corresponding Indexes'])
        for nombre, track in zip(lista_nombres, indices):
            track_str = ' '.join(map(str, track + 1))
            writer.writerow([nombre, track_str])

    logger.info(f"Submission saved to: {output_path}")
    logger.info("Done.")
