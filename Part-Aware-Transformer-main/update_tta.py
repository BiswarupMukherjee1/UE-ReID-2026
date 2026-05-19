import os, csv, torch, argparse
import numpy as np
from config import cfg
from model import make_model
from utils.logger import setup_logger
from utils.re_ranking import re_ranking
from data.build_DG_dataloader import build_reid_test_loader
from processor.part_attention_vit_processor import do_inference as do_inf_pat


def extract_feature(model, dataloaders, num_query):
    """
    Query-only horizontal flip TTA.

    REASONING FOR THIS SPECIFIC CHALLENGE:
    - All query images come from c004 (backward-facing camera).
    - All gallery images come from c001/c002/c003 (forward-facing).
    - Flipping a backward-facing query makes it look more like a
      forward-facing gallery image -> directly targets the viewpoint gap.
    - Flipping gallery images (already forward-facing) makes them look
      backward-facing -> adds noise, no benefit.

    Therefore: average original + flip ONLY for query.
    Gallery: original features only.
    """
    orig_features = []
    flip_features = []
    model.eval()

    for data in dataloaders:
        img, a, b, _, _ = data.values()
        img = img.cuda()
        with torch.no_grad():
            f_orig = model(img).float()
            f_flip = model(torch.flip(img, dims=[3])).float()
        orig_features.append(f_orig)
        flip_features.append(f_flip)

    orig_features = torch.cat(orig_features, 0)
    flip_features = torch.cat(flip_features, 0)

    # Query only: average original + flipped (c004 backward -> flip helps)
    qf = (orig_features[:num_query] + flip_features[:num_query]) / 2
    # Gallery: original only (forward-facing, flip would hurt)
    gf = orig_features[num_query:]

    # L2 normalize
    qf = qf / torch.norm(qf, p=2, dim=1, keepdim=True)
    gf = gf / torch.norm(gf, p=2, dim=1, keepdim=True)

    return qf, gf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="", type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--track", default="", type=str)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("PAT", output_dir, if_train=False)
    logger.info("TTA: query-only horizontal flip (c004 backward-facing queries)")

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
    model.load_param(cfg.TEST.WEIGHT)
    model.eval()

    for testname in cfg.DATASETS.TEST:
        val_loader, num_query = build_reid_test_loader(cfg, testname)
        do_inf_pat(cfg, model, val_loader, num_query)

    with torch.no_grad():
        qf, gf = extract_feature(model, val_loader, num_query)

    qf = qf.cpu().numpy()
    gf = gf.cpu().numpy()
    np.save("./qf.npy", qf)
    np.save("./gf.npy", gf)

    q_g_dist = np.dot(qf, np.transpose(gf))
    re_rank_dist = re_ranking(q_g_dist, np.dot(qf, qf.T), np.dot(gf, gf.T))
    indices = np.argsort(re_rank_dist, axis=1)[:, :100]

    with open(args.track, 'wb') as f:
        for i in range(len(indices)):
            f.write((' '.join(map(str, indices[i] + 1)) + '\n').encode())

    with open(args.track.replace('.txt', '_submission.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['imageName', 'Corresponding Indexes'])
        for i, idx in enumerate(indices):
            w.writerow([f"{i+1:06d}.jpg", ' '.join(map(str, idx + 1))])