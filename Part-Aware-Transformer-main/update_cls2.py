import os
import csv
import torch
import types
import argparse
import numpy as np

from config import cfg
from model import make_model
from utils.logger import setup_logger
from data.build_DG_dataloader import build_reid_test_loader


def patch_model_cls2(model):
    """
    Patch inference to concatenate CLS tokens from last 2 ViT layers.
    Aymen (0.13553) validated N=2 is the sweet spot.
    Output: 1536-dim instead of 768-dim.
    """
    def new_forward(self, x):
        layerwise_tokens = self.base(x)            # list of 12 tensors [B, 132, 768]
        cls_11 = layerwise_tokens[-2][:, 0]        # 2nd-to-last layer CLS [B, 768]
        cls_12 = layerwise_tokens[-1][:, 0]        # last layer CLS        [B, 768]
        feat = torch.cat([cls_11, cls_12], dim=1)  # [B, 1536]
        return feat

    model.forward = types.MethodType(new_forward, model)
    print("Model patched: last 2 CLS tokens concatenated -> 1536-dim")
    return model


def extract_feature(model, dataloaders, num_query):
    features = []
    model.eval()
    with torch.no_grad():
        for data in dataloaders:
            img, a, b, _, _ = data.values()
            img = img.cuda()
            ff = model(img).float()                           # [B, 1536]
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            features.append(ff.cpu())

    features = torch.cat(features, 0)
    qf = features[:num_query]
    gf = features[num_query:]
    return qf, gf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLS2 Concat Inference")
    parser.add_argument("--config_file", default="./config/PAT.yml", type=str)
    parser.add_argument("--track", default="./submissions/exp11_submission.txt", type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("PAT", output_dir, if_train=False)
    logger.info("exp11: CLS2 concat — last 2 ViT layer CLS tokens, 1536-dim, no re-ranking")

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
    model.load_param(cfg.TEST.WEIGHT)
    model = patch_model_cls2(model)
    model.cuda()
    model.eval()

    for testname in cfg.DATASETS.TEST:
        val_loader, num_query = build_reid_test_loader(cfg, testname)

    with torch.no_grad():
        qf, gf = extract_feature(model, val_loader, num_query)

    print(f"Query features: {qf.shape}  Gallery features: {gf.shape}")
    # Expected: torch.Size([928, 1536]) and torch.Size([2844, 1536])

    # Save features
    np.save("./qf_cls2.npy", qf.numpy())
    np.save("./gf_cls2.npy", gf.numpy())

    qf_np = qf.numpy()
    gf_np = gf.numpy()

    # Pure cosine similarity — features already L2-normalised
    # dot product of normalised vectors = cosine similarity
    # negate because argsort is ascending (we want most similar first)
    q_g_dist = np.dot(qf_np, np.transpose(gf_np))
    indices = np.argsort(-q_g_dist, axis=1)[:, :100]

    # Write .txt
    with open(args.track, 'wb') as f_w:
        for i in range(len(indices)):
            line = ' '.join(map(str, (indices[i] + 1).tolist())) + '\n'
            f_w.write(line.encode())

    # Write submission CSV
    output_path = args.track.replace(".txt", "_submission.csv")
    lista_nombres = ["{:06d}.jpg".format(i) for i in range(1, len(indices) + 1)]
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['imageName', 'Corresponding Indexes'])
        for nombre, idx in zip(lista_nombres, indices):
            writer.writerow([nombre, ' '.join(map(str, idx + 1))])

    print(f"Submission saved to: {output_path}")
