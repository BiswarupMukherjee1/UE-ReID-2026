"""
Train script for exp20_control_random.
DANN + Standard RandomIdentitySampler (NO CameraEqualizedSampler).

PURPOSE: Ablation control for exp17.
Identical to exp17 in EVERY way EXCEPT the sampler.
exp17 used CameraEqualizedSampler. This uses default RandomIdentitySampler.
Everything else: 19.9K data, lambda=0.3, 60 epochs, same SGD, same DANN.

This isolates the sampler contribution:
  exp20 (random, 19.9K, lam=0.3) vs exp17 (equalized, 19.9K, lam=0.3)
"""

import sys
import os

EXP20_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP20_DIR)

PAT_DIR = '/media/DiscoLocal/IPCV/UE-ReID/Part-Aware-Transformer-main'
sys.path.insert(0, PAT_DIR)
os.chdir(PAT_DIR)

from processor_exp20 import part_attention_vit_do_train_with_amp_exp17
from dann_components import CameraClassifier

from utils.logger import setup_logger
from data.build_DG_dataloader import build_reid_train_loader, build_reid_test_loader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss.build_loss import build_loss

import random
import torch
import torch.utils.data
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import argparse
from config import cfg
import loss as Patchloss

LAMBDA_DANN = 0.3
NUM_CAMERAS = 4


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID exp20 DANN + Random Sampling (control)")
    parser.add_argument("--config_file", default="", type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    output_dir = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("PAT", output_dir, if_train=True)
    logger.info("Experiment: exp20_control_random")
    logger.info("SAMPLER: RandomIdentitySampler (default — NO CameraEqualizedSampler)")
    logger.info("LAMBDA_DANN: {}".format(LAMBDA_DANN))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded config: {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            logger.info("\n" + cf.read())
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # ── KEY DIFFERENCE FROM EXP17: use default RandomIdentitySampler ──
    train_loader = build_reid_train_loader(cfg)
    train_set    = train_loader.dataset

    # Dataset sanity checks — abort immediately if wrong data loaded
    c004_total = sum(1 for item in train_set.img_items if item[2] == 4)
    all_total  = len(train_set.img_items)
    c004_pids  = len(set(item[1] for item in train_set.img_items if item[2] == 4))
    total_pids = len(set(item[1] for item in train_set.img_items))

    logger.info("Dataset: {:,} images, {:,} c004 ({:.1f}%)".format(
        all_total, c004_total, 100.0 * c004_total / all_total))
    logger.info("Identities: {:,} total, {:,} c004-capable ({:.1f}%)".format(
        total_pids, c004_pids, 100.0 * c004_pids / total_pids))
    logger.info("Train loader: {} batches/epoch".format(len(train_loader)))

    assert all_total > 19000, "ABORT: Expected ~19870 images, got {}. Wrong dataset.".format(all_total)
    assert c004_pids > 300,   "ABORT: Expected ~322 c004-capable pids, got {}.".format(c004_pids)
    logger.info("Dataset sanity check PASSED.")

    val_name              = cfg.DATASETS.TEST[0]
    val_loader, num_query = build_reid_test_loader(cfg, val_name)
    num_classes           = len(train_set.pids)
    logger.info("num_classes: {}".format(num_classes))

    model_name = cfg.MODEL.NAME
    model = make_model(cfg, modelname=model_name,
                       num_class=num_classes,
                       camera_num=None, view_num=None)

    if cfg.MODEL.FREEZE_PATCH_EMBED and 'resnet' not in model_name:
        model.base.patch_embed.proj.weight.requires_grad = False
        model.base.patch_embed.proj.bias.requires_grad   = False
        logger.info("Frozen patch_embed for stability")

    camera_classifier = CameraClassifier(in_dim=1024, num_cameras=NUM_CAMERAS)
    camera_classifier = camera_classifier.cuda()
    logger.info("Camera classifier: 1024->512->256->{}".format(NUM_CAMERAS))

    loss_func, center_cri = build_loss(cfg, num_classes=num_classes)
    optimizer = make_optimizer(cfg, model)
    optimizer.add_param_group({
        'params':       camera_classifier.parameters(),
        'lr':           cfg.SOLVER.BASE_LR,
        'weight_decay': cfg.SOLVER.WEIGHT_DECAY,
    })
    logger.info("Added camera_classifier to optimizer")

    scheduler = create_scheduler(cfg, optimizer)

    patch_centers = Patchloss.PatchMemory(momentum=0.1, num=1)
    pc_criterion  = Patchloss.Pedal(scale=cfg.MODEL.PC_SCALE,
                                    k=cfg.MODEL.CLUSTER_K).cuda()

    if cfg.MODEL.SOFT_LABEL and model_name == 'part_attention_vit':
        logger.info("Using soft label")

    if model_name == 'part_attention_vit':
        part_attention_vit_do_train_with_amp_exp17(
            cfg, model, train_loader, val_loader,
            optimizer, scheduler, loss_func, num_query,
            args.local_rank,
            patch_centers     = patch_centers,
            pc_criterion      = pc_criterion,
            camera_classifier = camera_classifier,
            lambda_dann       = LAMBDA_DANN,
        )
    else:
        raise ValueError("exp20 only supports part_attention_vit, got: {}".format(model_name))
