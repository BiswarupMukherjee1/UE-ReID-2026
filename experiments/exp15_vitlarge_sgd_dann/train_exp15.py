"""
Train script for exp15_vitlarge_sgd_dann
DANN camera-adversarial training for Urban Elements ReID 2026.

Run from Part-Aware-Transformer-main directory:
    python /path/to/exp15_dann/train_exp15.py \
        --config_file /path/to/exp15_dann/config/train.yml

Key additions vs train.py:
1. Imports processor_dann instead of part_attention_vit_processor
2. Creates camera_classifier as separate module
3. Adds camera_classifier parameters to existing optimizer
4. Passes camera_classifier + lambda_dann to training function

NO existing files are modified.
"""

import sys
import os

# Add experiment directory to path so we can import dann_components and processor_dann
EXP15_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP15_DIR)

# Add PAT main directory to path
PAT_DIR = '/media/DiscoLocal/IPCV/UE-ReID/Part-Aware-Transformer-main'
sys.path.insert(0, PAT_DIR)
os.chdir(PAT_DIR)  # Important: train.py expects to be run from PAT_DIR

from processor_dann import part_attention_vit_do_train_with_amp_dann
from dann_components import CameraClassifier

from utils.logger import setup_logger
from data.build_DG_dataloader import build_reid_train_loader, build_reid_test_loader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss.build_loss import build_loss

import random
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import argparse
from config import cfg
import loss as Patchloss

# Lambda for DANN camera adversarial loss
# Aymen used 0.1 (got 0.16), recommends 0.2-0.3 for more invariance
LAMBDA_DANN = 0.2

# Number of cameras: c001=1, c002=2, c003=3, c004=4
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
    parser = argparse.ArgumentParser(description="ReID DANN Training - exp15")
    parser.add_argument(
        "--config_file",
        default="",
        help="path to config file",
        type=str)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("PAT", output_dir, if_train=True)
    logger.info("Saving model in the path: {}".format(output_dir))
    logger.info("DANN lambda: {}".format(LAMBDA_DANN))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # Build data loaders
    train_loader = build_reid_train_loader(cfg)
    val_name = cfg.DATASETS.TEST[0]
    val_loader, num_query = build_reid_test_loader(cfg, val_name)
    num_classes = len(train_loader.dataset.pids)

    print("num_classes: {}".format(num_classes))

    # Build model (unchanged — standard build_part_attention_vit)
    model_name = cfg.MODEL.NAME
    model = make_model(
        cfg,
        modelname=model_name,
        num_class=num_classes,
        camera_num=None,
        view_num=None)

    if cfg.MODEL.FREEZE_PATCH_EMBED and 'resnet' not in cfg.MODEL.NAME:
        model.base.patch_embed.proj.weight.requires_grad = False
        model.base.patch_embed.proj.bias.requires_grad = False
        print("====== freeze patch_embed for stability ======")

    # Build loss
    loss_func, center_cri = build_loss(cfg, num_classes=num_classes)

    # Build camera classifier (separate from main model)
    camera_classifier = CameraClassifier(
        in_dim=1024,
        num_cameras=NUM_CAMERAS)
    camera_classifier = camera_classifier.cuda()
    print("Camera classifier created: 1024 -> 256 -> {}".format(NUM_CAMERAS))

    # Build optimizer — includes BOTH model and camera_classifier parameters
    optimizer = make_optimizer(cfg, model)

    # Add camera_classifier parameters to the existing optimizer
    optimizer.add_param_group({
        'params': camera_classifier.parameters(),
        'lr': cfg.SOLVER.BASE_LR,
        'weight_decay': cfg.SOLVER.WEIGHT_DECAY,
    })
    print("Added camera_classifier params to optimizer")

    # Build scheduler
    scheduler = create_scheduler(cfg, optimizer)

    # Patch loss (required by processor even when PC_LOSS=False)
    patch_centers = Patchloss.PatchMemory(momentum=0.1, num=1)
    pc_criterion = Patchloss.Pedal(
        scale=cfg.MODEL.PC_SCALE, k=cfg.MODEL.CLUSTER_K).cuda()

    if cfg.MODEL.SOFT_LABEL and cfg.MODEL.NAME == 'part_attention_vit':
        print("========using soft label========")

    # Train with DANN
    if model_name == 'part_attention_vit':
        part_attention_vit_do_train_with_amp_dann(
            cfg,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            loss_func,
            num_query,
            args.local_rank,
            patch_centers=patch_centers,
            pc_criterion=pc_criterion,
            camera_classifier=camera_classifier,
            lambda_dann=LAMBDA_DANN,
        )
    else:
        raise ValueError(
            "exp15 only supports part_attention_vit, got: {}".format(model_name))