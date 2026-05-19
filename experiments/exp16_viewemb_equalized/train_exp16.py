"""
Train script for exp16 — view embeddings + camera equalization.

Key differences vs train.py:
1. Uses CameraEqualizedSampler instead of RandomIdentitySampler
2. Uses processor_exp16 (passes view_ids to model)
3. No camera_classifier / DANN
4. UAM_Unified.py now includes query+test splits

Run from Part-Aware-Transformer-main directory:
    python /path/to/exp16/train_exp16.py \
        --config_file /path/to/exp16/config/train.yml

NO shared files are modified except:
  - data/datasets/UAM_Unified.py  (replaced with exp16 version)
  - model/backbones/vit_pytorch.py  (view_embedding added)
  - model/make_model.py  (forward passes view_ids)
"""

import sys
import os

EXP16_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP16_DIR)

PAT_DIR = '/media/DiscoLocal/IPCV/UE-ReID/Part-Aware-Transformer-main'
sys.path.insert(0, PAT_DIR)
os.chdir(PAT_DIR)

from processor_exp16 import part_attention_vit_do_train_with_amp_exp16
from sampler_exp16 import CameraEqualizedSampler

from utils.logger import setup_logger
from data.build_DG_dataloader import build_reid_test_loader
from data.build_DG_dataloader import fast_batch_collator
from data.build_DG_dataloader import build_reid_train_loader
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


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID exp16 training")
    parser.add_argument("--config_file", default="", type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
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
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded config: {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            logger.info("\n" + cf.read())
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # ------------------------------------------------------------------ #
    # Build train loader — then REPLACE sampler with CameraEqualizedSampler
    # ------------------------------------------------------------------ #
    # First build the default loader to get the properly-constructed CommDataset
    # (which includes domain info added by build_DG_dataloader)
    train_loader_default = build_reid_train_loader(cfg)
    train_set            = train_loader_default.dataset  # CommDataset

    # Verify c004 count in the dataset
    c004_total = sum(1 for item in train_set.img_items if item[2] == 4)
    all_total  = len(train_set.img_items)
    logger.info(f"Dataset: {all_total} images, {c004_total} c004 ({100*c004_total/all_total:.1f}%)")

    # Create CameraEqualizedSampler
    sampler = CameraEqualizedSampler(
        data_source   = train_set.img_items,
        batch_size    = cfg.SOLVER.IMS_PER_BATCH,
        num_instances = cfg.DATALOADER.NUM_INSTANCE,
    )
    logger.info(f"Using CameraEqualizedSampler (guarantee 1 c004 per c004-capable identity group)")

    # Build new DataLoader with custom sampler
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, cfg.SOLVER.IMS_PER_BATCH, drop_last=True)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers  = cfg.DATALOADER.NUM_WORKERS,
        batch_sampler= batch_sampler,
        collate_fn   = fast_batch_collator,
    )
    logger.info(f"Train loader: {len(train_loader)} batches/epoch")

    # ------------------------------------------------------------------ #
    # Validation loader
    # ------------------------------------------------------------------ #
    val_name = cfg.DATASETS.TEST[0]
    val_loader, num_query = build_reid_test_loader(cfg, val_name)
    num_classes = len(train_set.pids)
    logger.info(f"num_classes: {num_classes}")

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    model_name = cfg.MODEL.NAME
    model = make_model(cfg, modelname=model_name, num_class=num_classes,
                       camera_num=None, view_num=None)

    if cfg.MODEL.FREEZE_PATCH_EMBED and 'resnet' not in model_name:
        model.base.patch_embed.proj.weight.requires_grad = False
        model.base.patch_embed.proj.bias.requires_grad   = False
        print("====== freeze patch_embed for stability ======")

    # Confirm view_embedding is in model
    if hasattr(model.base, 'view_embedding'):
        logger.info(f"view_embedding shape: {model.base.view_embedding.shape}, "
                    f"lambda_view: {model.base.lambda_view}")
    else:
        raise RuntimeError(
            "model.base.view_embedding not found. "
            "Did you apply the vit_pytorch.py patch?")

    # ------------------------------------------------------------------ #
    # Loss, optimizer, scheduler
    # ------------------------------------------------------------------ #
    loss_func, center_cri = build_loss(cfg, num_classes=num_classes)
    optimizer = make_optimizer(cfg, model)
    scheduler = create_scheduler(cfg, optimizer)

    # ------------------------------------------------------------------ #
    # Patch loss
    # ------------------------------------------------------------------ #
    patch_centers = Patchloss.PatchMemory(momentum=0.1, num=1)
    pc_criterion  = Patchloss.Pedal(
        scale=cfg.MODEL.PC_SCALE, k=cfg.MODEL.CLUSTER_K).cuda()

    if cfg.MODEL.SOFT_LABEL and model_name == 'part_attention_vit':
        print("========using soft label========")

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    if model_name == 'part_attention_vit':
        part_attention_vit_do_train_with_amp_exp16(
            cfg,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            loss_func,
            num_query,
            args.local_rank,
            patch_centers = patch_centers,
            pc_criterion  = pc_criterion,
        )
    else:
        raise ValueError(
            f"exp16 only supports part_attention_vit, got: {model_name}")