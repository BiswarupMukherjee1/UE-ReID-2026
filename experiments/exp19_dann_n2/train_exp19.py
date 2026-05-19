"""
Train script for exp18_dann_oversample.
DANN + Identity-Level c004 Oversampling (bug-fixed sampler).

Changes from train_exp17.py:
  1. Sampler: CameraEqualizedSampler → CameraOversampleSampler
     with n_c004_per_batch=4 (forces 4 c004-capable identities per batch)
  2. LOG_NAME: exp17_dann_equalized → exp18_dann_oversample
  3. Logging: reports oversampling stats

Everything else identical to exp17:
  - Camera DANN (lambda=0.3, 3-layer MLP classifier)
  - Processor: same forward pass, same DANN loss
  - Config: same hyperparameters, same augmentations
  - UAM_Unified with all splits

Run from Part-Aware-Transformer-main directory:
    python /media/DiscoLocal/IPCV/UE-ReID/experiments/exp18_dann_oversample/train_exp18.py \
        --config_file /media/DiscoLocal/IPCV/UE-ReID/experiments/exp18_dann_oversample/config/train.yml
"""

import sys
import os

EXP18_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP18_DIR)

PAT_DIR = '/media/DiscoLocal/IPCV/UE-ReID/Part-Aware-Transformer-main'
sys.path.insert(0, PAT_DIR)
os.chdir(PAT_DIR)

from processor_exp19 import part_attention_vit_do_train_with_amp_exp17
from sampler_exp19 import CameraOversampleSampler
from dann_components import CameraClassifier

from utils.logger import setup_logger
from data.build_DG_dataloader import build_reid_train_loader, build_reid_test_loader
from data.build_DG_dataloader import fast_batch_collator
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

# ─────────────────────────────────────────────────────────────────────────────
# Experiment constants
# ─────────────────────────────────────────────────────────────────────────────
LAMBDA_DANN      = 0.3
NUM_CAMERAS      = 4
N_C004_PER_BATCH = 2   # force 2 c004-capable identities per batch (out of 8 total)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID exp18 DANN + c004 Oversampling")
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
    logger.info("Experiment: exp18_dann_oversample")
    logger.info("LAMBDA_DANN: {}".format(LAMBDA_DANN))
    logger.info("N_C004_PER_BATCH: {} / {} (50% of batch identities are c004-capable)".format(
        N_C004_PER_BATCH, cfg.SOLVER.IMS_PER_BATCH // cfg.DATALOADER.NUM_INSTANCE))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded config: {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            logger.info("\n" + cf.read())
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # ─────────────────────────────────────────────────────────────────────────
    # Build train loader with CameraOversampleSampler
    # ─────────────────────────────────────────────────────────────────────────
    train_loader_default = build_reid_train_loader(cfg)
    train_set            = train_loader_default.dataset  # CommDataset

    # Dataset stats
    c004_total    = sum(1 for item in train_set.img_items if item[2] == 4)
    all_total     = len(train_set.img_items)
    c004_pids     = len(set(item[1] for item in train_set.img_items if item[2] == 4))
    total_pids    = len(set(item[1] for item in train_set.img_items))
    logger.info("Dataset: {:,} images, {:,} c004 ({:.1f}%)".format(
        all_total, c004_total, 100.0 * c004_total / all_total))
    logger.info("Identities: {:,} total, {:,} c004-capable ({:.1f}%)".format(
        total_pids, c004_pids, 100.0 * c004_pids / total_pids))

    sampler = CameraOversampleSampler(
        data_source      = train_set.img_items,
        batch_size       = cfg.SOLVER.IMS_PER_BATCH,
        num_instances    = cfg.DATALOADER.NUM_INSTANCE,
        n_c004_per_batch = N_C004_PER_BATCH,
    )
    logger.info("CameraOversampleSampler: {} c004-capable + {} forward-only identities per batch".format(
        N_C004_PER_BATCH,
        cfg.SOLVER.IMS_PER_BATCH // cfg.DATALOADER.NUM_INSTANCE - N_C004_PER_BATCH))

    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, cfg.SOLVER.IMS_PER_BATCH, drop_last=True)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers   = cfg.DATALOADER.NUM_WORKERS,
        batch_sampler = batch_sampler,
        collate_fn    = fast_batch_collator,
    )
    logger.info("Train loader: {} batches/epoch".format(len(train_loader)))

    # ─────────────────────────────────────────────────────────────────────────
    # Validation loader
    # ─────────────────────────────────────────────────────────────────────────
    val_name              = cfg.DATASETS.TEST[0]
    val_loader, num_query = build_reid_test_loader(cfg, val_name)
    num_classes           = len(train_set.pids)
    logger.info("num_classes: {}".format(num_classes))

    # ─────────────────────────────────────────────────────────────────────────
    # Model
    # ─────────────────────────────────────────────────────────────────────────
    model_name = cfg.MODEL.NAME
    model = make_model(cfg, modelname=model_name,
                       num_class=num_classes,
                       camera_num=None, view_num=None)

    if cfg.MODEL.FREEZE_PATCH_EMBED and 'resnet' not in model_name:
        model.base.patch_embed.proj.weight.requires_grad = False
        model.base.patch_embed.proj.bias.requires_grad   = False
        logger.info("Frozen patch_embed for stability")

    # ─────────────────────────────────────────────────────────────────────────
    # Camera classifier
    # ─────────────────────────────────────────────────────────────────────────
    camera_classifier = CameraClassifier(in_dim=1024, num_cameras=NUM_CAMERAS)
    camera_classifier = camera_classifier.cuda()
    logger.info("Camera classifier: 1024 -> 512 (BN+ReLU) -> 256 (BN+ReLU) -> {}".format(NUM_CAMERAS))

    # ─────────────────────────────────────────────────────────────────────────
    # Loss + Optimizer + Scheduler
    # ─────────────────────────────────────────────────────────────────────────
    loss_func, center_cri = build_loss(cfg, num_classes=num_classes)
    optimizer = make_optimizer(cfg, model)
    optimizer.add_param_group({
        'params':       camera_classifier.parameters(),
        'lr':           cfg.SOLVER.BASE_LR,
        'weight_decay': cfg.SOLVER.WEIGHT_DECAY,
    })
    logger.info("Added camera_classifier params to optimizer")

    scheduler = create_scheduler(cfg, optimizer)

    # ─────────────────────────────────────────────────────────────────────────
    # Patch loss
    # ─────────────────────────────────────────────────────────────────────────
    patch_centers = Patchloss.PatchMemory(momentum=0.1, num=1)
    pc_criterion  = Patchloss.Pedal(scale=cfg.MODEL.PC_SCALE,
                                    k=cfg.MODEL.CLUSTER_K).cuda()

    if cfg.MODEL.SOFT_LABEL and model_name == 'part_attention_vit':
        logger.info("Using soft label")

    # ─────────────────────────────────────────────────────────────────────────
    # Train
    # ─────────────────────────────────────────────────────────────────────────
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
        raise ValueError(
            "exp18 only supports part_attention_vit, got: {}".format(model_name))