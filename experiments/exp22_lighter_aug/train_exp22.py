"""
Train script for exp21_224x224.
DANN + Camera Equalization (NO view embeddings).

Combines:
  - exp15: camera-adversarial DANN loss (lambda=0.3)
  - exp16: CameraEqualizedSampler (guarantees c004 in every batch)
  - exp16: UAM_Unified with train+query+test splits (already in place)

Why this is expected to beat exp15 (0.14941):
  exp15 DANN failed because c004 never appeared in batches (0/32 confirmed).
  Camera classifier learned c001/c002/c003 only -> cam_acc stayed at 0.77.
  DANN provided regularization noise but NOT true camera invariance.

  exp17 fixes this: CameraEqualizedSampler guarantees 1-7 c004 per batch.
  Camera classifier now sees all 4 cameras -> GRL can fight cam_acc
  down toward 0.25 -> true cross-camera feature invariance.

  lambda_dann = 0.3 (vs 0.2 in exp15): justified because equalization
  guarantees c004 gradient signal, so stronger pressure is safe.

Changes vs train_exp15.py:
  1. LAMBDA_DANN: 0.2 -> 0.3
  2. DataLoader: build_reid_train_loader -> CameraEqualizedSampler override
     (same pattern as train_exp16.py)
  3. No view_embedding check (exp17 does not use view embeddings)
  4. Imports from processor_exp17 (not processor_dann)

Changes vs train_exp16.py:
  1. DANN camera_classifier added
  2. camera_classifier params added to optimizer
  3. No view_embedding check (removed — exp17 doesn't need it)
  4. Imports from processor_exp17 (with DANN, not processor_exp16)

Run from Part-Aware-Transformer-main directory:
    python /media/DiscoLocal/IPCV/UE-ReID/experiments/exp21_224x224/train_exp17.py \
        --config_file /media/DiscoLocal/IPCV/UE-ReID/experiments/exp21_224x224/config/train.yml

NO shared files are modified. UAM_Unified.py is already the exp16 version
(all splits). vit_pytorch.py view_embedding patch is present but view_ids
is never passed -> view_embedding stays at zeros -> adds nothing to forward.
"""

import sys
import os

EXP17_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP17_DIR)

PAT_DIR = '/media/DiscoLocal/IPCV/UE-ReID/Part-Aware-Transformer-main'
sys.path.insert(0, PAT_DIR)
os.chdir(PAT_DIR)

from processor_exp17 import part_attention_vit_do_train_with_amp_exp17
from sampler_exp17 import CameraEqualizedSampler
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
import torchvision.transforms as TF_aug
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import argparse
from config import cfg
import loss as Patchloss

# ─────────────────────────────────────────────────────────────────────────────
# Experiment constants
# ─────────────────────────────────────────────────────────────────────────────

# lambda_dann = 0.3 (higher than exp15's 0.2)
# Justified: CameraEqualizedSampler guarantees c004 in every batch,
# so GRL receives real c004 signal and can apply stronger invariance pressure.
LAMBDA_DANN  = 0.3

# c001=1, c002=2, c003=3, c004=4
NUM_CAMERAS  = 4


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = True



class StochasticAugmentation:
    """
    Stochastic augmentation for exp22 — DANN-compatible.

    Pool: perspective distortion + color jitter.
    CJ is disabled from PAT config and moved here into the stochastic pool
    so that ALL augmentations are controlled by one probability gate.

    p_none=0.40: 40% clean images — camera classifier learns cameras stably.
    p_one=0.60:  60% one randomly chosen augmentation from pool.
    No stacking: max 1 augmentation per image.
    No rotation: c004 gap is horizontal viewpoint reversal, not tilt.

    RPT stays in PAT config (batch-level operation, cannot be PIL-pooled).

    Perspective distortion_scale=0.05:
      6px shift on 128px-wide image. Camera geometric cues survive.
      DANN camera classifier unaffected.

    ColorJitter matches PAT config values (same params, now stochastic).
    """

    def __init__(self):
        self.persp = TF_aug.RandomPerspective(
            distortion_scale=0.05, p=1.0, fill=128)

    def __call__(self, img):
        r = random.random()
        if r < 0.40:
            return img              # 40%: clean
        else:
            return self.persp(img)  # 60%: gentle perspective (6px on 128px)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID exp17 DANN + Equalization")
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
    logger.info("Experiment: exp22_lighter_aug")
    logger.info("LAMBDA_DANN: {}".format(LAMBDA_DANN))
    logger.info("NUM_CAMERAS: {}".format(NUM_CAMERAS))
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

    # ─────────────────────────────────────────────────────────────────────────
    # Build train loader with CameraEqualizedSampler
    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: build default loader to get the properly-constructed CommDataset
    # (includes domain info added by build_DG_dataloader)
    train_loader_default = build_reid_train_loader(cfg)
    train_set            = train_loader_default.dataset   # CommDataset

    # Log c004 stats to verify equalization will have effect
    c004_total = sum(1 for item in train_set.img_items if item[2] == 4)
    all_total  = len(train_set.img_items)
    logger.info("Dataset: {:,} images, {:,} c004 ({:.1f}%)".format(
        all_total, c004_total, 100.0 * c004_total / all_total))

    # Step 2: replace sampler with CameraEqualizedSampler
    sampler = CameraEqualizedSampler(
        data_source   = train_set.img_items,
        batch_size    = cfg.SOLVER.IMS_PER_BATCH,
        num_instances = cfg.DATALOADER.NUM_INSTANCE,
    )
    logger.info("Using CameraEqualizedSampler "
                "(guarantees 1 c004 per group for c004-capable identities)")

    # Step 3: build new DataLoader with custom batch sampler
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, cfg.SOLVER.IMS_PER_BATCH, drop_last=True)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers   = cfg.DATALOADER.NUM_WORKERS,
        batch_sampler = batch_sampler,
        collate_fn    = fast_batch_collator,
    )
    logger.info("Train loader: {} batches/epoch".format(len(train_loader)))

    # ── Stochastic augmentation: inject BEFORE existing PAT transforms ──────
    # Applied to PIL images; existing Resize(224,224) fixes any size changes.
    original_transform = train_set.transform
    
    train_set.transform = TF_aug.Compose([StochasticAugmentation(), original_transform])
    logger.info("StochasticAugmentation injected: perspective + rotation "
                "(p_none=0.15, p_one=0.45, p_two=0.40)")


    # ─────────────────────────────────────────────────────────────────────────
    # Validation loader
    # ─────────────────────────────────────────────────────────────────────────
    val_name   = cfg.DATASETS.TEST[0]
    val_loader, num_query = build_reid_test_loader(cfg, val_name)
    num_classes = len(train_set.pids)
    logger.info("num_classes: {}".format(num_classes))

    # ─────────────────────────────────────────────────────────────────────────
    # Model
    # ─────────────────────────────────────────────────────────────────────────
    model_name = cfg.MODEL.NAME
    model = make_model(cfg, modelname=model_name,
                       num_class=num_classes,
                       camera_num=None,
                       view_num=None)

    if cfg.MODEL.FREEZE_PATCH_EMBED and 'resnet' not in model_name:
        model.base.patch_embed.proj.weight.requires_grad = False
        model.base.patch_embed.proj.bias.requires_grad   = False
        logger.info("Frozen patch_embed for training stability (MoCo v3 trick)")

    # Note: vit_pytorch.py has view_embedding from exp16 patch.
    # view_ids is NEVER passed in exp17 -> view_embedding stays at zeros
    # and is never applied. It is a dead parameter.
    #if hasattr(model.base, 'view_embedding'):
    #    logger.info("view_embedding present in model but will NOT be used "
    #                "(view_ids never passed in exp17)")

    # ─────────────────────────────────────────────────────────────────────────
    # Camera classifier (separate from main model, DANN-specific)
    # ─────────────────────────────────────────────────────────────────────────
    camera_classifier = CameraClassifier(in_dim=1024, num_cameras=NUM_CAMERAS)
    camera_classifier = camera_classifier.cuda()
    logger.info("Camera classifier: 1024 -> 512 (BN+ReLU) -> "
                "256 (BN+ReLU) -> {}".format(NUM_CAMERAS))

    # ─────────────────────────────────────────────────────────────────────────
    # Loss + Optimizer
    # ─────────────────────────────────────────────────────────────────────────
    loss_func, center_cri = build_loss(cfg, num_classes=num_classes)

    # Build optimizer for main model
    optimizer = make_optimizer(cfg, model)

    # Add camera_classifier parameters to the same optimizer
    # Same lr and weight_decay as main model
    optimizer.add_param_group({
        'params':       camera_classifier.parameters(),
        'lr':           cfg.SOLVER.BASE_LR,
        'weight_decay': cfg.SOLVER.WEIGHT_DECAY,
    })
    logger.info("Added camera_classifier params to optimizer "
                "(lr={}, wd={})".format(cfg.SOLVER.BASE_LR,
                                        cfg.SOLVER.WEIGHT_DECAY))

    # ─────────────────────────────────────────────────────────────────────────
    # Scheduler
    # ─────────────────────────────────────────────────────────────────────────
    scheduler = create_scheduler(cfg, optimizer)

    # ─────────────────────────────────────────────────────────────────────────
    # Patch loss (required by processor even when PC_LOSS=False)
    # ─────────────────────────────────────────────────────────────────────────
    patch_centers = Patchloss.PatchMemory(momentum=0.1, num=1)
    pc_criterion  = Patchloss.Pedal(
        scale=cfg.MODEL.PC_SCALE, k=cfg.MODEL.CLUSTER_K).cuda()

    if cfg.MODEL.SOFT_LABEL and model_name == 'part_attention_vit':
        logger.info("Using soft label")

    # ─────────────────────────────────────────────────────────────────────────
    # Train
    # ─────────────────────────────────────────────────────────────────────────
    if model_name == 'part_attention_vit':
        part_attention_vit_do_train_with_amp_exp17(
            cfg,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            loss_func,
            num_query,
            args.local_rank,
            patch_centers     = patch_centers,
            pc_criterion      = pc_criterion,
            camera_classifier = camera_classifier,
            lambda_dann       = LAMBDA_DANN,
        )
    else:
        raise ValueError(
            "exp17 only supports part_attention_vit, got: {}".format(model_name))