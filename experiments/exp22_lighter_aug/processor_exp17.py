"""
DANN Processor for exp17_dann_equalized.
Based on exp15_vitlarge_sgd_dann/processor_dann.py.

Key design for exp17 vs exp15:
  - NO view_ids passed to model (view_embedding is a dead param, never applied)
  - CameraEqualizedSampler (in train_exp17.py) guarantees c004 in every batch
    -> DANN camera classifier now receives c004 gradient signal
    -> cam_acc can be properly fought down toward 0.25 (all 4 cameras)
  - lambda_dann = 0.3 (set in train_exp17.py, higher than exp15's 0.2)
    -> justified because equalization guarantees c004, so GRL can push harder

Progressive lambda schedule (step-level, not epoch-level):
  p = current_step / total_steps  (0 -> 1 over training)
  lambda_effective = lambda_dann * (2/(1+exp(-10*p)) - 1)
  -> near 0 at start (classifier learns cameras freely)
  -> near lambda_dann at end (full adversarial pressure)

Loss:
  total_loss = reid_loss + l_ploss * ploss + camera_loss
  NOTE: camera_loss is NOT multiplied by lambda_effective in the forward pass.
        GRL already applies -lambda_effective during backward.
        Double-multiplying would incorrectly scale gradients as lambda_eff^2.

Checkpoints saved every CHECKPOINT_PERIOD epochs for both:
  - part_attention_vit_{epoch}.pth  (main model)
  - camera_classifier_{epoch}.pth   (camera classifier, for debugging)
"""

import logging
import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp

from model.make_model import make_model
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from data.build_DG_dataloader import build_reid_test_loader, build_reid_train_loader
from torch.utils.tensorboard import SummaryWriter

from dann_components import grad_reverse


def part_attention_vit_do_train_with_amp_exp17(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        local_rank,
        patch_centers=None,
        pc_criterion=None,
        camera_classifier=None,
        lambda_dann=0.3):

    assert camera_classifier is not None, "camera_classifier required for DANN"

    log_period        = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period       = cfg.SOLVER.EVAL_PERIOD
    device            = "cuda"
    epochs            = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("PAT.train")
    logger.info('start training exp17 DANN (target lambda={})'.format(lambda_dann))

    tb_path = os.path.join(cfg.TB_LOG_ROOT, cfg.LOG_NAME)
    tbWriter = SummaryWriter(tb_path)
    print("saving tblog to {}".format(tb_path))

    if device:
        model.to(local_rank)
        camera_classifier.to(device)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], find_unused_parameters=True)

    total_loss_meter = AverageMeter()
    reid_loss_meter  = AverageMeter()
    pc_loss_meter    = AverageMeter()
    cam_loss_meter   = AverageMeter()
    cam_acc_meter    = AverageMeter()
    acc_meter        = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler    = amp.GradScaler(init_scale=512)

    # PC_LOSS initialization
    if cfg.MODEL.PC_LOSS:
        print('initialize the patch centers')
        model.train()
        for i, informations in enumerate(train_loader):
            with torch.no_grad():
                input = informations['images'].cuda(non_blocking=True)
                vid   = informations['targets']
                camid = informations['camid']
                path  = informations['img_path']
                # No view_ids — exp17 does not use view embeddings
                _, _, layerwise_feat_list = model(input)
                patch_centers.get_soft_label(
                    path, layerwise_feat_list[-1], vid=vid, camid=camid)
        print('patch center initialization done')

    best_index = 1

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        total_loss_meter.reset()
        reid_loss_meter.reset()
        acc_meter.reset()
        pc_loss_meter.reset()
        cam_loss_meter.reset()
        cam_acc_meter.reset()
        evaluator.reset()

        scheduler.step(epoch)
        model.train()
        camera_classifier.train()

        for n_iter, informations in enumerate(train_loader):

            # Step-level progressive lambda: near 0 at start, near lambda_dann at end
            current_step = n_iter + (epoch - 1) * len(train_loader)
            total_steps  = epochs * len(train_loader)
            p = current_step / total_steps
            lambda_effective = lambda_dann * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)

            img      = informations['images']
            vid      = informations['targets']
            camid    = informations['camid']
            img_path = informations['img_path']
            t_domains = informations['others']['domains']

            optimizer.zero_grad()
            img         = img.to(device)
            target      = vid.to(device)
            target_cam  = camid.to(device)    # values: 1,2,3,4
            cam_labels  = target_cam - 1      # 0-indexed: 0,1,2,3
            t_domains   = t_domains.to(device)

            model.to(device)

            with amp.autocast(enabled=True):
                # No view_ids — exp17 does not use view embeddings
                score, layerwise_global_feat, layerwise_feat_list = model(img)

                patch_agent, position = patch_centers.get_soft_label(
                    img_path, layerwise_feat_list[-1], vid=vid, camid=camid)

                l_ploss = cfg.MODEL.PC_LR

                if cfg.MODEL.PC_LOSS:
                    feat = torch.stack(layerwise_feat_list[-1], dim=0)
                    feat = feat[:, ::1, :]
                    ploss, all_posvid = pc_criterion(
                        feat, patch_agent, position, patch_centers,
                        vid=target, camid=target_cam)
                    reid_loss = loss_fn(
                        score, layerwise_global_feat[-1], target,
                        all_posvid=all_posvid,
                        soft_label=cfg.MODEL.SOFT_LABEL,
                        soft_weight=cfg.MODEL.SOFT_WEIGHT,
                        soft_lambda=cfg.MODEL.SOFT_LAMBDA)
                else:
                    ploss     = torch.tensor([0.]).cuda()
                    reid_loss = loss_fn(
                        score, layerwise_global_feat[-1], target,
                        soft_label=cfg.MODEL.SOFT_LABEL)

                # ----------------------------------------------------------
                # DANN: Camera adversarial loss
                # layerwise_global_feat[-1] = CLS token before bottleneck BN
                # Shape: (B, 1024)
                #
                # GRL applies -lambda_effective during backward.
                # Do NOT multiply camera_loss by lambda_effective here —
                # that would double-scale the gradient as lambda_eff^2.
                # ----------------------------------------------------------
                feat_for_dann   = layerwise_global_feat[-1]
                feat_reversed   = grad_reverse(feat_for_dann, lambda_effective)
                camera_logits   = camera_classifier(feat_reversed)
                camera_loss     = F.cross_entropy(camera_logits, cam_labels)

                cam_acc = (camera_logits.max(1)[1] == cam_labels).float().mean()

                total_loss = reid_loss + l_ploss * ploss + camera_loss

            scaler.scale(total_loss).backward()

            # Gradient clipping for SGD stability
            if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(camera_classifier.parameters()),
                    max_norm=1.0)

            # NaN/Inf guard
            all_params = list(model.parameters()) + list(camera_classifier.parameters())
            if any(torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                   for p in all_params if p.grad is not None):
                optimizer.zero_grad()
                scaler.update()
                continue

            scaler.step(optimizer)
            scaler.update()

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            total_loss_meter.update(total_loss.item(), img.shape[0])
            reid_loss_meter.update(reid_loss.item(), img.shape[0])
            acc_meter.update(acc, 1)
            pc_loss_meter.update(ploss.item(), img.shape[0])
            cam_loss_meter.update(camera_loss.item(), img.shape[0])
            cam_acc_meter.update(cam_acc.item(), img.shape[0])

            torch.cuda.synchronize()

            if (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iter[{}/{}] loss:{:.3f} reid:{:.3f} pc:{:.3f} "
                    "cam:{:.3f} cam_acc:{:.3f} ID_acc:{:.3f} lam:{:.4f} lr:{:.2e}".format(
                        epoch, n_iter + 1, len(train_loader),
                        total_loss_meter.avg, reid_loss_meter.avg,
                        pc_loss_meter.avg, cam_loss_meter.avg,
                        cam_acc_meter.avg, acc_meter.avg,
                        lambda_effective, scheduler._get_lr(epoch)[0]))
                tbWriter.add_scalar('train/reid_loss', reid_loss_meter.avg,
                                    n_iter + 1 + (epoch - 1) * len(train_loader))
                tbWriter.add_scalar('train/acc', acc_meter.avg,
                                    n_iter + 1 + (epoch - 1) * len(train_loader))
                tbWriter.add_scalar('train/cam_loss', cam_loss_meter.avg,
                                    n_iter + 1 + (epoch - 1) * len(train_loader))
                tbWriter.add_scalar('train/cam_acc', cam_acc_meter.avg,
                                    n_iter + 1 + (epoch - 1) * len(train_loader))
                tbWriter.add_scalar('train/lambda_eff', lambda_effective,
                                    n_iter + 1 + (epoch - 1) * len(train_loader))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info("Epoch {} done. lambda_eff={:.4f} cam_acc={:.3f} "
                    "Time/batch:{:.3f}s Speed:{:.1f}img/s".format(
                        epoch, lambda_effective, cam_acc_meter.avg,
                        time_per_batch, cfg.SOLVER.IMS_PER_BATCH / time_per_batch))

        log_path = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)

        if epoch % eval_period == 0:
            cmc, mAP = do_inference(cfg, model, val_loader, num_query)
            tbWriter.add_scalar('val/Rank@1', cmc[0], epoch)
            tbWriter.add_scalar('val/mAP', mAP, epoch)

        if epoch % checkpoint_period == 0:
            best_index = epoch
            logger.info("=====saving epoch: {}=====".format(best_index))
            torch.save(model.state_dict(),
                       os.path.join(log_path,
                                    cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            torch.save(camera_classifier.state_dict(),
                       os.path.join(log_path,
                                    'camera_classifier_{}.pth'.format(epoch)))

        torch.cuda.empty_cache()

    # Final evaluation on the last saved checkpoint
    load_path = os.path.join(log_path,
                             cfg.MODEL.NAME + '_{}.pth'.format(best_index))
    eval_model = make_model(cfg, modelname=cfg.MODEL.NAME,
                            num_class=0, camera_num=None, view_num=None)
    eval_model.load_param(load_path)
    print('load weights from {}_{}.pth'.format(cfg.MODEL.NAME, best_index))

    for testname in cfg.DATASETS.TEST:
        if 'ALL' in testname:
            testname = 'DG_' + testname.split('_')[1]
        val_loader, num_query = build_reid_test_loader(cfg, testname)
        do_inference(cfg, eval_model, val_loader, num_query)

    print('saving final checkpoint. Do not interrupt!')
    torch.save(eval_model.state_dict(),
               os.path.join(log_path, cfg.MODEL.NAME + '_{}.pth'.format(best_index)))
    print('done!')


def do_inference(cfg, model, val_loader, num_query):
    device = "cuda"
    logger = logging.getLogger("PAT.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    t0 = time.time()

    for n_iter, informations in enumerate(val_loader):
        img    = informations['images']
        pid    = informations['targets']
        camids = informations['camid']
        imgpath = informations['img_path']
        with torch.no_grad():
            img  = img.to(device)
            feat = model(img)
            evaluator.update((feat, pid, camids))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    logger.info("total inference time: {:.2f}".format(time.time() - t0))
    return cmc, mAP