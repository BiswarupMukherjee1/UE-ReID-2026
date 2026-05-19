"""
Processor for exp16 — view-specific embeddings + camera equalization.

Changes vs part_attention_vit_processor.py:
1. Derives view_ids from target_cam every iteration
2. Calls model(img, view_ids=view_ids) instead of model(img)
3. Validation loop also passes view_ids for accurate mAP logging
4. All other logic (PC_LOSS, AMP, grad clipping, NaN guard) unchanged

view_ids: (B,) long tensor
  0 = forward view  (camid 1, 2, 3)
  1 = backward view (camid 4 = c004)

NO DANN in this processor.
"""

import logging
import os
import time
import torch
import torch.nn as nn
from model.make_model import make_model
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from data.build_DG_dataloader import build_reid_test_loader
from torch.utils.tensorboard import SummaryWriter


def part_attention_vit_do_train_with_amp_exp16(
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
        pc_criterion=None):
    """
    Training loop for exp16.
    Identical to original processor but passes view_ids to the model.
    """
    log_period        = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period       = cfg.SOLVER.EVAL_PERIOD
    device            = "cuda"
    epochs            = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("PAT.train")
    logger.info('start training exp16 (view embeddings + camera equalization)')

    tb_path  = os.path.join(cfg.TB_LOG_ROOT, cfg.LOG_NAME)
    tbWriter = SummaryWriter(tb_path)
    print(f"saving tblog to {tb_path}")

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print(f'Using {torch.cuda.device_count()} GPUs for training')
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], find_unused_parameters=True)

    total_loss_meter = AverageMeter()
    reid_loss_meter  = AverageMeter()
    pc_loss_meter    = AverageMeter()
    acc_meter        = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler    = amp.GradScaler(init_scale=512)

    # ------------------------------------------------------------------ #
    # PC_LOSS initialization (unchanged from original processor)
    # ------------------------------------------------------------------ #
    if cfg.MODEL.PC_LOSS:
        print('initialize the centers')
        model.train()
        for i, informations in enumerate(train_loader):
            with torch.no_grad():
                input  = informations['images'].cuda(non_blocking=True)
                vid    = informations['targets']
                camid  = informations['camid']
                path   = informations['img_path']
                # view_ids for initialization forward pass
                view_ids_init = (camid == 4).long().cuda()
                _, _, layerwise_feat_list = model(input, view_ids=view_ids_init)
                patch_centers.get_soft_label(
                    path, layerwise_feat_list[-1], vid=vid, camid=camid)
        print('initialization done')

    best_index = 1
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        total_loss_meter.reset()
        reid_loss_meter.reset()
        acc_meter.reset()
        pc_loss_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()

        for n_iter, informations in enumerate(train_loader):
            img       = informations['images']
            vid       = informations['targets']
            camid     = informations['camid']
            img_path  = informations['img_path']
            t_domains = informations['others']['domains']

            optimizer.zero_grad()
            img       = img.to(device)
            target    = vid.to(device)
            target_cam = camid.to(device)
            t_domains  = t_domains.to(device)

            # ---------------------------------------------------------------- #
            # Derive view_ids: 0=forward (c001/c002/c003), 1=backward (c004)
            # ---------------------------------------------------------------- #
            view_ids = (target_cam == 4).long()   # (B,) on same device as target

            model.to(device)
            with amp.autocast(enabled=True):
                # Pass view_ids to model — the key change for exp16
                score, layerwise_global_feat, layerwise_feat_list = \
                    model(img, view_ids=view_ids)

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

                total_loss = reid_loss + l_ploss * ploss

            scaler.scale(total_loss).backward()

            # Gradient clipping (same as exp15 with SGD)
            if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)

            # NaN/Inf guard
            if any(torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                   for p in model.parameters() if p.grad is not None):
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

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                # Log c004 count this batch for monitoring
                c004_in_batch = (target_cam == 4).sum().item()
                logger.info(
                    "Epoch[{}] Iter[{}/{}] loss:{:.3f} reid:{:.3f} "
                    "pc:{:.3f} Acc:{:.3f} c004:{} lr:{:.2e}".format(
                        epoch, n_iter + 1, len(train_loader),
                        total_loss_meter.avg, reid_loss_meter.avg,
                        pc_loss_meter.avg, acc_meter.avg,
                        c004_in_batch,
                        scheduler._get_lr(epoch)[0]))
                tbWriter.add_scalar('train/reid_loss', reid_loss_meter.avg,
                                    n_iter + 1 + (epoch - 1) * len(train_loader))
                tbWriter.add_scalar('train/acc', acc_meter.avg,
                                    n_iter + 1 + (epoch - 1) * len(train_loader))
                tbWriter.add_scalar('train/pc_loss', pc_loss_meter.avg,
                                    n_iter + 1 + (epoch - 1) * len(train_loader))
                tbWriter.add_scalar('train/c004_per_batch', c004_in_batch,
                                    n_iter + 1 + (epoch - 1) * len(train_loader))

        end_time       = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info(
            "Epoch {} done. Time/batch:{:.3f}s Speed:{:.1f}samples/s".format(
                epoch, time_per_batch,
                cfg.SOLVER.IMS_PER_BATCH / time_per_batch))

        log_path = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)

        if epoch % eval_period == 0:
            cmc, mAP = do_inference_exp16(cfg, model, val_loader, num_query)
            tbWriter.add_scalar('val/Rank@1', cmc[0], epoch)
            tbWriter.add_scalar('val/mAP',    mAP,    epoch)

        if epoch % checkpoint_period == 0:
            best_index = epoch
            logger.info(f"=====saving epoch: {best_index}=====")
            torch.save(model.state_dict(),
                       os.path.join(log_path,
                                    cfg.MODEL.NAME + f'_{epoch}.pth'))
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Final evaluation on best checkpoint
    # ------------------------------------------------------------------ #
    load_path  = os.path.join(log_path,
                               cfg.MODEL.NAME + f'_{best_index}.pth')
    eval_model = make_model(cfg, modelname=cfg.MODEL.NAME,
                             num_class=0, camera_num=None, view_num=None)
    eval_model.load_param(load_path)
    print(f'Loaded weights from {cfg.MODEL.NAME}_{best_index}.pth')

    for testname in cfg.DATASETS.TEST:
        if 'ALL' in testname:
            testname = 'DG_' + testname.split('_')[1]
        val_loader_final, num_query_final = build_reid_test_loader(cfg, testname)
        do_inference_exp16(cfg, eval_model, val_loader_final, num_query_final)

    print('saving final checkpoint. Do not interrupt!')
    torch.save(eval_model.state_dict(),
               os.path.join(log_path, cfg.MODEL.NAME + f'_{epoch}.pth'))
    print('done!')


def do_inference_exp16(cfg, model, val_loader, num_query):
    """
    Inference with view_ids for accurate mAP logging during training.
    view_ids derived from camid in the val_loader batch.
    """
    device = "cuda"
    logger = logging.getLogger("PAT.test")
    logger.info("Enter inferencing (exp16 with view embeddings)")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print(f'Using {torch.cuda.device_count()} GPUs for inference')
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    t0 = time.time()

    for n_iter, informations in enumerate(val_loader):
        img    = informations['images']
        pid    = informations['targets']
        camids = informations['camid']
        with torch.no_grad():
            img      = img.to(device)
            view_ids = (camids == 4).long().to(device)
            feat     = model(img, view_ids=view_ids)
            evaluator.update((feat, pid, camids))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results")
    logger.info(f"mAP: {mAP:.1%}")
    for r in [1, 5, 10]:
        logger.info(f"CMC curve, Rank-{r:<3}:{cmc[r-1]:.1%}")
    logger.info(f"total inference time: {time.time() - t0:.2f}")
    return cmc, mAP