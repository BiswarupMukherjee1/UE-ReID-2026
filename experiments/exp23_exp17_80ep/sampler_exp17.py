"""
CameraEqualizedSampler for exp17_dann_equalized.

Identical to exp16_viewemb_equalized/sampler_exp16.py.
Copied here so exp17 is fully self-contained.

Guarantees at least 1 c004 image per num_instances group
for c004-capable identities.

WHY:
  Standard RandomIdentitySampler gives 0 c004 images per batch of 32
  (confirmed empirically in exp13/exp15). With 1262 c004 training images
  across 322 c004-capable identities (out of 1779 total), the model
  almost never sees backward-view images.

  In exp15, DANN camera classifier never received c004 gradient signal
  -> cam_acc went UP to 0.77 (only learned c001/c002/c003).
  In exp17, this sampler guarantees c004 in every batch -> DANN
  sees all 4 cameras -> cam_acc can be properly fought down toward 0.25.

GUARANTEE:
  Exactly 1 c004 per chunk of num_instances images (not proportional).
  Gives ~8-9% c004 per batch (vs 0% with standard sampler).

COMPATIBILITY:
  - data_source items: (img_path, pid, camid, add_info_dict)
    item[2] = camid (int, 1-4)
  - Drop-in replacement for RandomIdentitySampler.
  - Does NOT modify any shared files.
"""

import copy
import random
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler


class CameraEqualizedSampler(Sampler):
    """
    Camera-equalized identity sampler.
    Guarantees >= 1 c004 image per num_instances group for c004 identities.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source        = data_source
        self.batch_size         = batch_size
        self.num_instances      = num_instances
        self.num_pids_per_batch = batch_size // num_instances

        # Index maps
        self.index_dic   = defaultdict(list)   # pid -> [all image indices]
        self.c004_dic    = defaultdict(list)   # pid -> [c004 image indices]
        self.nonc004_dic = defaultdict(list)   # pid -> [non-c004 image indices]

        for index, item in enumerate(data_source):
            pid   = item[1]   # original PID (before CommDataset global relabel)
            camid = item[2]   # actual camera ID (1-4)
            self.index_dic[pid].append(index)
            if camid == 4:
                self.c004_dic[pid].append(index)
            else:
                self.nonc004_dic[pid].append(index)

        self.pids = list(self.index_dic.keys())

        # Epoch length (same formula as RandomIdentitySampler)
        self.length = 0
        for pid in self.pids:
            num = len(self.index_dic[pid])
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __len__(self):
        return self.length

    def __iter__(self):
        # ------------------------------------------------------------------ #
        # Step 1: Pre-group each identity's images into chunks of num_instances
        #         with c004 guarantee for c004-capable identities.
        # ------------------------------------------------------------------ #
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            all_idxs  = self.index_dic[pid]
            c004_idxs = self.c004_dic[pid]
            fwd_idxs  = self.nonc004_dic[pid]
            has_c004  = len(c004_idxs) > 0

            if not has_c004:
                # Standard grouping: no c004 for this identity
                idxs = copy.deepcopy(all_idxs)
                if len(idxs) < self.num_instances:
                    idxs = list(np.random.choice(
                        idxs, size=self.num_instances, replace=True))
                random.shuffle(idxs)
                batch_idxs = []
                for idx in idxs:
                    batch_idxs.append(idx)
                    if len(batch_idxs) == self.num_instances:
                        batch_idxs_dict[pid].append(batch_idxs)
                        batch_idxs = []
            else:
                # Camera-equalized grouping: guarantee 1 c004 per chunk.
                total = len(all_idxs)
                if total < self.num_instances:
                    total = self.num_instances
                n_chunks = total // self.num_instances

                # c004 pool (with repetition allowed across chunks)
                c004_pool = list(np.random.permutation(c004_idxs))
                fwd_pool  = list(np.random.permutation(
                    fwd_idxs if fwd_idxs else all_idxs))

                for chunk_i in range(n_chunks):
                    chunk = []

                    # Slot 0: guaranteed c004
                    if len(c004_pool) == 0:
                        c004_pool = list(np.random.permutation(c004_idxs))
                    chunk.append(c004_pool.pop(0))

                    # Slots 1..(num_instances-1): from forward pool
                    remaining_pool = (fwd_pool if fwd_pool else
                                      list(np.random.permutation(all_idxs)))
                    for _ in range(self.num_instances - 1):
                        if len(remaining_pool) == 0:
                            remaining_pool = list(np.random.permutation(
                                all_idxs))
                        chunk.append(remaining_pool.pop(0))

                    # Shuffle within chunk so c004 isn't always at index 0
                    random.shuffle(chunk)
                    batch_idxs_dict[pid].append(chunk)

        # ------------------------------------------------------------------ #
        # Step 2: Build epoch order (same as RandomIdentitySampler)
        # ------------------------------------------------------------------ #
        avai_pids  = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)