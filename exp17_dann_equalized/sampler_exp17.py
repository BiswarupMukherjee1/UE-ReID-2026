"""Camera-balanced identity sampler.

Drop-in replacement for PAT's RandomIdentitySampler. Batches contain
num_pids_per_batch identities with num_instances images each. For every
identity that has backward-camera (c004) images, each group of num_instances
images is built as one c004 image plus forward-camera images of the same
identity. Identities without c004 images are grouped exactly as in the
default sampler.

This spreads the backward-view images across groups, so the triplet loss
sees more same-identity pairs that span the forward/backward viewpoint gap.

data_source items: (img_path, pid, camid, info), camid in 1..4.
"""

import copy
import random
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler


class CameraEqualizedSampler(Sampler):
    """Identity sampler that places one c004 image in each group of a c004-capable identity."""

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
            pid   = item[1]
            camid = item[2]   # camera ID, 1-4
            self.index_dic[pid].append(index)
            if camid == 4:
                self.c004_dic[pid].append(index)
            else:
                self.nonc004_dic[pid].append(index)

        self.pids = list(self.index_dic.keys())

        # Epoch length, same formula as RandomIdentitySampler
        self.length = 0
        for pid in self.pids:
            num = len(self.index_dic[pid])
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __len__(self):
        return self.length

    def __iter__(self):
        # Group each identity's images into chunks of num_instances.
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            all_idxs  = self.index_dic[pid]
            c004_idxs = self.c004_dic[pid]
            fwd_idxs  = self.nonc004_dic[pid]
            has_c004  = len(c004_idxs) > 0

            if not has_c004:
                # No c004 images: default grouping
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
                # One c004 image per chunk, the rest from forward cameras
                total = len(all_idxs)
                if total < self.num_instances:
                    total = self.num_instances
                n_chunks = total // self.num_instances

                # c004 images are reused across chunks if needed
                c004_pool = list(np.random.permutation(c004_idxs))
                fwd_pool  = list(np.random.permutation(
                    fwd_idxs if fwd_idxs else all_idxs))

                for chunk_i in range(n_chunks):
                    chunk = []

                    # first slot: a c004 image
                    if len(c004_pool) == 0:
                        c004_pool = list(np.random.permutation(c004_idxs))
                    chunk.append(c004_pool.pop(0))

                    # remaining slots: forward-camera images
                    remaining_pool = (fwd_pool if fwd_pool else
                                      list(np.random.permutation(all_idxs)))
                    for _ in range(self.num_instances - 1):
                        if len(remaining_pool) == 0:
                            remaining_pool = list(np.random.permutation(
                                all_idxs))
                        chunk.append(remaining_pool.pop(0))

                    # shuffle so the c004 image is not always first
                    random.shuffle(chunk)
                    batch_idxs_dict[pid].append(chunk)

        # Build the epoch order, same procedure as RandomIdentitySampler.
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
