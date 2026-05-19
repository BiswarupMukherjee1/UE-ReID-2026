"""
CameraOversampleSampler for exp18_dann_oversample.

Two improvements over CameraEqualizedSampler (exp17/exp16):

BUG FIX — remaining_pool alias:
  exp17 code: remaining_pool = (fwd_pool if fwd_pool else ...)
  Because Python list assignment is reference, not copy,
  remaining_pool and fwd_pool pointed to the SAME list object.
  Popping from remaining_pool depleted fwd_pool permanently.
  Once fwd_pool was empty, subsequent chunks created a fresh
  all_idxs permutation PER CHUNK instead of sequential sampling.
  Some forward images were never seen; the equalization guarantee
  broke for identities with few forward images.

  Fix: operate directly on fwd_pool in the inner loop. No alias.

IDENTITY-LEVEL OVERSAMPLING (with-replacement):
  Motivation: at inference, 100% of queries are c004.
  Standard random batch selection gives ~1.4 c004-capable identities
  per batch (18.1% of 1,779). This experiment forces n_c004_per_batch=4
  out of 8 (50%) throughout the FULL epoch.

  Implementation: c004-capable identities' chunk lists are extended by
  repeat_factor (~5×) at the start of __iter__ so they never exhaust
  while forward-only identities are consumed once. Batch selection uses
  separate pools: 4 from c004-capable, 4 from forward-only.

  Known trade-off: c004-capable identities' images repeat ~5× per epoch.
  Accepted deliberately: cross-view exposure is the training signal we
  most need given 100% c004 query at inference.

  When c004-capable pool runs dry (shouldn't happen with repeat_factor),
  gracefully falls back to filling from forward-only pool.

COMPATIBILITY:
  - data_source items: (img_path, pid, camid, add_info_dict)
  - Drop-in replacement: same __init__ signature + n_c004_per_batch kwarg
  - Does NOT modify any shared files
"""

import copy
import math
import random
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler


class CameraOversampleSampler(Sampler):
    """
    Camera oversampling sampler with bug fix and identity-level c004 forcing.
    Forces n_c004_per_batch c004-capable identities per batch throughout epoch.
    """

    def __init__(self, data_source, batch_size, num_instances, n_c004_per_batch=4):
        self.data_source        = data_source
        self.batch_size         = batch_size
        self.num_instances      = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        self.n_c004_per_batch   = n_c004_per_batch
        self.n_fwd_per_batch    = self.num_pids_per_batch - n_c004_per_batch

        assert n_c004_per_batch < self.num_pids_per_batch, \
            "n_c004_per_batch must be < num_pids_per_batch"

        # Index maps
        self.index_dic   = defaultdict(list)
        self.c004_dic    = defaultdict(list)
        self.nonc004_dic = defaultdict(list)

        for index, item in enumerate(data_source):
            pid   = item[1]
            camid = item[2]
            self.index_dic[pid].append(index)
            if camid == 4:
                self.c004_dic[pid].append(index)
            else:
                self.nonc004_dic[pid].append(index)

        self.pids             = list(self.index_dic.keys())
        self.c004_capable_pids = [p for p in self.pids if self.c004_dic[p]]
        self.fwd_only_pids    = [p for p in self.pids if not self.c004_dic[p]]

        # Epoch length estimate (based on standard formula, approximate)
        self.length = 0
        for pid in self.pids:
            num = len(self.index_dic[pid])
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    #def __len__(self):
    #    return self.length

    #def __len__(self):
    # Estimate actual epoch length based on forward-only pool exhaustion
     #   total_fwd_chunks = sum(
      #  max(len(self.index_dic[p]), self.num_instances) // self.num_instances
      #  for p in self.fwd_only_pids
    #)
      #  n_batches = total_fwd_chunks // self.n_fwd_per_batch
       # return n_batches * self.batch_size

    def __len__(self):
        # Estimate actual epoch length based on forward-only pool exhaustion
        total_fwd_chunks = sum(
            max(len(self.index_dic[p]), self.num_instances) // self.num_instances
            for p in self.fwd_only_pids
        )
        n_batches = total_fwd_chunks // self.n_fwd_per_batch
        return n_batches * self.batch_size

    def _build_chunks_for_pid(self, pid):
        """
        Build a list of image-index chunks (each of length num_instances)
        for a single identity. c004-capable identities get 1 guaranteed c004
        per chunk. Bug-fixed: no remaining_pool alias.
        """
        all_idxs  = self.index_dic[pid]
        c004_idxs = self.c004_dic[pid]
        fwd_idxs  = self.nonc004_dic[pid]
        has_c004  = bool(c004_idxs)

        total = len(all_idxs)
        if total < self.num_instances:
            total = self.num_instances
        n_chunks = total // self.num_instances

        chunks = []

        if not has_c004:
            # ── Forward-only identity: standard sequential sampling ──
            idxs = copy.deepcopy(all_idxs)
            if len(idxs) < self.num_instances:
                idxs = list(np.random.choice(idxs, size=self.num_instances,
                                             replace=True))
            random.shuffle(idxs)
            batch = []
            for idx in idxs:
                batch.append(idx)
                if len(batch) == self.num_instances:
                    chunks.append(batch)
                    batch = []
        else:
            # ── c004-capable identity: 1 guaranteed c004 per chunk ──
            c004_pool = list(np.random.permutation(c004_idxs))
            fwd_pool  = list(np.random.permutation(
                fwd_idxs if fwd_idxs else all_idxs))

            for _ in range(n_chunks):
                chunk = []

                # Slot 0: guaranteed c004
                if not c004_pool:
                    c004_pool = list(np.random.permutation(c004_idxs))
                chunk.append(c004_pool.pop(0))

                # Slots 1..(num_instances-1): sequential from fwd_pool
                # BUG FIX: operate directly on fwd_pool, no alias
                for _ in range(self.num_instances - 1):
                    if not fwd_pool:
                        # Exhausted: refill from forward images only to protect 1-in-4 ratio
                        fwd_pool = list(np.random.permutation(fwd_idxs if fwd_idxs else all_idxs))
                    chunk.append(fwd_pool.pop(0))

                random.shuffle(chunk)
                chunks.append(chunk)

        return chunks

    def __iter__(self):
        # ──────────────────────────────────────────────────────────────── #
        # Step 1: Build base chunks for every identity
        # ──────────────────────────────────────────────────────────────── #
        batch_idxs_dict = {}
        for pid in self.pids:
            batch_idxs_dict[pid] = self._build_chunks_for_pid(pid)

        # ──────────────────────────────────────────────────────────────── #
        # Step 2: Extend c004-capable chunks (with-replacement oversampling)
        # Goal: ensure c004-capable pool lasts for the full epoch
        # ──────────────────────────────────────────────────────────────── #
        total_fwd_chunks = sum(len(batch_idxs_dict[p])
                               for p in self.fwd_only_pids)
        n_batches_est   = (total_fwd_chunks // self.n_fwd_per_batch
                           if self.n_fwd_per_batch > 0 else 0)
        n_c004_needed   = n_batches_est * self.n_c004_per_batch
        n_c004_avail    = sum(len(batch_idxs_dict[p])
                              for p in self.c004_capable_pids)

        if n_c004_avail > 0 and n_c004_needed > n_c004_avail:
            repeat_factor = math.ceil(n_c004_needed / n_c004_avail)
            for pid in self.c004_capable_pids:
                base = batch_idxs_dict[pid]
                extended = []
                for _ in range(repeat_factor):
                    # Reshuffle order of chunks each repeat for variety
                    reshuffled = copy.deepcopy(base)
                    random.shuffle(reshuffled)
                    extended.extend(reshuffled)
                batch_idxs_dict[pid] = extended

        # ──────────────────────────────────────────────────────────────── #
        # Step 3: Batch selection — 4 c004-capable + 4 forward-only
        # Stops when forward-only pool is exhausted (defines epoch length)
        # ──────────────────────────────────────────────────────────────── #
        avai_c004 = [p for p in self.c004_capable_pids
                     if batch_idxs_dict[p]]
        avai_fwd  = [p for p in self.fwd_only_pids
                     if batch_idxs_dict[p]]
        final_idxs = []

        while len(avai_fwd) >= self.n_fwd_per_batch:

            # How many c004-capable identities can we provide this batch?
            n_c004_this = min(self.n_c004_per_batch, len(avai_c004))
            n_fwd_this  = self.num_pids_per_batch - n_c004_this

            # Forward pool must have enough for this batch
            if len(avai_fwd) < n_fwd_this:
                break

            sel_c004 = (random.sample(avai_c004, n_c004_this)
                        if n_c004_this > 0 else [])
            sel_fwd  = random.sample(avai_fwd, n_fwd_this)

            for pid in sel_c004 + sel_fwd:
                if not batch_idxs_dict[pid]:
                    continue
                final_idxs.extend(batch_idxs_dict[pid].pop(0))

                # Remove from available pool if chunks exhausted
                if not batch_idxs_dict[pid]:
                    if pid in avai_c004:
                        avai_c004.remove(pid)
                    if pid in avai_fwd:
                        avai_fwd.remove(pid)

        return iter(final_idxs)