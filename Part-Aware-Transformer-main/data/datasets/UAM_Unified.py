# encoding: utf-8
"""
UAM_Unified dataset — exp16 version.

CHANGES vs original:
- Now reads image_query (517 c004 images, 153 identities)
  AND image_test (1791 forward images, 212 identities)
  in addition to image_train (6387 images, 479 identities).
- All three splits are added to self.train for training.
- self.query = [] and self.gallery = [] remain empty (UAM splits
  are not used for PAT's internal validation).

WHY:
- UAM query (c004, labeled) gives 517 more backward-view training images.
- UAM test identities 480-691 have BOTH c004 (in query) AND forward
  (in test) images → 153 identities with cross-camera pairs.
- FhIOSB (2nd place 2025) confirmed using all UAM splits helps.

PID HANDLING:
- train.csv objectIDs: 1-479
- query.csv objectIDs: 480-691 (153 unique, all also in test)
- test.csv objectIDs:  480-691 (212 unique, superset of query)
- Combined unique PIDs: 1-691 (sorted)
- With PID_OFFSET=2000: maps to 2000-2690
- Competition PIDs after UrbanElementsReID relabeling: 0-1087
- CommDataset global relabel: {0-1087} ∪ {2000-2690} → 0..1778
- No collision guaranteed.

CAMERA IDs:
- train: c001(1651), c002(1612), c003(2379), c004(745)
- query: c004(517) only
- test:  c001(597), c002(597), c003(597)
"""

import os.path as osp
import csv
from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY

UAM_ROOT = '/media/DiscoLocal/IPCV/UE-ReID/UAM_Unified/UAM_Unified'
PID_OFFSET = 2000  # competition PIDs go 0-1087; UAM maps to 2000-2690


@DATASET_REGISTRY.register()
class UAM_Unified(ImageDataset):
    """
    UAM_Unified external dataset for Urban Elements ReID 2026.
    Uses train + query + test splits for maximum c004 coverage.
    """

    def __init__(self, root=None, verbose=True, **kwargs):
        # ------------------------------------------------------------------ #
        # Step 1: collect raw (img_path, raw_pid, camid) tuples from each split
        # ------------------------------------------------------------------ #
        raw_items = []

        # --- train split (c001/c002/c003/c004, objectIDs 1-479) ---
        train_csv = osp.join(UAM_ROOT, 'train.csv')
        img_dir_train = osp.join(UAM_ROOT, 'image_train')
        raw_items.extend(self._read_split(train_csv, img_dir_train))

        # --- query split (c004 only, objectIDs 480-691, 153 unique) ---
        query_csv = osp.join(UAM_ROOT, 'query.csv')
        img_dir_query = osp.join(UAM_ROOT, 'image_query')
        raw_items.extend(self._read_split(query_csv, img_dir_query))

        # --- test split (c001/c002/c003, objectIDs 480-691, 212 unique) ---
        test_csv = osp.join(UAM_ROOT, 'test.csv')
        img_dir_test = osp.join(UAM_ROOT, 'image_test')
        raw_items.extend(self._read_split(test_csv, img_dir_test))

        # ------------------------------------------------------------------ #
        # Step 2: build global PID → label mapping with offset
        # ------------------------------------------------------------------ #
        unique_pids = sorted(set(pid for _, pid, _ in raw_items))
        pid2label = {pid: idx + PID_OFFSET
                     for idx, pid in enumerate(unique_pids)}

        # ------------------------------------------------------------------ #
        # Step 3: build final train list
        # ------------------------------------------------------------------ #
        train = []
        for img_path, raw_pid, camid in raw_items:
            train.append((img_path, pid2label[raw_pid], camid))

        self.train   = train
        self.query   = []   # not used; CommDataset relabels globally
        self.gallery = []

        if verbose:
            c004_count = sum(1 for _, _, c in train if c == 4)
            n_ids = len(unique_pids)
            print(f"UAM_Unified (exp16): {len(train)} images, "
                  f"{n_ids} identities, {c004_count} c004 images")

        super(UAM_Unified, self).__init__(
            self.train, self.query, self.gallery, **kwargs)

    # ---------------------------------------------------------------------- #
    # Helper: read one CSV split
    # ---------------------------------------------------------------------- #
    def _read_split(self, csv_path, img_dir):
        """
        Reads a CSV with columns: cameraID, imageName, objectID
        Returns list of (img_path, raw_pid_int, camid_int).
        Skips images that do not exist on disk.
        """
        items = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = osp.join(img_dir, row['imageName'])
                if not osp.exists(img_path):
                    continue
                raw_pid = int(row['objectID'])
                camid   = int(row['cameraID'][1:])  # 'c004' -> 4
                items.append((img_path, raw_pid, camid))
        return items