# encoding: utf-8
"""
UAM_Unified dataset — official external data provided by challenge organizers.
Structure identical to competition dataset. Uses train.csv for labels.
6,387 images, 479 identities, includes 745 c004 back-view images.
"""

import os.path as osp
import pandas as pd
from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY

UAM_ROOT = '/media/DiscoLocal/IPCV/UE-ReID/UAM_Unified/UAM_Unified'
PID_OFFSET = 2000  # competition has 1088 IDs, offset avoids collision


@DATASET_REGISTRY.register()
class UAM_Unified(ImageDataset):
    """
    Official UAM_Unified external dataset for Urban Elements ReID 2026.
    Combined with competition data to bridge the c004 domain gap.
    """

    def __init__(self, root=None, verbose=True, **kwargs):
        train_csv = osp.join(UAM_ROOT, 'train.csv')
        img_dir   = osp.join(UAM_ROOT, 'image_train')

        df = pd.read_csv(train_csv)

        # Map objectID to global PID with offset
        unique_pids = sorted(df['objectID'].unique())
        pid2label   = {pid: idx + PID_OFFSET for idx, pid in enumerate(unique_pids)}

        # Map cameraID string to int (c001->1, c002->2, etc.)
        def cam_to_int(cam_str):
            return int(cam_str[1:])

        train = []
        for _, row in df.iterrows():
            img_path = osp.join(img_dir, row['imageName'])
            if not osp.exists(img_path):
                continue
            pid   = pid2label[row['objectID']]
            camid = cam_to_int(row['cameraID'])
            train.append((img_path, pid, camid))

        self.train   = train
        self.query   = []
        self.gallery = []

        if verbose:
            c004 = sum(1 for _, _, c in train if c == 4)
            print(f"UAM_Unified: {len(train)} images, "
                  f"{len(unique_pids)} identities, {c004} c004 images")

        super(UAM_Unified, self).__init__(self.train, self.query, self.gallery, **kwargs)
