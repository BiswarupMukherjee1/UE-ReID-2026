# encoding: utf-8
"""
UrbAM-ReID dataset adapter for Part-Aware Transformer.
Loads training data from UrbAM splits (containers, rubishbins, crosswalks)
using the inv/train_label.xml which includes c004 (July inverse) images.
"""

import os.path as osp
import xml.etree.ElementTree as ET

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY

# Hardcoded UrbAM path — PAT passes the Kaggle root, so we ignore it
URBAM_ROOT = '/media/DiscoLocal/IPCV/UE-ReID/UrbAM-ReID/ICIP_UrbAM-ReID'

# PID offset so UrbAM identities don't collide with Kaggle identities (1088 IDs)
PID_OFFSET = 2000

# Which splits to use — the 'inv' variant includes c004 camera images
URBAM_SPLITS = {
    'Containers':  'containers',
    'rubishbinss': 'rubishbins',
    'crosswalks':  'crosswalk',
}


def _parse_xml(xml_path, img_dir, pid_offset=0):
    """Parse a train_label.xml and return list of (img_path, pid, camid) and pid2label."""
    if not osp.exists(xml_path):
        return [], {}

    tree = ET.parse(xml_path)
    root = tree.getroot()

    pid_set = set()
    for item in root.findall('Items/Item'):
        pid_set.add(int(item.get('objectID')))

    pid2label = {pid: idx + pid_offset for idx, pid in enumerate(sorted(pid_set))}

    dataset = []
    for item in root.findall('Items/Item'):
        cam_str  = item.get('cameraID')
        img_name = item.get('imageName')
        pid_raw  = int(item.get('objectID'))

        camid    = int(cam_str[1:])
        pid      = pid2label[pid_raw]
        img_path = osp.join(img_dir, img_name)

        if osp.exists(img_path):
            dataset.append((img_path, pid, camid))

    return dataset, pid2label


def _load_all_splits():
    """Load training data from all three object class splits using inv variant."""
    splits_dir = osp.join(URBAM_ROOT, 'splits')
    all_data = []
    current_offset = PID_OFFSET

    for class_folder, split_name in URBAM_SPLITS.items():
        split_dir = osp.join(splits_dir, class_folder, split_name, 'inv')
        xml_path  = osp.join(split_dir, 'train_label.xml')
        img_dir   = osp.join(split_dir, 'image_train')

        if not osp.exists(xml_path):
            print(f"Warning: {xml_path} not found, skipping {class_folder}")
            continue
        if not osp.exists(img_dir):
            print(f"Warning: {img_dir} not found, skipping {class_folder}")
            continue

        data, pid2label = _parse_xml(xml_path, img_dir, pid_offset=current_offset)
        n_ids = len(pid2label)
        current_offset += n_ids

        print(f"  UrbAM {class_folder}: {len(data)} images, {n_ids} identities")
        all_data.extend(data)

    return all_data


@DATASET_REGISTRY.register()
class UrbAM(ImageDataset):
    """
    UrbAM-ReID dataset — all cameras including c004 (July inverse).
    Used as auxiliary training data to bridge the c004 domain gap.
    """

    def __init__(self, root=None, verbose=True, **kwargs):
        # Ignore root passed by PAT — always use hardcoded UrbAM path
        train = _load_all_splits()

        self.train   = train
        self.query   = []
        self.gallery = []

        if verbose:
            c004 = [x for x in train if x[2] == 4]
            print(f"UrbAM loaded: {len(train)} total images, {len(c004)} c004 images")

        super(UrbAM, self).__init__(self.train, self.query, self.gallery, **kwargs)


@DATASET_REGISTRY.register()
class UrbAM_c004only(ImageDataset):
    """
    UrbAM-ReID dataset — c004 images ONLY.
    Only adds back-view (July inverse) images, directly targeting the domain gap.
    """

    def __init__(self, root=None, verbose=True, **kwargs):
        splits_dir = osp.join(URBAM_ROOT, 'splits')
        all_data = []
        current_offset = PID_OFFSET

        for class_folder, split_name in URBAM_SPLITS.items():
            split_dir = osp.join(splits_dir, class_folder, split_name, 'inv')
            xml_path  = osp.join(split_dir, 'train_label.xml')
            img_dir   = osp.join(split_dir, 'image_train')

            if not osp.exists(xml_path) or not osp.exists(img_dir):
                continue

            tree = ET.parse(xml_path)
            root_xml = tree.getroot()

            c004_pids = set()
            for item in root_xml.findall('Items/Item'):
                if item.get('cameraID') == 'c004':
                    c004_pids.add(int(item.get('objectID')))

            pid2label = {pid: idx + current_offset for idx, pid in enumerate(sorted(c004_pids))}
            current_offset += len(pid2label)

            for item in root_xml.findall('Items/Item'):
                if item.get('cameraID') != 'c004':
                    continue
                img_name = item.get('imageName')
                pid_raw  = int(item.get('objectID'))
                img_path = osp.join(img_dir, img_name)

                if osp.exists(img_path) and pid_raw in pid2label:
                    all_data.append((img_path, pid2label[pid_raw], 4))

            print(f"  UrbAM_c004only {class_folder}: {len(c004_pids)} identities")

        self.train   = all_data
        self.query   = []
        self.gallery = []

        if verbose:
            print(f"UrbAM_c004only loaded: {len(all_data)} c004 images")

        super(UrbAM_c004only, self).__init__(self.train, self.query, self.gallery, **kwargs)
