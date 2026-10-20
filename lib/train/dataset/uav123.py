import os
import cv2
import torch
import numpy as np
import random
from .base_video_dataset import BaseVideoDataset
from lib.train.data.image_loader import jpeg4py_loader_w_failsafe
from lib.train.admin import env_settings
from lib.test.evaluation.uav123dataset import UAV123Dataset


class Uav123(BaseVideoDataset):
    """ UAV123 training dataset loader for LTR/PyTracking framework.

    Publication:
        A Benchmark and Simulator for UAV Tracking.
        Matthias Mueller, Neil Smith and Bernard Ghanem
        ECCV, 2016.

    Directory structure:
        <root>/
            anno/UAV123/
                bike1.txt, bike2.txt, ...
            data_seq/UAV123/
                bike1/000001.jpg, ...
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader_w_failsafe):
        root = env_settings().uav123_dir if root is None else root
        super().__init__('UAV123', root, image_loader)

        # Sử dụng danh sách 123 sequences chuẩn mực từ benchmark
        self.sequence_info_list = UAV123Dataset._get_sequence_info_list(None)

        self.sequence_list = [s['name'] for s in self.sequence_info_list]
        self.class_list = sorted(list(set(s['object_class'] for s in self.sequence_info_list)))

    def get_name(self):
        return 'uav123'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return False

    def get_sequence_info(self, seq_id):
        seq_info = self.sequence_info_list[seq_id]
        anno_path = os.path.join(self.root, seq_info['anno_path'])

        if not os.path.exists(anno_path):
            raise FileNotFoundError(f"Annotation file not found: {anno_path}")

        # Đọc tọa độ bounding box dạng (x, y, w, h)
        raw_gt = np.loadtxt(anno_path, delimiter=',', dtype=np.float32)
        if raw_gt.ndim == 1:
            raw_gt = raw_gt.reshape(1, 4)

        bbox = torch.tensor(raw_gt, dtype=torch.float32)

        # Lọc sạch các frame chứa giá trị NaN (khi mục tiêu khuất / ngoài tầm nhìn)
        nan_mask = torch.isnan(bbox).any(dim=1)
        bbox[nan_mask] = 0.0

        # Frame hợp lệ: không chứa NaN và có w > 0, h > 0
        valid = (~nan_mask) & (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone()
        visible_ratio = visible.float()

        return {
            'bbox': bbox,
            'valid': valid,
            'visible': visible,
            'visible_ratio': visible_ratio
        }

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_info = self.sequence_info_list[seq_id]
        if anno is None:
            anno = self.get_sequence_info(seq_id)

        nz = seq_info['nz']
        ext = seq_info['ext']
        start_frame = seq_info['startFrame']
        seq_dir = os.path.join(self.root, seq_info['path'])

        frame_list = []
        for fid in frame_ids:
            frame_num = start_frame + int(fid)
            img_path = os.path.join(seq_dir, f"{frame_num:0{nz}}.{ext}")

            frame = None
            if os.path.exists(img_path):
                if self.image_loader is not None:
                    try:
                        frame = self.image_loader(img_path)
                    except Exception:
                        frame = None

                if frame is None:
                    img_bgr = cv2.imread(img_path)
                    if img_bgr is not None:
                        frame = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            if frame is None:
                # Fallback khung hình đen an toàn nếu file ảnh thiếu
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)

            frame_list.append(frame)

        anno_frames = {
            k: [v[fid, ...].clone() if v.ndim > 1 else v[fid].clone() for fid in frame_ids]
            for k, v in anno.items()
        }
        obj_meta = {'object_class_name': seq_info['object_class']}

        return frame_list, anno_frames, obj_meta