import sys
sys.path.append('.')

import json
import os
import os.path as osp
import random
import PIL
import cv2
import moxing as mox
import numpy as np
from datasets.base.base_dataset import BaseDataset
from datetime import datetime
import torch
from datasets.camera_perturbation_numpy import add_camera_perturbation
from pi3.utils.geometry import depthmap_to_camera_coordinates
# from pi3.utils.basic import write_ads_dataset_point_and_images, mat4_to_pose

def depth_extrinsic_remap(depthmap, camera_pose, intrinsics, extrinsics_offset):
    # 校正深度图外参偏差：将深度图反投到相机坐标系，沿相机z轴向后移动20cm，再投影回来
    # 1. 反投深度图到相机坐标系3D点
    pts3d_cam, valid_mask = depthmap_to_camera_coordinates(depthmap, intrinsics)
    # 2. 沿相机z轴向后移动20cm（z值增加0.2m）
    pts3d_cam_adjusted = pts3d_cam.copy()
    # pts3d_cam_adjusted[..., 2] -= 0.0  # 相机坐标系z轴向前，向后移动即z值增加
    # extrinsics_offset[:3, :3] = np.eye(3)
    pts3d_cam_adjusted = pts3d_cam_adjusted @ np.transpose(extrinsics_offset[:3, :3]) + np.transpose(extrinsics_offset[:3, 3:])
    # 3. 重新投影到图像平面
    H_d, W_d = depthmap.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    # 计算投影坐标
    z_cam_adj = pts3d_cam_adjusted[..., 2]
    x_img = pts3d_cam_adjusted[..., 0] * fx / (z_cam_adj + 1e-8) + cx
    y_img = pts3d_cam_adjusted[..., 1] * fy / (z_cam_adj + 1e-8) + cy
    # 取整
    u = np.clip(np.round(x_img).astype(np.int32), 0, W_d - 1)
    v = np.clip(np.round(y_img).astype(np.int32), 0, H_d - 1)
    # 4. 生成新的深度图
    depthmap_corrected = np.zeros_like(depthmap)
    # 只保留有效且深度为正的像素
    valid_proj = valid_mask & (z_cam_adj > 0)
    # 使用向量化方式填充（v作为行索引，u作为列索引）
    depthmap_corrected[v[valid_proj], u[valid_proj]] = z_cam_adj[valid_proj]
    depthmap = depthmap_corrected
    camera_pose_offset = np.linalg.inv(extrinsics_offset)
    camera_pose = camera_pose @ camera_pose_offset
    return depthmap, camera_pose

class AdsDataset(BaseDataset):
    def __init__(
        self,
        obs_path,
        crop=None,
        set_black=None,
        trans_step = 0,
        rot_step = 0,
        input_depth = False,
        rand_rot = None,
        use_relative_pose = False,
        aug_T = False,
        twins_method = False,
        rel_pose_perturb = False,
        save_original_data = False,
        save_resized_data = False,
        save_data_dir = None,
        resize_depth_fine=False,
        extrinsic_trans = None,
        load_seg=False,
        use_all_as_val = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dataset_label = 'Ads'
        self.resolution = kwargs['resolution'][0]
        self.obs_path = obs_path

        self.sequences = []
        self.sequences_files = []
        self.sequences_extrinsic = []
        self.sequences_intrinsic = []
        self.sequences_wh = []
        self.sequences_rel_pose = []  # 存储 rel_pose (车体到世界系)
        self.sequences_cam_extrinsic = []  # 存储 cam_extrinsic (相机到车体系)
        n_frames = 0
        if obs_path.startswith('obs://'):
            tar_file = './' + os.path.basename(obs_path)
            if not mox.file.exists(tar_file):
                mox.file.copy(obs_path, tar_file)
            obs_path = tar_file
        print(f"{datetime.now().strftime('%H:%M:%S')} parser ads dataset from {osp.abspath(obs_path)}")
        with mox.file.File(obs_path, 'rb') as f:
            seq_dict = np.load(f, allow_pickle=True).item()
        if self.mode == 'train' and 'train' in seq_dict:
            seq_dict = seq_dict['train']
        elif self.mode == 'test' and 'val' in seq_dict:
            val_data = seq_dict['val']
            if use_all_as_val:
                if 'train' in seq_dict:
                    train_data = seq_dict['train']
                    for k, v in train_data.items():
                        if k in val_data:
                            val_data[k].update(v)
                        else:
                            val_data[k] = v
                if 'test' in seq_dict:
                    test_data = seq_dict['test']
                    for k,v in test_data.items():
                        if k in val_data:
                            val_data[k].update(v)
                        else:
                            val_data[k] = v
            seq_dict = val_data
        print(f"{datetime.now().strftime('%H:%M:%S')} load npy finish")
        n_seq = 0
        # # if 'obs://yw-ads-training-gy1/data/external/personal/z58801726/pi3/data/20260311_garage' in seq_dict:
        # #     seq_dict.pop('obs://yw-ads-training-gy1/data/external/personal/z58801726/pi3/data/20260311_garage')
        # if 'obs://yw-ads-training-gy1/data/external/personal/z58801726/pi3/data/20260115/' in seq_dict:
        #     seq_dict.pop('obs://yw-ads-training-gy1/data/external/personal/z58801726/pi3/data/20260115/')
        # if 'obs://yw-ads-training-gy1/data/external/personal/z58801726/pi3/data/20260115_validation/' in seq_dict:
        #     seq_dict.pop('obs://yw-ads-training-gy1/data/external/personal/z58801726/pi3/data/20260115_validation/')
        #
        for seq_dir, seq_list in seq_dict.items():
            n_seq += len(seq_list)
        for seq_dir,seq_list in seq_dict.items():
            for seq_name, frms in seq_list.items():
                # if frms['wh'][0,0] != 3840 or frms['wh'][0,1] != 2160:
                #     continue
                if len(self.sequences) % 500 == 0:
                    print(f"{datetime.now().strftime('%H:%M:%S')} parser ads dataset {len(self.sequences)} in {n_seq}")
                self.sequences.append(osp.join(seq_dir, seq_name))
                if isinstance(frms, dict):
                    self.sequences_files.append(frms['name'])
                    n_frames += len(frms['name'])
                else:
                    self.sequences_files.append(frms)
                    n_frames += len(frms)
                # Keep all per-sequence metadata lists aligned with self.sequences.
                if isinstance(frms, dict):
                    self.sequences_extrinsic.append(frms.get('extrinsic'))
                    self.sequences_intrinsic.append(frms.get('intrinsic'))
                    self.sequences_wh.append(frms.get('wh'))
                    self.sequences_rel_pose.append(frms.get('rel_pose'))
                    self.sequences_cam_extrinsic.append(frms.get('cam_extrinsic'))
                else:
                    self.sequences_extrinsic.append(None)
                    self.sequences_intrinsic.append(None)
                    self.sequences_wh.append(None)
                    self.sequences_rel_pose.append(None)
                    self.sequences_cam_extrinsic.append(None)


        print(f'[{self.dataset_label}] Found {len(self.sequences)} sequences {n_frames} frames in {self.obs_path}', flush=True)

        self.save_original_data = save_original_data
        self.save_resized_data = save_resized_data
        self.save_data_dir = save_data_dir
        if self.save_data_dir is not None:
            os.makedirs(self.save_data_dir, exist_ok=True)
            os.chmod(self.save_data_dir, 0o777)
        if self.save_original_data:
            from pi3.models.seg.seg_model import SegModel
            self.seg_model = SegModel(r'mask2former_train_eval/configs/adsdataset/seg_cls28_crop_384_672_bs80_iter200_10000.yaml',
                                      '/media/data/model_zoo/ads_semantic/model_0059999.pt')
        self.rel_pose_perturb = rel_pose_perturb
        self.twins_method = twins_method
        self.use_relative_pose = use_relative_pose
        self.resize_depth_fine = resize_depth_fine
        self.aug_T = aug_T
        self.crop = crop
        self.set_black = set_black
        self.trans_step = trans_step
        self.rot_step = rot_step
        self.input_depth = input_depth
        self.rand_rot = rand_rot
        self.dtof_preproc_train = DToFPreprocessor(
            max_range=10, extra_range=5, cut_hfov_deg=30, voxel_res=0.1,
            noise_alpha=0.03, noise_beta=0, noise_dropout=0.2, pose_rot_noise=0.01, pose_trans_noise=0.05
        )
        self.dtof_preproc_val = DToFPreprocessor(max_range=10, voxel_res=0.1)
        # self.in_list = [
        #     'a6aa21fbb27d4376bdc06832c035eb85',
        #     'fcdae87fb911495bb2a2b9b10a20287f',
        #     'eba0bb04f39d4932809c13e374e9f853',
        #     'd1702c0f43224e9aa5f4e1ec8fca4965',
        #     '893d040ac2de4c7fb92fd3c6003caa83',
        #     '3ae4323f52384b35996638cff921822a',
        #     '2890b2bf96614ca586e0e090eb116863',
        # ]
        self.extrinsic_trans=extrinsic_trans
        
        self.frame_step = kwargs.get("frame_step", 1)
        self.num_imgs = {
            self.sequences[i]: len(self.sequences_files[i])
            for i in range(len(self.sequences))
        }

    def get_real_frame_num(self, num):
        if not self.twins_method:
            return num
        even_num = (num // 2) * 2
        return even_num + (even_num // 2)

    def __len__(self):
        return len(self.sequences)

    def resize_depth_no_loss(
        self,
        depth_ori: np.ndarray,
        new_h: int,
        new_w: int,
        valid_threshold: float = 0.0,
    ) -> np.ndarray:
        assert depth_ori.ndim in (2, 3), "输入必须是 H×W 或 N×H×W"

        is_single = depth_ori.ndim == 2
        if is_single:
            depth_ori = depth_ori[None]

        n, ori_h, ori_w = depth_ori.shape
        out = np.zeros((n, new_h, new_w), dtype=np.float32)
        valid_mask = np.isfinite(depth_ori) & (depth_ori > valid_threshold)

        # 正向映射：缩小时尽量保留有效深度点
        n_idx, y_src, x_src = np.where(valid_mask)
        values = depth_ori[n_idx, y_src, x_src].astype(np.float32)

        scale_y = new_h / ori_h
        scale_x = new_w / ori_w

        y_dst = np.clip(
            np.round(y_src * scale_y).astype(np.int64),
            0,
            new_h - 1,
        )
        x_dst = np.clip(
            np.round(x_src * scale_x).astype(np.int64),
            0,
            new_w - 1,
        )

        # 保持旧实现行为：发生冲突时后写入的值覆盖前值
        out[n_idx, y_dst, x_dst] = values

        # 反向最近邻填充：用于放大时减少规则空洞
        y_grid, x_grid = np.meshgrid(
            np.arange(new_h),
            np.arange(new_w),
            indexing="ij",
        )

        y_back = np.clip(
            np.round(y_grid * ori_h / new_h).astype(np.int64),
            0,
            ori_h - 1,
        )
        x_back = np.clip(
            np.round(x_grid * ori_w / new_w).astype(np.int64),
            0,
            ori_w - 1,
        )

        for i in range(n):
            fill_values = depth_ori[i, y_back, x_back]
            fill_valid = valid_mask[i, y_back, x_back]
            empty = out[i] <= valid_threshold

            fill_mask = empty & fill_valid
            out[i][fill_mask] = fill_values[fill_mask]

        return out[0] if is_single else out


    def resize_depth_no_loss_fine(
        self,
        depth_ori: np.ndarray,
        new_h: int,
        new_w: int,
        valid_threshold: float = 0.0,
    ) -> np.ndarray:
        assert depth_ori.ndim in (2, 3), "输入必须是 H×W 或 N×H×W"

        is_single = depth_ori.ndim == 2
        if is_single:
            depth_ori = depth_ori[None]

        n, ori_h, ori_w = depth_ori.shape
        out = np.zeros((n, new_h, new_w), dtype=np.float32)

        y_grid, x_grid = np.meshgrid(
            np.arange(new_h),
            np.arange(new_w),
            indexing="ij",
        )

        # 目标像素映射回原图浮点坐标
        y_float = y_grid * ori_h / new_h
        x_float = x_grid * ori_w / new_w

        y0 = np.clip(np.floor(y_float).astype(np.int64), 0, ori_h - 1)
        y1 = np.clip(np.ceil(y_float).astype(np.int64), 0, ori_h - 1)
        x0 = np.clip(np.floor(x_float).astype(np.int64), 0, ori_w - 1)
        x1 = np.clip(np.ceil(x_float).astype(np.int64), 0, ori_w - 1)

        for i in range(n):
            p0 = depth_ori[i, y0, x0]
            p1 = depth_ori[i, y0, x1]
            p2 = depth_ori[i, y1, x0]
            p3 = depth_ori[i, y1, x1]

            v0 = np.isfinite(p0) & (p0 > valid_threshold)
            v1 = np.isfinite(p1) & (p1 > valid_threshold)
            v2 = np.isfinite(p2) & (p2 > valid_threshold)
            v3 = np.isfinite(p3) & (p3 > valid_threshold)

            d0 = (y_float - y0) ** 2 + (x_float - x0) ** 2
            d1 = (y_float - y0) ** 2 + (x_float - x1) ** 2
            d2 = (y_float - y1) ** 2 + (x_float - x0) ** 2
            d3 = (y_float - y1) ** 2 + (x_float - x1) ** 2

            dists = np.stack(
                [
                    np.where(v0, d0, np.inf),
                    np.where(v1, d1, np.inf),
                    np.where(v2, d2, np.inf),
                    np.where(v3, d3, np.inf),
                ],
                axis=0,
            )

            values = np.stack([p0, p1, p2, p3], axis=0)

            nearest_idx = np.argmin(dists, axis=0)
            out_i = np.take_along_axis(
                values,
                nearest_idx[None],
                axis=0,
            )[0].astype(np.float32)

            has_valid_neighbor = np.isfinite(dists.min(axis=0))
            out_i[~has_valid_neighbor] = 0.0
            out[i] = out_i

        return out[0] if is_single else out

    def _apply_aug_T(self, img_original, depthmap_original, camera_pose, rel_pose_in, intrinsics, R_right, T_right):
        H_matrix = intrinsics @ R_right.T @ np.linalg.inv(intrinsics)
        h_img, w_img = img_original.shape[:2]
        img_original = cv2.warpPerspective(img_original, H_matrix, (w_img, h_img), flags=cv2.INTER_LINEAR)
        depthmap_original = cv2.warpPerspective(depthmap_original, H_matrix, (w_img, h_img), flags=cv2.INTER_NEAREST)
        camera_pose = camera_pose @ T_right
        rel_pose_in = rel_pose_in @ T_right
        return img_original, depthmap_original, camera_pose, rel_pose_in

    def _get_views(self, index, resolution, rng, is_test = False, new_cam_extrinsic = None, save_name = None, run_all_frm=False):
        # index = [i for (i,x) in enumerate(self.sequences) if self.in_list[0] in x][0]
        # self.in_list = self.in_list[1:]
        sequence_dir = self.sequences[index]
        sequence = sequence_dir.split('/')[-1]
        depth_files = self.sequences_files[index]
        img_num = len(self.sequences_files[index])

        self.this_views_info = dict(
            dir = sequence_dir,
            scene = sequence,
        )

        if img_num < self.frame_num:
            return None
        if not is_test:
            img_idx = random.randrange(img_num)
        else:
            img_idx = (img_num // 2)
        front_num = (self.frame_num - 1) // 2
        back_num = self.frame_num - 1 - front_num

        if img_idx - front_num < 0:
            begin = 0
            end = self.frame_num
        elif img_idx + back_num >= img_num:
            begin = img_num - self.frame_num
            end = img_num
        else:
            begin = img_idx - front_num
            end = img_idx + back_num + 1
        idxs = range(begin, end)

        if self.trans_step > 0 and self.rot_step > 0:
            idxs = self.range_adjust(index, idxs, self.trans_step, self.rot_step, self.rand_rot)

        if run_all_frm:
            idxs = range(0, img_num)

        views = []
        views_original = []
        self.this_views_info = dict(
            dir = sequence_dir,
            scene = sequence,
            pairs = idxs,
        )

        if self.input_depth:
            dtof_preproc = self.dtof_preproc_train
            pose_noise = dtof_preproc.random_pose_perturb()
            hfov_cut = dtof_preproc.random_hfov_perturb()

        apply_ext_aug = random.random() < 0.5 and self.aug_T
        T_right = np.eye(4, dtype=np.float32)
        R_right = np.eye(3, dtype=np.float32)
        if apply_ext_aug:
            R_right = get_euler_rotation(
                random.uniform(-1.0, 1.0),  # pitch
                random.uniform(-1.0, 1.0),  # yaw
                random.uniform(-1.0, 1.0)  # roll
            )
            T_right[:3, :3] = R_right

        for idx in idxs:
            # local
            # img = cv2.resize(cv2.imread(osp.join(self.data_root, sequence, 'images', depth_files[idx].replace('.npy', '.png'))), size)
            # depthmap = cv2.resize(np.load(osp.join(self.data_root, sequence, 'depth_npy', depth_files[idx])), size, interpolation=cv2.INTER_NEAREST).astype(np.float32)
            # camera_pose = np.loadtxt(osp.join(self.data_root, sequence, 'extrinsics', depth_files[idx].replace('.npy', '_extrinsic.txt')), dtype=np.float32)
            # intrinsics = np.loadtxt(osp.join(self.data_root, sequence, 'intrinsics', depth_files[idx].replace('.npy', '_intrinsic.txt')), dtype=np.float32)
            # with open(osp.join(self.data_root, sequence, 'intrinsics', depth_files[idx].replace('.npy', '_params.json')), 'r') as f:
            #     data = json.load(f)

            # moxing
            img_path = osp.join(sequence_dir, 'images', depth_files[idx] + '.png')
            if not mox.file.exists(img_path):
                img_path = img_path.replace('.png', '.jpg')
            img = cv2.imdecode(np.frombuffer(mox.file.read(img_path, binary=True), np.uint8), cv2.IMREAD_COLOR)
            img_original = img[..., [2,1,0]]

            # npy depth and npz depth
            depth_npy_path = osp.join(sequence_dir, 'depth_npy', depth_files[idx] + '.npy')
            depth_npz_path = osp.join(sequence_dir, 'depth_npz', depth_files[idx] + '.npz')
            if mox.file.exists(depth_npz_path):
                with mox.file.File(depth_npz_path, 'rb') as f:
                    depthmap_original = np.load(f)['depth'].astype(np.float32)
            elif mox.file.exists(depth_npy_path):
                with mox.file.File(depth_npy_path, 'rb') as f:
                    depthmap_original = np.load(f).astype(np.float32)
            else:
                raise FileNotFoundError(f"Depth file not found: {depth_npy_path} or {depth_npz_path}")

            extrinsic_seq = self.sequences_extrinsic[index]
            if extrinsic_seq is None:
                extrinsic_path = osp.join(
                    sequence_dir, 'extrinsics', depth_files[idx] + '_extrinsic.txt'
                )
                with mox.file.File(extrinsic_path, 'r') as f:
                    camera_pose = np.loadtxt(f, dtype=np.float32)
            else:
                camera_pose = extrinsic_seq[idx].astype(np.float32)
            camera_pose_original = camera_pose.copy()

            rel_pose_in = camera_pose.copy()
            if self.use_relative_pose:
                # [NEW] 使用 rel_pose 和 cam_extrinsic 计算 camera_pose
                rel_pose_seq = self.sequences_rel_pose[index]
                cam_extrinsic_seq = self.sequences_cam_extrinsic[index]
                if rel_pose_seq is not None and cam_extrinsic_seq is not None:
                    rel_pose = rel_pose_seq[idx].copy().astype(np.float32)
                    cam_extrinsic = cam_extrinsic_seq[idx].copy().astype(np.float32)
                    rel_pose_in = (rel_pose @ cam_extrinsic).astype(np.float32)
                else:
                    rel_pose_file = osp.join(sequence_dir, 'rel_poses', depth_files[idx].removesuffix('_cam_1_') + '_rel_pose.txt')
                    cam_extrinsic_file = osp.join(sequence_dir, 'intrinsics', depth_files[idx] + '_params.json')

                    if mox.file.exists(rel_pose_file):
                        with mox.file.File(rel_pose_file, 'r') as f:
                            rel_pose = np.loadtxt(f, dtype=np.float32)

                        with mox.file.File(cam_extrinsic_file, 'r') as f: #cam_extrinsic (rotation: 3x3, translation: 3x1)
                            params = json.load(f)
                            rotation = np.array(params['rotation'], dtype=np.float32)  # 3x3
                            translation = np.array(params['translation'], dtype=np.float32)  # 3x1 or 3
                        cam_extrinsic = np.eye(4, dtype=np.float32)
                        cam_extrinsic[:3, :3] = rotation
                        if translation.ndim == 1:
                            cam_extrinsic[:3, 3] = translation
                        else:
                            cam_extrinsic[:3, 3] = translation.flatten()
                        rel_pose_in = rel_pose @ cam_extrinsic
            else:
                cam_extrinsic_seq = self.sequences_cam_extrinsic[index]
                if cam_extrinsic_seq is not None:
                    cam_extrinsic = cam_extrinsic_seq[idx].copy().astype(np.float32)
                else:
                    cam_extrinsic = np.eye(4, dtype=np.float32)
                cam_extrinsic_original = cam_extrinsic.copy()

            intrinsic_seq = self.sequences_intrinsic[index]
            if intrinsic_seq is None:
                intrinsic_path = osp.join(
                    sequence_dir, 'intrinsics', depth_files[idx] + '_intrinsic.txt'
                )
                with mox.file.File(intrinsic_path, 'r') as f:
                    intrinsics = np.loadtxt(f, dtype=np.float32)
            else:
                intrinsics = intrinsic_seq[idx].copy().astype(np.float32)

            wh_seq = self.sequences_wh[index]
            if wh_seq is None:
                params_path = osp.join(
                    sequence_dir, 'intrinsics', depth_files[idx] + '_params.json'
                )
                with mox.file.File(params_path, 'r') as f:
                    data = json.load(f)
            else:
                data = {
                    'width': wh_seq[idx][0],
                    'height': wh_seq[idx][1],
                }
            intrinsics_original = intrinsics.copy()
            intrinsics_raw = intrinsics.copy()

            if apply_ext_aug:
                img_original, depthmap_original, camera_pose, rel_pose_in = self._apply_aug_T(
                    img_original,
                    depthmap_original,
                    camera_pose,
                    rel_pose_in,
                    intrinsics,
                    R_right,
                    T_right,
                )

            if self.crop is None:
                # uncrop
                img = cv2.resize(img_original, self.resolution)
                # depthmap = cv2.resize(depthmap, self.resolution, interpolation=cv2.INTER_NEAREST).astype(np.float32)
                if self.resize_depth_fine:
                    if depthmap_original.shape[0] > self.resolution[1] and depthmap_original.shape[1] > self.resolution[0]:
                        depthmap = self.resize_depth_no_loss_fine(depthmap_original, self.resolution[1], self.resolution[0])
                    else:
                        depthmap = cv2.resize(depthmap_original, self.resolution, interpolation=cv2.INTER_NEAREST).astype(np.float32)
                else:
                    depthmap = self.resize_depth_no_loss(depthmap_original, self.resolution[1], self.resolution[0])
                if self.set_black is not None:
                    img[:self.set_black[0]] = 0
                    img[(img.shape[0]-self.set_black[1]):] = 0
                    depthmap[:self.set_black[0]] = 0
                    depthmap[(depthmap.shape[0]-self.set_black[1]):] = 0
                factor_w = self.resolution[0] / data['width']
                factor_h = self.resolution[1] / data['height']
                intrinsics[0] *= factor_w
                intrinsics[1] *= factor_h
                intrinsics_original[0] *= float(depthmap_original.shape[1]) / data['width']
                intrinsics_original[1] *= float(depthmap_original.shape[0]) / data['height']
                if self.extrinsic_trans is not None and '20260115' not in sequence_dir:
                    extrinsic_trans_cur = np.eye(4)
                    extrinsic_trans_cur[1,3] = self.extrinsic_trans
                    depthmap, camera_pose = depth_extrinsic_remap(depthmap, camera_pose, intrinsics, extrinsic_trans_cur)
                if new_cam_extrinsic is not None:
                    extrinsic_offset = np.linalg.inv(new_cam_extrinsic) @ cam_extrinsic
                    # print("extrinsic_offset:", mat4_to_pose(extrinsic_offset))
                    depthmap, camera_pose = depth_extrinsic_remap(depthmap, camera_pose, intrinsics, extrinsic_offset)
                    cam_extrinsic = new_cam_extrinsic
                    depthmap_original, camera_pose_original = depth_extrinsic_remap(depthmap_original, camera_pose_original, intrinsics_original, extrinsic_offset)
                    cam_extrinsic_original = new_cam_extrinsic
            else:
                top_crop, bot_crop = self.crop
                ori_top_crop, ori_bot_crop = float(data['height']) / img.shape[0] * self.crop[0], float(data['height']) / img.shape[0] * self.crop[1]
                depth_scale_x, depth_scale_y = float(img.shape[1]) / depthmap.shape[1] , float(img.shape[0]) / depthmap.shape[0]
                assert depth_scale_x == depth_scale_y
                img = cv2.resize(img[top_crop:-bot_crop], self.resolution)
                top_depth_crop, bot_depth_crop = int(top_crop / depth_scale_x), int(bot_crop / depth_scale_x)
                # depthmap = cv2.resize(depthmap[top_depth_crop:-bot_depth_crop], self.resolution, interpolation=cv2.INTER_NEAREST).astype(np.float32)
                depthmap = depthmap_original[top_depth_crop:-bot_depth_crop]
                if self.resize_depth_fine:
                    if depthmap.shape[0] > self.resolution[1] and depthmap.shape[1] > self.resolution[0]:
                        depthmap = self.resize_depth_no_loss_fine(depthmap, self.resolution[1], self.resolution[0])
                    else:
                        depthmap = cv2.resize(depthmap, self.resolution, interpolation=cv2.INTER_NEAREST).astype(np.float32)
                else:
                    depthmap = self.resize_depth_no_loss(depthmap, self.resolution[1], self.resolution[0])
                factor_w = self.resolution[0] / data['width']
                factor_h = self.resolution[1] / (data['height'] - (ori_bot_crop + ori_top_crop))
                intrinsics[0] *= factor_w
                intrinsics[1, 2] -= ori_top_crop
                intrinsics[1] *= factor_h
                pass

            if self.input_depth:
                depthmap_dtof = dtof_preproc(depthmap, intrinsics, pose_noise, hfov_cut)
                depthmap_dtof = depthmap_dtof.astype(depthmap.dtype)
            else:
                depthmap_dtof = None

            img, depthmap, intrinsics = self._crop_resize_if_necessary(
                img, depthmap, intrinsics, resolution, rng=rng
            )

            # The current Pi3 cropping utilities do not support synchronized
            # DTOF cropping/resizing. Keep it disabled unless that pipeline is
            # extended explicitly.
            if depthmap_dtof is not None:
                raise NotImplementedError(
                    'input_depth=True requires DTOF support in pi3.utils.cropping'
                )
            # if not isinstance(img, PIL.Image.Image):
            #     img = PIL.Image.fromarray(img)

            views.append(dict(
                img=img,
                depthmap=depthmap,
                depthmap_dtof=depthmap_dtof,
                rel_pose=rel_pose_in,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                camera_extrinsics=cam_extrinsic,
                raw_intrinsics=intrinsics_raw,
                dataset= self.dataset_label,
                sequence=str(sequence),
                path=sequence_dir,
                label=depth_files[idx],
                instance=str(idx),
                prefix=f'{sequence[:5]}_{depth_files[idx][:12]}',
            ))
            if self.save_original_data:
                if img_original.shape[0] != depthmap_original.shape[0] or img_original.shape[1] != depthmap_original.shape[1]:
                    img_original = cv2.resize(img_original, (depthmap_original.shape[1], depthmap_original.shape[0]))
                img_original = PIL.Image.fromarray(img_original)
                views_original.append(dict(
                    img=img_original,
                    depthmap=depthmap_original,
                    depthmap_dtof=depthmap_dtof,
                    rel_pose=rel_pose_in,
                    camera_pose=camera_pose_original,
                    camera_intrinsics=intrinsics_original,
                    camera_extrinsics=cam_extrinsic_original,
                    raw_intrinsics=intrinsics_raw,
                    dataset=self.dataset_label,
                    sequence=str(sequence),
                    path=sequence_dir,
                    label=depth_files[idx],
                    instance=str(idx),
                    prefix=f'{sequence[:5]}_{depth_files[idx][:12]}',

                ))


        if self.rel_pose_perturb:
            all_intrinsic = np.stack([view['camera_intrinsics'] for view in views], axis=0)
            all_rel_pose = np.stack([view['rel_pose'] for view in views], axis=0)
            perturb_intrinsic, perturb_pose = add_camera_perturbation(all_intrinsic[None], all_rel_pose[None], is_training= not is_test,
                                                                      intrin_noise_scale = 0.02, pose_rot_noise = 0.01, pose_trans_noise = 0.05)
            for idx, view in enumerate(views):
                view['rel_pose'] = perturb_pose[0, idx]
                view['camera_intrinsics'] = perturb_intrinsic[0, idx]
            pass

        # if self.save_resized_data:
        #     write_ads_dataset_point_and_images(views, save_name + '_train', seg_model=self.seg_model, save_curb=False, save_split_curb=False, save_split_pcd = True)

        # if self.save_original_data:
        #     write_ads_dataset_point_and_images(views_original, save_name + '_ori', seg_model=self.seg_model, save_curb=False, save_split_curb=False, save_split_pcd = True)
        return views

def get_euler_rotation(pitch_deg, yaw_deg, roll_deg):
    p, y, r = np.radians(pitch_deg), np.radians(yaw_deg), np.radians(roll_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]], dtype=np.float32)
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]], dtype=np.float32)
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]], dtype=np.float32)
    return Rz @ Ry @ Rx  # Z-Y-X

def list_all_seq_file():
    ads_list = []
    ll = [r'obs://yw-ads-training-gy1/data/external/personal/z58801726/pi3/data/20260115/']
    seq_list = [r'obs://yw-ads-training-gy1/data/external/personal/z58801726/pi3/data/20260115.txt']
    set_to_seq = {}
    for idx, l in enumerate(ll):
        print(f'proc {l}')
        with mox.file.File(os.path.join(seq_list[idx]), 'r') as f:
            seq = [ln.strip() for ln in f.readlines()]
        seq = [a for a in seq if len(a) == 32]
        seq_to_depth = {}
        for i, sq in enumerate(seq):
            print(f'proc {i} in {len(seq)}')
            depth_npy_dir = os.path.join(l, sq, 'depth_npy')
            depth_npz_dir = os.path.join(l, sq, 'depth_npz')
            if mox.file.exists(depth_npz_dir):
                depth = mox.file.list_directory(depth_npz_dir)
                depth = [a.replace('.npz', '') for a in depth]
            elif mox.file.exists(depth_npy_dir):
                depth = mox.file.list_directory(depth_npy_dir)
                depth = [a.replace('.npy', '') for a in depth]
            else:
                continue
            depth = sorted(depth)
            if len(depth) < 8:
                continue
            seq_to_depth[sq] = depth
        set_to_seq[l] = seq_to_depth
    np.save('train_1.0.0.npy', set_to_seq)

def list_all_seq_with_pose_task(l, seq, thread_id):
    seq_to_depth = {}
    for i, sq in enumerate(seq):
        print(f'proc {i} in {len(seq)} at thread {thread_id}')
        #sq = 'cf84feea92f74b769bc76a28c512479a'
        depth_npy_dir = os.path.join(l, sq, 'depth_npy')
        depth_npz_dir = os.path.join(l, sq, 'depth_npz')
        if mox.file.exists(depth_npz_dir):
            depth_dir = depth_npz_dir
            depth_ext = '.npz'
        elif mox.file.exists(depth_npy_dir):
            depth_dir = depth_npy_dir
            depth_ext = '.npy'
        else:
            print(f'depth dir {sq} does not exist')
            continue
        depth = mox.file.list_directory(depth_dir)
        depth = [a.replace(depth_ext, '') for a in depth]
        depth = sorted(depth)
        if len(depth) < 8:
            print(f'skip with find {len(depth)} for {sq}')
            continue
        camera_pose_list = []
        intrinsic_list = []
        wh = []
        depth_list = []
        for d in depth:
            extrinsic_file = osp.join(l, sq, 'extrinsics', d + '_extrinsic.txt')
            intrinsic_file = osp.join(l, sq, 'intrinsics', d + '_intrinsic.txt')
            intrinsic_json = osp.join(l, sq, 'intrinsics', d + '_params.json')
            if not mox.file.exists(extrinsic_file):
                print(f'not exist {extrinsic_file}')
                continue
            if not mox.file.exists(intrinsic_file):
                print(f'not exist {intrinsic_file}')
                continue
            if not mox.file.exists(intrinsic_json):
                print(f'not exist {intrinsic_json}')
                continue
            depth_list.append(d)
            with mox.file.File(extrinsic_file, 'r') as f:
                camera_pose = np.loadtxt(f, dtype=np.float32)
                camera_pose_list.append(camera_pose)
            with mox.file.File(intrinsic_file, 'r') as f:
                intrinsic_list.append(np.loadtxt(f, dtype=np.float32))
            with mox.file.File(intrinsic_json, 'r') as f:
                data = json.load(f)
                wh.append(np.array([data['width'], data['height']]))
        if len(depth_list) < 8:
            print(f'skip with find {len(depth_list)} while have {len(depth)} for {sq}')
            continue
        # curr_pose = np.stack(camera_pose_list, axis= 0)
        # prev_pose = np.stack([camera_pose_list[0]] + camera_pose_list[:-1], axis=0)
        # delta_pose = np.linalg.inv(prev_pose) @ curr_pose
        # dist = np.linalg.norm(delta_pose[:,:3,3], dim=-1)
        # ang = np.acos(np.minimum(np.maximum(np.trace(delta_pose[:, :3, :3], axis1=1, axis2=2), -1), 3)*0.5 - 0.5) * 180 / np.pi
        seq_to_depth[sq] = {'name': depth_list,
                            'extrinsic': np.stack(camera_pose_list, axis=0),
                            'intrinsic': np.stack(intrinsic_list, axis=0),
                            'wh': np.stack(wh, axis=0),}
    return seq_to_depth

def list_all_seq_with_pose_mt(baseData = None):
    from concurrent.futures import ThreadPoolExecutor
    import random
    ll = [r'obs://yw-ads-training-gy1/data/external/personal/z58801726/pi3/data/20260213/']
    seq_list = [r'/media/data/gtfactory/day_park_part_001_seq.txt']
    if baseData is None:
        set_to_seq = {}
    else:
        set_to_seq = np.load(baseData, allow_pickle=True).item()
    print(f"{datetime.now().strftime('%H:%M:%S')} start read mt")
    for idx, l in enumerate(ll):
        print(f'proc {l}')
        with mox.file.File(os.path.join(seq_list[idx]), 'r') as f:
            seq = [ln.strip() for ln in f.readlines()]
        seq = [a for a in seq if len(a) == 32]
        random.shuffle(seq)
        #seq = seq[:3]
        thread_num = 18
        seq_split = [seq[i::thread_num] for i in range(thread_num)]
        seq_sum = sum(len(a) for a in seq_split)
        print(f"ori seq_num {len(seq)} split sum {seq_sum}")
        set_to_seq[l] = {}
        with ThreadPoolExecutor(max_workers=thread_num) as executor:
            future_list = []
            for i in range(thread_num):
                future_list.append(executor.submit(list_all_seq_with_pose_task, l, seq_split[i], i))
            for i in range(thread_num):
                set_to_seq[l].update(future_list[i].result())
            pass
        pass
        print(f'proc {l} finish with {len(set_to_seq[l])} seqs')
        frame_sum = sum(len(seq_content['name']) for seq_name,seq_content in set_to_seq[l].items())
        print(f'proc {l} finish with {frame_sum} frames')
    np.save('ads_train_urban_campus_1.0.1.npy', set_to_seq)
    #os.chmod('ads_train_urban_1.0.1.npy', 0o777)
    print(f"{datetime.now().strftime('%H:%M:%S')} finish read mt")

def get_seq_list():
    with open('/media/data/gtfactory/day_park_part_001.txt', 'r') as f:
        seq = [ln.strip() for ln in f.readlines()]
    seq = seq[::5]
    seq = [ln.split('/')[9] for ln in seq]
    with open('/media/data/gtfactory/day_park_part_001_seq.txt', 'w') as f:
        for sq in seq:
            f.write(sq + '\n')
    pass


class DToFPreprocessor:
    def __init__(
        self, hfov_deg=120, vfov_deg=100, max_range=10,
        extra_range=0, cut_hfov_deg=0,
        voxel_res=None,
        noise_alpha=0,  # base noise (meters)
        noise_beta=0,   # distance-dependent noise
        noise_dropout=0,  # randomly dropout some rays
        pose_rot_noise=0,
        pose_trans_noise=0,
        num_rings=72,
        num_azimuth=600,
        test_max=False,
    ):
        # 前处理0：dtof测量范围
        self.hfov_deg = hfov_deg
        self.vfov_deg = vfov_deg
        self.max_range = max_range
        self.extra_range = extra_range
        self.cut_hfov_deg = cut_hfov_deg
        # 前处理1：降采样
        self.voxel_res = voxel_res
        # 前处理2：深度噪声
        self.noise_alpha = noise_alpha
        self.noise_beta = noise_beta
        self.noise_dropout = noise_dropout
        # 前处理3：相机 to dtof 位姿噪声
        self.pose_rot_noise = pose_rot_noise
        self.pose_trans_noise = pose_trans_noise
        self.test_max = test_max
        # 前处理4：模拟激光ring
        self.num_rings = num_rings
        self.num_azimuth = num_azimuth

        # original + flip about z axis
        self.cam2veh = np.array((
                (-1, 0, 0, -0.0248),
                (0, -1, 0, 1.444),
                (0, 0, 1, 1.825),
                (0, 0, 0, 1)
            ))
        # hardcoded
        # self.cam2veh = np.array((
        #     (1, 0, 0, -0.0248),
        #     (0, 1, 0, 1.454),
        #     (0, 0, 1, 1.8053),
        #     (0, 0, 0, 1)
        # ))

        # original
        # self.dtof2veh = np.array((
        #     (1, 0, 0, 3.5362),
        #     (0, 1, 0, 0),
        #     (0, 0, 1, 0.8026),
        #     (0, 0, 0, 1)
        # ))
        # swap axis
        self.dtof2veh = np.array((
            (1, 0, 0, 0),
            (0, 1, 0, 0.8026),
            (0, 0, 1, 3.5362),
            (0, 0, 0, 1)
        ))
        # add some augmentation to put it lower
        # self.dtof2veh = np.array((
        #     (1, 0, 0, 0),
        #     (0, 1, 0, -1.8026),
        #     (0, 0, 1, 3.5362),
        #     (0, 0, 0, 1)
        # ))
        # hardcoded
        # self.dtof2veh = np.array((
        #     (1, 0, 0, 0),
        #     (0, 1, 0, 5.8026),
        #     (0, 0, 1, 3.5362),
        #     (0, 0, 0, 1)
        # ))
        self.cam2dtof = self.invert_pose(self.dtof2veh) @ self.cam2veh

    def simulate_lidar_ring(self, points_input):
        points = points_input.copy()
        X = points[:, 0]
        Y = points[:, 1]
        Z = points[:, 2]

        elev_min = np.deg2rad(self.vfov_deg * -0.5)
        elev_max = np.deg2rad(self.vfov_deg * 0.5)

        # spherical coords
        r = np.sqrt(X**2 + Y**2 + Z**2)
        az = np.arctan2(X, Z)          # yaw / azimuth
        el = np.arcsin(Y / r)          # elevation

        # quantize
        ring = ((el - elev_min) / (elev_max - elev_min) * self.num_rings).astype(np.int32)
        az_id = ((az + np.pi) / (2 * np.pi) * self.num_azimuth).astype(np.int32)

        # calculate bin centers
        az_center = (az_id + 0.5) / self.num_azimuth * 2 * np.pi - np.pi
        el_center = (ring + 0.5) / self.num_rings * (elev_max - elev_min) + elev_min
        angular_error = (az - az_center) ** 2 + (el - el_center) ** 2

        # valid bins
        valid = (
            (ring >= 0) & (ring < self.num_rings) &
            (az_id >= 0) & (az_id < self.num_azimuth)
        )
        idx = np.nonzero(valid)[0]
        ring = ring[valid]
        az_id = az_id[valid]
        r = r[valid]
        angular_error = angular_error[valid]

        # sort by (bin, error from bin center)
        bin_id = ring * self.num_azimuth + az_id
        order = np.lexsort((angular_error, bin_id))
        bin_id = bin_id[order]
        idx = idx[order]

        # keep closest per bin
        keep = np.ones(len(bin_id), dtype=bool)
        keep[1:] = bin_id[1:] != bin_id[:-1]
        selected_idx = idx[keep]

        # final mask
        mask = np.zeros(points.shape[0], dtype=bool)
        mask[selected_idx] = True

        return mask, points

    @staticmethod
    def invert_pose(T: np.ndarray) -> np.ndarray:
        """
        Invert a 4x4 rigid transform.
        """
        R = T[:3, :3]
        t = T[:3, 3]

        T_inv = np.eye(4)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -R.T @ t
        return T_inv

    @staticmethod
    def save_ply(filename, points, colors=None):
        """
        Save point cloud to PLY.
        """
        if colors is None:
            colors = np.ones_like(points) * 255

        with open(filename, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            for p, c in zip(points, colors):
                f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")

    @staticmethod
    def save_depth_vis(path, depth, lo=0, hi=100):
        depth = np.clip(depth, lo, hi)
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth_norm = (depth - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth)
        depth_img = (depth_norm * 255.0).astype(np.uint8)
        cv2.imwrite(path, depth_img)

    @staticmethod
    def inpaint_depth_local(depth, ksize=3, min_valid_neighbor=2):
        """
        depth: (H, W) float, 0 = invalid
        ksize: kernel size (3 or 5 recommended)
        min_valid: minimum valid neighbors to fill
        """

        depth = depth.astype(np.float32)
        mask = (depth > 0).astype(np.float32)

        # sum of neighbors
        kernel = np.ones((ksize, ksize), np.float32)

        depth_sum = cv2.filter2D(depth, -1, kernel)
        valid_count = cv2.filter2D(mask, -1, kernel)

        # avoid division by zero
        avg = depth_sum / (valid_count + 1e-6)

        # fill only where:
        # - original is invalid
        # - enough valid neighbors
        fill_mask = (depth == 0) & (valid_count >= min_valid_neighbor)

        out = depth.copy()
        out[fill_mask] = avg[fill_mask]

        return out

    def random_pose_perturb(self):
        """
        Generate a random 4x4 SE(3) perturbation matrix.

        Args:
            rot_sigma: rotation std (radians)
            trans_sigma: translation std (same unit as your scene)

        Returns:
            T: (4, 4) numpy array
        """

        # --- 1. sample gaussian noise ---
        w = np.random.randn(3) * self.pose_rot_noise   # so(3)
        t = np.random.randn(3) * self.pose_trans_noise # translation
        if self.test_max:
            w = np.ones_like(w) * self.pose_rot_noise
            t = np.ones_like(t) * self.pose_trans_noise

        # --- 2. Rodrigues (so3 -> SO3) ---
        theta = np.linalg.norm(w)

        if theta < 1e-8:
            R = np.eye(3)
        else:
            k = w / theta
            K = np.array([
                [0, -k[2], k[1]],
                [k[2], 0, -k[0]],
                [-k[1], k[0], 0]
            ])

            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

        # --- 3. build SE(3) ---
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        return T

    def random_hfov_perturb(self):
        if self.test_max:
            return self.cut_hfov_deg
        hfov_random_cut = np.random.randint(0, self.cut_hfov_deg)
        return hfov_random_cut

    def depth2points(self, depth_cam, K_cam):
        """
        Convert depth map to point cloud in camera frame.
        """
        H, W = depth_cam.shape

        fx = K_cam[0, 0]
        fy = K_cam[1, 1]
        cx = K_cam[0, 2]
        cy = K_cam[1, 2]

        # pixel grid
        u, v = np.meshgrid(np.arange(W), np.arange(H))

        # backproject camera depth -> camera frame
        Z = depth_cam
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        points_cam = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

        return points_cam # points_vehicle[:, :3], points_vehicle

    def points2depth(self, points, K_cam, image_size):
        """
        Project 3D points to a depth map using pinhole camera model.

        Args:
            points: (N, 3) array in camera coordinates (x, y, z)
            K_cam: (3, 3) intrinsics matrix
                [[fx,  0, cx],
                    [ 0, fy, cy],
                    [ 0,  0,  1]]
            image_size: (H, W)

        Returns:
            depth_map: (H, W) depth image (float32), 0 = invalid
        """

        H, W = image_size

        # ---- unpack intrinsics ----
        fx = K_cam[0, 0]
        fy = K_cam[1, 1]
        cx = K_cam[0, 2]
        cy = K_cam[1, 2]

        # ---- split coordinates ----
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # ---- keep only points in front of camera ----
        valid = z > 0
        x = x[valid]
        y = y[valid]
        z = z[valid]

        # ---- project to pixel coordinates ----
        u = fx * (x / z) + cx
        v = fy * (y / z) + cy

        # ---- round to nearest pixel ----
        u = np.round(u).astype(np.int32)
        v = np.round(v).astype(np.int32)

        # ---- keep pixels inside image ----
        valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u = u[valid]
        v = v[valid]
        z = z[valid]

        # ---- initialize depth map ----
        depth_map = np.zeros((H, W), dtype=np.float32)

        # ---- z-buffer: keep closest point ----
        # flatten indices
        idx = v * W + u

        # sort by depth (nearest first)
        order = np.argsort(z)
        idx = idx[order]
        z = z[order]

        # keep first occurrence (closest)
        unique_idx, unique_indices = np.unique(idx, return_index=True)

        depth_map_flat = depth_map.reshape(-1)
        depth_map_flat[unique_idx] = z[unique_indices]

        return depth_map

    def voxelize(self, points):
        """ Downsample lidar points with voxels """
        if self.voxel_res is None:
            return np.ones(len(points), dtype=np.bool_)
        # compute voxel coordinates
        voxel_idx = np.floor(points / self.voxel_res).astype(np.int64)
        # unique voxels
        _, unique_indices = np.unique(voxel_idx, axis=0, return_index=True)
        mask = np.zeros(len(points), dtype=bool)
        mask[unique_indices] = True

        return mask

    def add_lidar_noise(self, depth):
        """
        depth: (H, W) depth map in meters
        """

        depth = depth.copy()

        # ---- valid mask ----
        invalid = depth <= 0

        # ---- range-dependent Gaussian noise ----
        sigma = self.noise_alpha + self.noise_beta * depth
        noise = np.random.randn(*depth.shape) * sigma
        if self.test_max:
            noise = np.ones_like(noise) * (self.noise_alpha + self.noise_beta * depth)
        depth_noisy = depth + noise

        # ---- dropout (missing returns) ----
        drop_mask = np.random.rand(*depth.shape) < self.noise_dropout

        depth_noisy[drop_mask | invalid] = 0.0

        return depth_noisy

    def __call__(self, depth_gt, K_cam_gt, pose_noise=np.eye(4), hfov_cut=0, output_prefix=None, return_torch=False):
        """
        Mask a camera-aligned depth map using dToF visibility.

        Assumptions
        ----------
        Camera & dToF share the same axis convention:
            x = right
            y = down
            z = forward (depth)

        Output
        ------
        depth_dtof : depth map aligned with camera image
        """
        depth_cam = depth_gt.copy()
        K_cam = K_cam_gt.copy()

        H, W = depth_cam.shape

        # 0. Densify the depth a bit
        # depth_cam = self.inpaint_depth_local(depth_cam, ksize=5)
        # tifffile.imwrite('densified_depth_cam.tif', depth_cam)

        # 1. Cast depth map into 3d points
        points_cam = self.depth2points(depth_cam, K_cam)

        # 2. Cam -> vehicle -> dtof
        points_cam_homo = np.hstack([points_cam, np.ones((points_cam.shape[0], 1))])
        # 2.1 Cam -> vehicle (for debugging only)
        points_vehicle = (self.cam2veh @ points_cam_homo.T).T[:, :3]
        # 2.2 Cam -> dtof
        points_dtof = (self.cam2dtof @ points_cam_homo.T).T[:, :3]
        
        # 3. Simulate lidar rings
        mask_keep_lidar_ring, _ = self.simulate_lidar_ring(points_dtof)

        # 4. Filter out invisible points in dtof
        x = points_dtof[:, 0]
        y = points_dtof[:, 1]
        z = points_dtof[:, 2]   # forward depth

        hfov = np.deg2rad(self.hfov_deg - hfov_cut)
        vfov = np.deg2rad(self.vfov_deg)

        theta = np.arctan2(x, z)  # horizontal angle
        phi = np.arctan2(y, z)    # vertical angle

        # randomly add max range
        max_range = self.max_range + np.random.random() * self.extra_range
        visible = (
            (z > 0)
            & (z <= max_range)
            & (np.abs(theta) <= hfov / 2)
            & (np.abs(phi) <= vfov / 2)
            & mask_keep_lidar_ring
        )
        points_dtof = points_dtof[visible]

        # 5. Augmentation: downsample 3d points
        voxel_valid_mask = self.voxelize(points_dtof)
        points_dtof = points_dtof[voxel_valid_mask]

        # 6. Add pose perturbation
        dtof2cam = self.invert_pose(self.cam2dtof)
        dtof2cam = dtof2cam @ pose_noise

        # 7. Dtof -> cam
        points_dtof_homo = np.hstack([points_dtof, np.ones((points_dtof.shape[0], 1))])
        points_cam_aug_homo = (dtof2cam @ points_dtof_homo.T).T[:, :3]
        depth_out = self.points2depth(points_cam_aug_homo, K_cam, (H, W))
        depth_out = self.add_lidar_noise(depth_out.reshape(H, W))

        # 8. (Optional, for debugging) visualize sensors and visible points
        if output_prefix is not None:
            if '/' in output_prefix:
                os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

            # -----------------------------
            # sensor centers in vehicle frame
            # -----------------------------

            pts_visible = points_vehicle[visible]
            cam_center = self.cam2veh[:3, 3]
            dtof_center = self.dtof2veh[:3, 3]

            centers = np.stack([cam_center, dtof_center])

            center_colors = np.array([
                [255, 0, 0],   # camera (red)
                [0, 255, 0],   # dtof (green)
            ])

            # -----------------------------
            # save debug PLYs
            # -----------------------------
            self.save_ply(
                f"{output_prefix}_camera_points.ply",
                points_vehicle,
                np.tile([200, 200, 200], (len(points_vehicle), 1))
            )

            self.save_ply(
                f"{output_prefix}_dtof_visible.ply",
                pts_visible,
                np.tile([0, 0, 255], (len(pts_visible), 1))
            )

            self.save_ply(
                f"{output_prefix}_sensor_centers.ply",
                centers,
                center_colors
            )

            print("Saved debug point clouds:")
            print(output_prefix + "_camera_points.ply")
            print(output_prefix + "_dtof_visible.ply")
            print(output_prefix + "_sensor_centers.ply")

        if return_torch:
            return torch.from_numpy(depth_out)

        return depth_out

def extrinsic_test(obs_npy = 'obs://yw-ads-training-gy1/data/external/personal/z58801726/pi3/data/ads_garage_1.1.2.npy',
                   save_dir = '/media/data/train_result/05011_extrinsic_usercar_all_opt'):
    """
    使用AdsCampus配置初始化AdsDataset类，并循环调用_get_views函数加载数据
    """
    import hydra
    from omegaconf import OmegaConf
    import numpy as np

    # 手动构建配置（对应configs/data/ads_campus.yaml的test_dataset.AdsCampus部分）
    config_dict = {
        '_target_': 'datasets.ads_dataset.AdsDataset',
        'obs_path': obs_npy,
        'z_far': 40,
        'frame_num': 16,
        'aug_crop': False,
        'transform': {
            '_partial_': True,
            '_target_': 'datasets.base.transforms.ImgToTensor'
        },
        'aug_focal': False,
        'mode': 'test',
        'crop': None,
        'set_black': None,
        'trans_step': 2,
        'rot_step': 10,
        'rand_rot': None,
        'input_depth': False,
        'use_relative_pose': False,
        'aug_T': False,
        'twins_method': False,
        'rel_pose_perturb': False,
        'resize_depth_fine': False,
        'extrinsic_trans': None,  # 可以根据需要设置为具体的值，例如0.05
        'save_original_data': True,
        'save_resized_data' : False,
        #'save_data_dir' : save_dir,
        'load_seg': False,
        'use_all_as_val': True,
        'resolution': [[672, 378]],  # 默认分辨率
    }

    # 创建OmegaConf配置对象
    cfg = OmegaConf.create(config_dict)

    # 使用hydra.utils.instantiate初始化AdsDataset（参考__init__.py第53行）
    print("Initializing AdsDataset with AdsCampus configuration...")
    dataset = hydra.utils.instantiate(cfg)
    dataset.convert_attributes()

    print(f"Dataset initialized. Total sequences: {len(dataset)}")
    print(f"Dataset mode: {dataset.mode}")
    print(f"Frame number: {dataset.frame_num}")
    print(f"Resolution: {dataset.resolution}")
    if hasattr(dataset, 'extrinsic_trans'):
        print(f"Extrinsic translation: {dataset.extrinsic_trans}")

    # 创建随机数生成器
    rng = np.random.default_rng(42)

    # 循环遍历数据集并调用_get_views
    num_samples = len(dataset)  # 最多测试10个样本
    print(f"\nLoading {num_samples} samples from dataset...")

    for idx in range(num_samples):
        seq_id = dataset.sequences[idx].split('/')[-1]
        # if seq_id != 'fcdae87fb911495bb2a2b9b10a20287f':# and seq_id != '048aa60fedf0477daa41caa30ca35243' and seq_id != 'e2d2f47ce2fb47c4aa14d0cb0a887068':
        #     continue
        try:
            print(f"\nProcessing sequence {idx}/{num_samples}: {dataset.sequences[idx]}")

            # 调用_get_views函数加载数据（参考__init__.py中的用法）
            trans_veh2cam_map = {
                '06dcf6e33666456bacd75d2a188f1b31': np.array([[-0.01107, -0.999937, -0.00173, 0.064486],[0.013391, 0.001582, -0.999909, 1.480552],[0.999849, -0.011093, 0.013373, -2.092963], [0,0,0,1]]),
                '048aa60fedf0477daa41caa30ca35243': np.array([[-0.011458, -0.999879, 0.010504, 0.008164],[0.000977, -0.010516, -0.999944, 1.621047],[0.999934, -0.011447, 0.001097, -2.061062], [0,0,0,1]]),
                'e2d2f47ce2fb47c4aa14d0cb0a887068': np.array([[0.002925, -0.999942, 0.010346, -0.041504],[-0.000839, -0.010348, -0.999946, 1.585167],[0.999995, 0.002916, -0.000869, -2.174642], [0,0,0,1]]),
            }
            #trans_veh2cam = np.array([[-0.005007, -0.999967, 0.006324, -0.028001],[-0.003639, -0.006306, -0.999973, 1.680725],[0.999981, -0.00503, -0.003607, -2.0877], [0,0,0,1]]) # a6aa21fbb27d4376bdc06832c035eb85
            #trans_veh2cam = np.array([[0.002925, -0.999942, 0.010346, -0.041504],[-0.000839, -0.010348, -0.999946, 1.585167],[0.999995, 0.002916, -0.000869, -2.174642], [0,0,0,1]]) # e2d2f47ce2fb47c4aa14d0cb0a887068
            #trans_veh2cam = np.array([[-0.011458, -0.999879, 0.010504, 0.008164],[0.000977, -0.010516, -0.999944, 1.621047],[0.999934, -0.011447, 0.001097, -2.061062], [0,0,0,1]]) # 048aa60fedf0477daa41caa30ca35243
            #trans_veh2cam = np.array([[-0.01107, -0.999937, -0.00173, 0.064486],[0.013391, 0.001582, -0.999909, 1.480552],[0.999849, -0.011093, 0.013373, -2.092963], [0,0,0,1]]) # 06dcf6e33666456bacd75d2a188f1b31
            #trans_veh2cam = trans_veh2cam_map[seq_id]
            #trans_cam2veh = np.linalg.inv(trans_veh2cam)
            os.makedirs(os.path.join(save_dir, seq_id), exist_ok=True)
            os.chmod(osp.join(save_dir, seq_id), 0o777)
            views = dataset._get_views(idx, np.array(dataset.resolution), rng, is_test=True, new_cam_extrinsic=None, save_name=osp.join(save_dir, seq_id, seq_id), run_all_frm=True)

            if views is None or len(views) == 0:
                print(f"  No views returned for sequence {idx}")
                continue

            print(f"  Successfully loaded {len(views)} views")

            # 打印第一个view的基本信息
            for i, view in enumerate(views[:1]):  # 只打印第一个view的信息
                print(f"  View {i}:")
                print(f"    - Image shape: {view['img'].shape if hasattr(view['img'], 'shape') else view['img'].size}")
                print(f"    - Depthmap shape: {view['depthmap'].shape}")
                print(f"    - Sequence: {view['sequence']}")
                print(f"    - Label: {view['label']}")
                print(f"    - Camera pose shape: {view['camera_pose'].shape}")
                print(f"    - Intrinsics shape: {view['camera_intrinsics'].shape}")

        except Exception as e:
            print(f"  Error processing sequence {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\nExtrinsic test completed.")

if __name__ == '__main__':

    S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "")
    if not S3_ENDPOINT.startswith("http://"):
        S3_ENDPOINT = "http://" + S3_ENDPOINT
    mox.file.set_auth(
        ak=os.environ.get("ACCESS_KEY_ID", ""),
        sk=os.environ.get("SECRET_ACCESS_KEY", ""),
        server=S3_ENDPOINT
    )
    print(f"ak:{os.environ.get('ACCESS_KEY_ID', '')}, sk:{os.environ.get('SECRET_ACCESS_KEY', '')}, server:{S3_ENDPOINT}")
    #extrinsic_test(obs_npy='/home/grw/codehub/pi3Train/base/RoadCode_Scale_Map_Advanced_Research/data_from_matao.npy', save_dir='/media/data/train_result/0513_matao_gtcar_opt')
    extrinsic_test(obs_npy='obs://yw-ads-training-gy1/data/external/personal/z58801726/pi3/data/ads_campus_mix_garage_1.1.7.npy', save_dir='/media/data/train_result/0519_395ab15afb1e4110a74dcb66fc99a447_v1.1.7')
    #list_all_seq_with_pose_mt(baseData='/media/data/gtfactory/ads_train_urban_1.0.1.npy')
    #get_seq_list()
    pass
