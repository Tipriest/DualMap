import gzip
import logging
import os
# import pdb
import shutil
import pickle
import threading
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import open3d as o3d
import open_clip
import supervision as sv
import torch
from omegaconf import DictConfig
from PIL import Image
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import SAM, YOLO, FastSAM

from utils.pcd_utils import (
    mask_depth_to_points,
    refine_points_with_clustering,
    safe_create_bbox,
)
from utils.time_utils import timing_context, get_timestamped_path
from utils.types import DataInput, LocalObservation, ObjectClasses
from utils.visualizer import ReRunVisualizer, visualize_result_rgb

# Set up the module-level logger
# 设置模块级日志记录器
logger = logging.getLogger(__name__)

class PoseLowPassFilter:
    """姿态低通滤波器，用于平滑相机位姿。"""

    def __init__(self, alpha=0.95):
        self.alpha = alpha
        self.initialized = False
        self.smoothed_translation = None
        self.smoothed_rotation = None  # Rotation object (scipy)

    def update(self, pose_mat: np.ndarray) -> np.ndarray:
        """
        input 4x4 pose matrix, output smoothed 4x4 pose matrix.
        """
        curr_translation = pose_mat[:3, 3]
        curr_rotation = R.from_matrix(pose_mat[:3, :3])

        if not self.initialized:
            self.smoothed_translation = curr_translation
            self.smoothed_rotation = curr_rotation
            self.initialized = True
        else:
            # translation filtering
            self.smoothed_translation = (
                self.alpha * self.smoothed_translation
                + (1 - self.alpha) * curr_translation
            )

            # 使用slerp进行旋转滤波
            slerp = Slerp(
                [0, 1], R.concatenate([self.smoothed_rotation, curr_rotation])
            )
            self.smoothed_rotation = slerp(1 - self.alpha)

        T_smooth = np.eye(4)
        T_smooth[:3, :3] = self.smoothed_rotation.as_matrix()
        T_smooth[:3, 3] = self.smoothed_translation
        return T_smooth

class Detector:
    # Given input output detection
    """
    检测器类，负责处理输入数据并输出检测结果
    """

    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        """
        Initialize the Detector class.

        Parameters:
        cfg (DictConfig): A configuration object containing paths, parameters, and settings for the detector.
        cfg (DictConfig): 包含检测器路径、参数和设置的配置对象

        Returns:
        None
        """

        # Object classes(对象类别)
        classes_path = cfg.yolo.classes_path
        if cfg.yolo.use_given_classes:
            classes_path = cfg.yolo.given_classes_path
            logger.info(f"[Detector][Init] Using given classes, path:{classes_path}")

        self.obj_classes = ObjectClasses(
            classes_file_path=classes_path,
            bg_classes=cfg.yolo.bg_classes,
            skip_bg=cfg.yolo.skip_bg,
        )

        # get detection paths(获取检测路径)
        self.detection_path = Path(cfg.detection_path)
        self.detection_path.mkdir(parents=True, exist_ok=True)

        # Configs
        self.cfg = cfg
        # Detection results
        # NOTICE: Detection results are stored in Batch, it is not separated by objects
        # 注意：检测结果按批次存储，不按对象分开
        self.curr_results = {}
        # Data input
        self.curr_data = DataInput()
        self.prev_data = None
        # KF for layout keyframe(用于布局关键帧的KF)
        self.prev_keyframe_data = None

        # masked points and colors(掩码点和颜色)
        self.masked_points = []
        self.masked_colors = []
        # Observations, a list for each obj observation(观测，每个对象观测的列表)
        self.curr_observations = []

        # visualizer
        self.visualizer = ReRunVisualizer()
        self.annotated_image = None

        # Variables for FastSAM
        self.unknown_class_id = len(self.obj_classes.get_classes_arr()) - 1
        self.annotated_image_fs = None
        self.annotated_image_fs_after = None

        # Layout Pointcloud(布局点云)
        self.layout_pointcloud = o3d.geometry.PointCloud()
        self.layout_num = 0
        self.layout_time = 0.0
        # For thread processing(用于线程处理)
        self.layout_lock = (
            threading.Lock()
        )  # 用于保护layout_pointcloud的线程锁
        self.data_thread = None  # 线程句柄
        self.data_event = threading.Event()  # 线程通知事件

        logger.info(f"[Detector][Init] Initilizating detection modules...")

        if cfg.run_detection:
            try:
                # CLIP module
                logger.info(
                    f"[Detector][Init] Loading CLIP model: {cfg.clip.model_name} with pretrained weights '{cfg.clip.pretrained}'"
                )

                self.clip_model, _, self.clip_preprocess = (
                    open_clip.create_model_and_transforms(
                        cfg.clip.model_name, pretrained=cfg.clip.pretrained
                    )
                )
                self.clip_model = self.clip_model.to(cfg.device)
                self.clip_model.eval()

                # Only reparameterize if the model is MobileCLIP
                # 仅当模型为MobileCLIP时才重新参数化
                if "MobileCLIP" in cfg.clip.model_name:
                    from mobileclip.modules.common.mobileone import reparameterize_model

                    self.clip_model = reparameterize_model(self.clip_model)

                self.clip_tokenizer = open_clip.get_tokenizer(cfg.clip.model_name)
            except Exception as e:
                logger.error(f"[Detector][Init] Error loading CLIP model: {e}")
                return

            try:
                # Detection module
                logger.info(
                    f"[Detector][Init] Loading YOLO model from\t{cfg.yolo.model_path}"
                )
                self.yolo:YOLO = YOLO(cfg.yolo.model_path)
                self.yolo.set_classes(self.obj_classes.get_classes_arr())
            except Exception as e:
                logger.error(f"[Detector][Init] Error loading YOLO model: {e}")
                return

            try:
                # Segmentation module
                logger.info(
                    f"[Detector][Init] Loading SAM model from\t{cfg.sam.model_path}"
                )
                self.sam = SAM(cfg.sam.model_path)
            except Exception as e:
                logger.error(f"[Detector][Init] Error loading SAM model: {e}")
                return

            # Open fastsam for open vocabulary detection(为开放词汇检测打开fastsam)
            if cfg.use_fastsam:
                try:
                    logger.info(
                        f"[Detector][Init] Loading FastSAM model from\t{cfg.fastsam.model_path}"
                    )
                    self.fastsam = FastSAM(cfg.fastsam.model_path)
                except Exception as e:
                    logger.error(f"[Detector][Init] Error loading FASTSAM model: {e}")
                    return

            logger.info("[Detector][Init] Initializing high-low mobility classifier.初始化固定/易移动的物体分类器")
            lm_examples = cfg.lm_examples
            hm_examples = cfg.hm_examples
            lm_descriptions = cfg.lm_descriptions
            num_examples = [len(lm_examples), len(hm_examples), len(lm_descriptions)]
            prototypes = lm_examples + hm_examples + lm_descriptions
            proto_feats = get_text_features(
                prototypes,
                self.clip_model,
                self.clip_tokenizer,
                device=cfg.device,
                clip_length=cfg.clip.clip_length,
            )
            self.num_examples = num_examples
            self.proto_feats = proto_feats

            # Get the text feats of all the classes(获取所有类别的文本特征)
            class_feats = get_text_features(
                self.obj_classes.get_classes_arr(),
                self.clip_model,
                self.clip_tokenizer,
                device=cfg.device,
                clip_length=cfg.clip.clip_length,
            )
            self.class_feats = class_feats

            # Used for unknown class(用于未知类别)
            if cfg.use_avg_feat_for_unknown:
                class_feats_mean = np.mean(class_feats, axis=0)
                self.class_feats_mean = class_feats_mean / np.linalg.norm(
                    class_feats_mean
                )

            with timing_context("Detection Filter", self):
                self.filter = Filter(
                    classes=self.obj_classes,
                    small_mask_size=self.cfg.small_mask_th,
                    skip_refinement=self.cfg.skip_refinement,
                )
                self.filter.set_device(self.cfg.device)

        # for filtering the pose of follower camera for visualization(用于可视化跟随相机的姿态滤波)
        self.pose_filter_follower = PoseLowPassFilter(alpha=0.95)

        logger.info(f"[Detector][Init] Finish Init.")

    def set_data_input_thread(self, curr_data: DataInput) -> None:
        """设置数据输入，并在后台线程中处理它。"""
        self.curr_data = curr_data

        if not self.cfg.preload_layout:
            # If a thread is already running, wait for it to finish(如果已有线程在运行，阻塞并等待它完成)
            if self.data_thread and self.data_thread.is_alive():
                self.data_thread.join()

            # Create a new thread to process data input(创建一个新线程来处理数据输入)
            self.data_thread = threading.Thread(target=self._process_data_input_thread)
            self.data_thread.start()

    def _process_data_input_thread(self):
        """_summary_
        1. 检查当前一帧的点云是否为关键帧点云
        2. 如果是关键帧点云的话, 将RGBD图像转换为点云并附加到self.layout_pointcloud中
        """
        # Initialize prev_kf_data and layout_pointcloud
        if self.prev_keyframe_data is None:
            self.prev_keyframe_data = self.curr_data.copy()
            layout_pcd = self.depth_to_point_cloud(sample_rate=16)
            with self.layout_lock:  # Ensure thread safety for layout_pointcloud(确保 layout_pointcloud 的线程安全)
                self.layout_pointcloud += layout_pcd.voxel_down_sample(
                    voxel_size=self.cfg.layout_voxel_size
                )
            logger.info(
                f"[Detector][Layout] Initialized layout pointcloud with {len(self.layout_pointcloud.points)} points."
            )
            return

        # Print current frame index
        logger.info(f"[Detector][Layout] Processing frame idx: {self.curr_data.idx}")

        # Check if layout_pointcloud needs to be updated
        if self.check_keyframe_for_layout_pcd():
            start_time = time.time()

            # Generate current frame point cloud(生成当前帧点云)
            current_pcd = self.depth_to_point_cloud(sample_rate=16)

            # Merge point clouds(合并点云)
            with self.layout_lock:
                self.layout_pointcloud += current_pcd
                logger.info(
                    f"[Detector][Layout] Points before downsample: {len(self.layout_pointcloud.points)}"
                )
                self.layout_pointcloud = self.layout_pointcloud.voxel_down_sample(
                    voxel_size=self.cfg.layout_voxel_size
                )
                logger.info(
                    f"[Detector][Layout] Points after downsample: {len(self.layout_pointcloud.points)}"
                )

            # Update prev_keyframe_data
            self.prev_keyframe_data = self.curr_data.copy()
            logger.info("[Detector][Layout] Updated layout pointcloud.(已更新布局点云)")

            # Update time and count(更新时间和计数)
            end_time = time.time()
            layout_time = end_time - start_time
            self.layout_time += layout_time
            self.layout_num += 1
            logger.info(
                f"[Detector][Layout] Layout update took {layout_time:.4f} seconds."
            )

    def update_state(self) -> None:
        """更新检测器状态，清空当前结果和观测。"""
        self.curr_results = {}
        self.curr_observations = []
        # self.prev_data = self.curr_data.copy()
        # self.curr_data.clear()

    def update_data(self) -> None:
        """更新数据，将当前数据复制到前一数据。"""
        # self.curr_results = {}
        # self.curr_observations = []
        # FIXME: 这里是不是应该用深拷贝啊
        self.prev_data = self.curr_data.copy()

        # self.curr_data.clear()

    def get_layout_pointcloud(self):
        """
        Return the current layout_pointcloud.
        """
        with self.layout_lock:
            return self.layout_pointcloud

    def save_layout(self):
        """保存布局点云到带时间戳的目录。"""
        if self.layout_pointcloud is not None:
            layout_pcd = self.get_layout_pointcloud()

            # Create timestamped directory
            map_save_path = self.cfg.map_save_path
            if os.path.exists(map_save_path):
                shutil.rmtree(map_save_path)
                logger.info(f"[Detector] Cleared the directory: {map_save_path}")
            os.makedirs(map_save_path)

            layout_pcd_path = os.path.join(map_save_path, "layout.pcd")
            o3d.io.write_point_cloud(layout_pcd_path, layout_pcd)
            logger.info(f"[Detector][Layout] Saving layout to: {layout_pcd_path}")

    def load_layout(self):
        """
        加载布局点云 layout.pcd, 优先使用 preload_path,
        如果不存在, 则使用 map_save_path。如果路径或文件丢失, 则跳过加载。
        """
        # 优先使用 preload_layout_path，如果不存在则使用 map_save_path
        if os.path.exists(self.cfg.preload_path):
            load_dir = self.cfg.preload_path
            logger.info(f"[Detector][Layout] 使用预加载布局路径: {load_dir}")
        else:
            load_dir = self.cfg.map_save_path
            logger.info(
                f"[Detector][Layout] 未找到预加载布局路径。使用默认地图保存路径: {load_dir}"
            )

        # Build layout point cloud file path
        layout_pcd_path = os.path.join(load_dir, "layout.pcd")

        # Check if layout point cloud file exists
        if not Path(layout_pcd_path).is_file():
            logger.info(
                f"[Detector][Layout] Layout file not found at: {layout_pcd_path}"
            )
            return None

        # Load layout point cloud(加载布局点云)
        layout_pcd = o3d.io.read_point_cloud(layout_pcd_path)
        logger.info(f"[Detector][Layout] Layout loaded from: {layout_pcd_path}")

        # Save to class attribute
        self.layout_pointcloud = layout_pcd

    def get_curr_data(
        self,
    ) -> DataInput:
        """获取当前数据输入。"""
        return self.curr_data

    def get_curr_observations(self):
        """获取当前观测。"""
        return self.curr_observations

    def check_keyframe_for_layout_pcd(self):
        """
        Check if the current frame should be selected as a keyframe based on
        time interval, pose difference (translation), and rotation difference.
        (根据时间间隔、位姿差异（平移）和旋转差异，检查当前帧是否应被选为关键帧。)
        """
        curr_pose = self.curr_data.pose
        prev_keyframe_pose = self.prev_keyframe_data.pose

        # Translation check(平移检查)
        translation_diff = np.linalg.norm(
            curr_pose[:3, 3] - prev_keyframe_pose[:3, 3]
        )  # Translation difference
        if translation_diff >= 1.0:
            logger.info(
                f"[Detector][Layout] 候选帧用于布局计算 -- 平移: {translation_diff}"
            )
            return True

        # Rotation check(旋转检查)
        curr_rotation = R.from_matrix(curr_pose[:3, :3])
        last_rotation = R.from_matrix(prev_keyframe_pose[:3, :3])
        rotation_diff = curr_rotation.inv() * last_rotation
        angle_diff = rotation_diff.magnitude() * (180 / np.pi)

        if angle_diff >= 20:
            logger.info(
                f"[Detector][Layout] 候选帧用于布局计算 -- 旋转: {angle_diff}"
            )
            return True

        return False


    def process_fastsam_results(self, color):
        """处理FastSAM的检测结果。"""
        results = self.fastsam(
            color,
            device="cuda",
            retina_masks=True,
            imgsz=1024,
            conf=self.cfg.fastsam_confidence,
            iou=0.9,
            verbose=False,
        )
        # Extract confidence scores(提取置信度分数)
        confidence_tensor = results[0].boxes.conf
        confidence_np = confidence_tensor.cpu().numpy()

        # Extract bounding box coordinates(提取边界框坐标)
        xyxy_tensor = results[0].boxes.xyxy
        xyxy_np = xyxy_tensor.cpu().numpy()

        # Extract Masks with protection against None(提取掩码，防止为None)
        if results[0].masks is not None:
            masks_tensor = results[0].masks.data
            masks_np = masks_tensor.cpu().numpy().astype(bool)
        else:
            logging.warning(
                "[Detector] fastSAM未返回任何掩码, 使用空掩码数组"
            )
            # If no mask is returned, create an empty array.
            # 假设掩码大小与输入图像的前两个维度匹配
            masks_np = np.empty((0,) + color.shape[:2], dtype=bool)

        # Extract class IDs (default all set to unknown_class_id)
        # 提取类别ID(默认全部设置为unknown_class_id)
        detection_class_id_tensor = results[0].boxes.cls
        detection_class_id_np = detection_class_id_tensor.cpu().numpy().astype(int)
        detection_class_id_np = np.full_like(
            detection_class_id_np, self.unknown_class_id
        )

        return confidence_np, detection_class_id_np, xyxy_np, masks_np

    def merge_detections(self, detections1, detections2):
        """合并两个检测结果。"""
        # Check if first detections is empty(检查第一个检测是否为空)
        if len(detections1.xyxy) == 0:
            return detections2

        # Check if second detections is empty(检查第二个检测是否为空)
        if len(detections2.xyxy) == 0:
            return detections1

        # 合并xyxy
        merged_xyxy = np.concatenate([detections1.xyxy, detections2.xyxy], axis=0)

        # Merge confidence(合并置信度)
        merged_confidence = np.concatenate(
            [detections1.confidence, detections2.confidence], axis=0
        )

        # Merge class_id(合并class_id)
        merged_class_id = np.concatenate(
            [detections1.class_id, detections2.class_id], axis=0
        )

        # Merge mask(合并掩码)
        merged_masks = np.concatenate([detections1.mask, detections2.mask], axis=0)

        # Create new sv.Detections object(创建新的sv.Detections对象)
        merged_detections = sv.Detections(
            xyxy=merged_xyxy,
            confidence=merged_confidence,
            class_id=merged_class_id,
            mask=merged_masks,
        )

        return merged_detections

    def process_fastsam(self, color):
        """运行FastSAM并处理其结果。"""

        with timing_context("FastSAM", self):
            fs_confidence_np, fs_class_id_np, fs_xyxy_np, fs_masks_np = (
                self.process_fastsam_results(color)
            )

        if len(fs_confidence_np) == 0:
            logger.warning("[Detector] FastSAM在当前帧中未找到任何检测")
            self.fastsam_detections = {}
            return

        # debug fastsam(调试fastsam)
        fs_detections = sv.Detections(
            xyxy=fs_xyxy_np,
            confidence=fs_confidence_np,
            class_id=fs_class_id_np,
            mask=fs_masks_np,
        )

        if self.cfg.visualize_detection and self.cfg.show_fastsam_debug:
            image_fs, _ = visualize_result_rgb(
                color, fs_detections, self.obj_classes.get_classes_arr()
            )

            self.annotated_image_fs = image_fs

        self.fastsam_detections = fs_detections

    def process_yolo_and_sam(self, color):
        """运行YOLO和SAM并处理它们的结果。"""
        with timing_context("YOLO", self):
            confidence, class_id, class_labels, xyxy = self.process_yolo_results(
                color, self.obj_classes
            )

        # if detection is empty, return(如果检测为空，则返回)
        if len(confidence) == 0:
            logger.warning("[Detector] 当前帧中未找到任何检测。")
            # 将当前结果设置为空字典
            self.curr_results = {}
            return
        with timing_context("Segmentation", self):
            sam_out = self.sam.predict(color, bboxes=xyxy, verbose=False)
            masks_tensor = sam_out[0].masks.data
            masks_np = masks_tensor.cpu().numpy()
            self.masks_np = masks_np

        curr_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            mask=masks_np,
        )

        self.curr_detections = curr_detections

    def process_yolo_results(self, color, obj_classes):
        """_summary_

        Args:
            color (_type_): _description_
            obj_classes (_type_): _description_

        Returns:
            confidence_np: 提取置信度分数
            detection_class_id_np: 类别id
            detection_class_labels: 类别标签
            xyxy_np: 边界框坐标
        """

        # Perform YOLO prediction(执行YOLO预测)
        results = self.yolo.predict(color, conf=0.2, verbose=False)

        # Extract confidence scores(提取置信度分数)
        confidence_tensor = results[0].boxes.conf
        confidence_np = confidence_tensor.cpu().numpy()

        # Extract class IDs(提取类别ID)
        detection_class_id_tensor = results[0].boxes.cls
        detection_class_id_np = detection_class_id_tensor.cpu().numpy().astype(int)

        # Generate class labels(生成类别标签)
        detection_class_labels = [
            f"{obj_classes.get_classes_arr()[class_id]} {class_idx}"
            for class_idx, class_id in enumerate(detection_class_id_np)
        ]

        # Extract bounding box coordinates(提取边界框坐标)
        xyxy_tensor = results[0].boxes.xyxy
        xyxy_np = xyxy_tensor.cpu().numpy()

        return confidence_np, detection_class_id_np, detection_class_labels, xyxy_np

    def filter_fs_detections_by_curr(
        self, fs_detections, curr_detections, iou_threshold=0.5, overlap_threshold=0.6
    ):
        """根据与当前检测的重叠来过滤FastSAM的检测结果。"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert numpy arrays to torch tensors and move to GPU
        # 将numpy数组转换为torch张量并移动到GPU
        fs_masks = torch.tensor(
            fs_detections.mask, dtype=torch.bool, device=device
        )  # (N1, H, W)
        fs_xyxy = torch.tensor(
            fs_detections.xyxy, dtype=torch.float32, device=device
        )  # (N1, 4)
        fs_confidence = torch.tensor(
            fs_detections.confidence, dtype=torch.float32, device=device
        )
        fs_class_id = torch.tensor(
            fs_detections.class_id, dtype=torch.int64, device=device
        )

        curr_masks = torch.tensor(
            curr_detections.mask, dtype=torch.bool, device=device
        )  # (N2, H, W)

        # Get total number of pixels in masks(获取掩码中的总像素数)
        num_fs = fs_masks.shape[0]  # N1
        num_curr = curr_masks.shape[0]  # N2

        # Flatten masks to (N, H * W) and convert to float32 for matrix multiplication
        # 将掩码展平为 (N, H * W) 并转换为float32以进行矩阵乘法
        fs_masks_flat = fs_masks.view(num_fs, -1).to(torch.float32)  # (N1, H * W)
        curr_masks_flat = curr_masks.view(num_curr, -1).to(torch.float32)  # (N2, H * W)

        # Compute intersection and union (using float operations)
        # 计算交集和并集（使用浮点运算）
        intersection = torch.matmul(fs_masks_flat, curr_masks_flat.T)  # (N1, N2)
        fs_area = fs_masks_flat.sum(dim=1, keepdim=True)  # (N1, 1)
        curr_area = curr_masks_flat.sum(dim=1).unsqueeze(0)  # (1, N2)
        union = fs_area + curr_area - intersection  # (N1, N2)

        # Compute IoU
        # 计算IoU
        iou_matrix = intersection / torch.clamp(union, min=1e-7)  # (N1, N2)

        # Compute overlap ratio
        # 计算重叠率
        overlap_ratio_fs = intersection / torch.clamp(fs_area, min=1e-7)  # (N1, N2)
        overlap_ratio_curr = intersection / torch.clamp(curr_area, min=1e-7)  # (N1, N2)

        # Initialize keep mask, default is to keep all fs_masks
        # 初始化保留掩码，默认为保留所有fs_masks
        keep_mask = torch.ones(num_fs, dtype=torch.bool, device=device)

        # Filter masks one by one
        # 逐个过滤掩码
        for i in range(num_fs):
            # Check if current fs_mask overlaps with curr_mask
            # 检查当前fs_mask是否与curr_mask重叠
            overlap = (
                (iou_matrix[i] > iou_threshold)
                | (overlap_ratio_fs[i] > overlap_threshold)
                | (overlap_ratio_curr[i] > overlap_threshold)
            )

            # If overlap exists, mark as not keep
            # 如果存在重叠，则标记为不保留
            if overlap.any():
                keep_mask[i] = False

        # Filter detections based on keep mask
        # 根据保留掩码过滤检测结果
        filtered_fs_detections = sv.Detections(
            xyxy=fs_xyxy[keep_mask].cpu().numpy(),
            confidence=fs_confidence[keep_mask].cpu().numpy(),
            class_id=fs_class_id[keep_mask].cpu().numpy(),
            mask=fs_masks[keep_mask].cpu().numpy(),
        )

        return filtered_fs_detections

    def add_extra_detections_from_fastsam(
        self, color, fastsam_detections, incoming_detections
    ):
        """从FastSAM添加额外的检测结果。"""

        with timing_context("mask_filter", self):
            fs_after_detections = self.filter_fs_detections_by_curr(
                fastsam_detections, incoming_detections
            )

        if self.cfg.visualize_detection and self.cfg.show_fastsam_debug:
            image_fs_after, _ = visualize_result_rgb(
                color, fs_after_detections, self.obj_classes.get_classes_arr()
            )
            self.annotated_image_fs_after = image_fs_after

        # merge_detctions(合并检测结果)
        merged_detctions = self.merge_detections(
            fs_after_detections, incoming_detections
        )
        return merged_detctions

    def process_detections(self):
        """处理所有检测, 包括YOLO, SAM, FastSAM, 和 CLIP """

        color = self.curr_data.color.astype(np.uint8)

        with timing_context("YOLO+Segmentation+FastSAM", self):
            # Run FastSAM
            if self.cfg.use_fastsam:
                fastsam_thread = threading.Thread(
                    target=self.process_fastsam, args=(color,)
                )
                fastsam_thread.start()

            # Run YOLO and SAM
            self.process_yolo_and_sam(color)

            # Waiting for FastSAM to finish
            if self.cfg.use_fastsam:
                fastsam_thread.join()

        with timing_context("Detection Filter", self):
            self.filter.update_detections(self.curr_detections, color)
            filtered_detections = self.filter.run_filter()

        if self.filter.get_len() == 0:
            logger.warning(
                "[Detector] 过滤后当前帧中没有有效的检测结果。"
            )
            self.curr_results = {}
            return

        # add extra detections from FastSAM results
        # if no detection from fastsam, just skip
        # 从FastSAM结果中添加额外的检测
        # 如果fastsam没有检测结果，则跳过
        if self.cfg.use_fastsam and self.fastsam_detections:
            filtered_detections = self.add_extra_detections_from_fastsam(
                color, self.fastsam_detections, filtered_detections
            )

        with timing_context("CLIP+Create Object Pointcloud", self):
            cluster_thread = threading.Thread(
                target=self.process_masks_thread, args=(filtered_detections.mask,)
            )
            cluster_thread.start()

            with timing_context("CLIP", self):
                image_crops, image_feats, text_feats = (
                    self.compute_clip_features_batched(
                        color,
                        filtered_detections,
                        self.clip_model,
                        self.clip_tokenizer,
                        self.clip_preprocess,
                        self.cfg.device,
                        self.obj_classes.get_classes_arr(),
                    )
                )

            cluster_thread.join()

        results = {
            # SAM Info
            "xyxy": filtered_detections.xyxy,
            "confidence": filtered_detections.confidence,
            "class_id": filtered_detections.class_id,
            "masks": filtered_detections.mask,
            # CLIP info
            "image_feats": image_feats,
            "text_feats": text_feats,
        }

        if self.cfg.visualize_detection:
            with timing_context("Visualize Detection", self):
                annotated_image, _ = visualize_result_rgb(
                    color, filtered_detections, self.obj_classes.get_classes_arr()
                )
                self.annotated_image = annotated_image

        self.curr_results = results

    def process_masks_thread(self, masks):
        """
        Processes the given masks to extract and refine 3D points and colors.
        处理给定的掩码以提取和优化点云和颜色

        Args:
            masks: A NumPy array of shape (N, H, W), where N is the number of masks.

        Returns:
            refined_points_list: A list of refined 3D points for each mask.
            refined_colors_list: A list of refined colors corresponding to the points for each mask.
            refined_points_list: 每个掩码的优化后的点云列表
            refined_colors_list: 每个掩码的对应的点云优化后的颜色列表
        """

        with timing_context("Create Object Pointcloud", self):
            N, _, _ = masks.shape

            # Convert input data to tensors(将输入数据转换为张量)
            depth_tensor = (
                torch.from_numpy(self.curr_data.depth)
                .to(self.cfg.device)
                .float()
                .squeeze()
            )
            masks_tensor = torch.from_numpy(masks).to(self.cfg.device).float()
            intrinsic_tensor = (
                torch.from_numpy(self.curr_data.intrinsics).to(self.cfg.device).float()
            )
            image_rgb_tensor = (
                torch.from_numpy(self.curr_data.color).to(self.cfg.device).float()
                / 255.0
            )

            # Generate 3D points and colors for the masks(为掩码生成3D点和颜色)
            points_tensor, colors_tensor = mask_depth_to_points(
                depth_tensor,
                image_rgb_tensor,
                intrinsic_tensor,
                masks_tensor,
                self.cfg.device,
            )

            refined_points_list = []
            refined_colors_list = []

            # Process each mask(处理每个掩码)
            for i in range(N):
                mask_points = points_tensor[i]
                mask_colors = colors_tensor[i]

                # Filter valid points based on Z-axis > 0(根据Z轴 > 0过滤有效点)
                valid_points_mask = mask_points[:, :, 2] > 0

                if torch.sum(valid_points_mask) < self.cfg.min_points_threshold:
                    refined_points_list.append(None)
                    refined_colors_list.append(None)
                    continue

                valid_points = mask_points[valid_points_mask]
                valid_colors = mask_colors[valid_points_mask]

                # Random sampling based on sample ratio(根据采样率进行随机采样)
                sample_ratio = self.cfg.pcd_sample_ratio
                num_points = valid_points.shape[0]

                if sample_ratio < 1.0:
                    sample_count = int(num_points * sample_ratio)
                    sample_indices = torch.randperm(num_points)[:sample_count]
                    downsampled_points = valid_points[sample_indices]
                    downsampled_colors = valid_colors[sample_indices]
                else:
                    downsampled_points = valid_points
                    downsampled_colors = valid_colors

                # Refine points using clustering(使用聚类优化点)
                refined_points, refined_colors = refine_points_with_clustering(
                    downsampled_points,
                    downsampled_colors,
                    eps=self.cfg.dbscan_eps,
                    min_points=self.cfg.dbscan_min_points,
                )

                refined_points_list.append(refined_points)
                refined_colors_list.append(refined_colors)

        self.masked_points = refined_points_list
        self.masked_colors = refined_colors_list

    def compute_max_cos_sim(self, image_feats, class_feats):
        """
        Compute the cosine similarity between image_feats and class_feats, and return the class index with the maximum similarity for each image_feat.


        Args:
            image_feats (np.ndarray): 所有当前图像的CLIP特征, 形状为 (N, 512)
            class_feats (np.ndarray): 类别CLIP特征, 形状为 (C, 512)

        Returns:
            max_indices (np.ndarray): 每个image_feat具有最大余弦相似度的类别索引, 形状为 (N,)
        """
        # Normalize the features to compute cosine similarity
        # 归一化特征以计算余弦相似度
        image_feats_norm = image_feats / np.linalg.norm(
            image_feats, axis=1, keepdims=True
        )
        # Normalize image_feats
        # 归一化image_feats
        class_feats_norm = class_feats / np.linalg.norm(
            class_feats, axis=1, keepdims=True
        )
        # Normalize class_feats
        # 归一化class_feats
        # 计算余弦相似度: (N, 512) @ (512, C) -> (N, C)
        cos_sim = np.dot(image_feats_norm, class_feats_norm.T)

        # 找到每个图像的最大相似度索引
        max_indices = np.argmax(cos_sim, axis=1)  # shape (N,)

        return max_indices

    def depth_to_point_cloud(self, sample_rate=1) -> o3d.geometry.PointCloud:
        """
        Convert depth image to a point cloud and transform it to world coordinates.
        将深度图像转换为点云并将其转换到世界坐标系。

        Parameters:
        - sample_rate: The downsampling rate for pixel selection.
        (1 means all pixels, 2 means every other pixel)
        参数:
        - sample_rate: 像素选择的下采样率。(1表示所有像素, 2表示每隔一个像素)

        Returns:
        - point_cloud: The point cloud in world coordinates as an Open3D PointCloud object.
        返回:
        - point_cloud: 世界坐标系中的点云, 作为Open3D PointCloud对象
        """
        # Extract necessary data from curr_data(从curr_data中提取必要数据)
        depth = self.curr_data.depth.squeeze(
            -1
        )
        # Remove the last dimension if depth is (H, W, 1)
        # 如果深度是(H, W, 1)，则移除最后一个维度
        intrinsics = self.curr_data.intrinsics
        pose = self.curr_data.pose

        # Mask out invalid depth values (e.g., depth = 0 or NaN)
        # 屏蔽无效深度值（例如，深度=0或NaN）
        valid_mask = (depth > 0) & (
            depth != np.inf
        )
        # Create a mask for valid depth values
        # 创建有效深度值的掩码
        depth = depth[valid_mask]  # Only keep valid depth values

        # Get the corresponding u, v coordinates for valid pixels
        # 获取有效像素对应的u, v坐标
        height, width = self.curr_data.depth.shape[:2]
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        u = u[valid_mask]
        v = v[valid_mask]

        # Downsample the points if needed (sampling every `sample_rate` pixels)
        # 如果需要，对点进行下采样（每`sample_rate`个像素采样一次）
        u = u[::sample_rate]
        v = v[::sample_rate]
        depth = depth[::sample_rate]

        # Use the intrinsic matrix to convert from pixel coordinates to camera coordinates (X, Y, Z)
        # 使用内参矩阵从像素坐标转换为相机坐标 (X, Y, Z)
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        # Convert from pixel coordinates to normalized camera coordinates
        # 从像素坐标转换为归一化相机坐标
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        # Stack the coordinates to form the point cloud in the camera coordinate system
        # 堆叠坐标以形成相机坐标系中的点云
        points_camera = np.vstack((x, y, z)).T

        # Convert points to homogeneous coordinates (4D) for transformation
        # 将点转换为齐次坐标 (4D) 以进行变换
        points_homogeneous = np.hstack(
            (points_camera, np.ones((points_camera.shape[0], 1)))
        )

        # Apply the pose transformation to move points to world coordinates
        # 应用位姿变换将点移动到世界坐标
        points_world_homogeneous = (pose @ points_homogeneous.T).T

        # Discard the homogeneous coordinate (last column) to get the final 3D points in world coordinates
        # 丢弃齐次坐标（最后一列）以获得世界坐标中的最终3D点
        points_world = points_world_homogeneous[:, :3]

        # Create a PointCloud object and set its points
        # 创建一个PointCloud对象并设置其点
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_world)

        return point_cloud

    def save_detection_results(
        self,
    ) -> None:
        """保存检测结果到磁盘。"""

        if self.curr_results == {}:
            logger.error("[Detector] 没有检测结果，无需保存")
            return

        output_det_path = self.detection_path / self.curr_data.color_name
        output_det_path.mkdir(exist_ok=True, parents=True)

        # save results(保存结果)
        for key, value in self.curr_results.items():
            save_path = Path(output_det_path) / f"{key}"
            if isinstance(value, np.ndarray):
                # Save NumPy arrays using .npz for efficient storage
                # (使用.npz保存NumPy数组以实现高效存储)
                np.savez_compressed(f"{save_path}.npz", value)
            else:
                # For other types, fall back to pickle
                # 对于其他类型，回退到pickle
                with gzip.open(f"{save_path}.pkl.gz", "wb") as f:
                    pickle.dump(value, f)

        # save annotated images(保存带注释的图像)
        output_file_path = (
            self.detection_path / "vis" / (self.curr_data.color_name + "_annotated.jpg")
        )
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        image = cv2.cvtColor(self.curr_data.color, cv2.COLOR_BGR2RGB)

        detections = sv.Detections(
            xyxy=self.curr_results["xyxy"],
            confidence=self.curr_results["confidence"],
            class_id=self.curr_results["class_id"],
            mask=self.curr_results["masks"],
        )
        annotated_image, _ = visualize_result_rgb(
            image, detections, self.obj_classes.get_classes_arr()
        )

        self.annotated_image = annotated_image

        cv2.imwrite(str(output_file_path), annotated_image)

    def load_detection_results(
        self,
    ):
        """从磁盘加载检测结果。"""
        det_path = self.detection_path / self.curr_data.color_name

        det_path = Path(det_path)

        # 如果当前帧在磁盘上没有检测结果，则返回空字典
        if not det_path.exists():
            logger.error(f"[Detector] 在 {det_path} 中未找到检测结果")
            self.curr_results = {}
            return

        # 从磁盘加载结果
        logger.info(f"[Detector] 从 {det_path} 加载检测结果")

        loaded_detections = {}

        for file_path in det_path.iterdir():
            # handle the files with their extensions
            # 根据文件扩展名处理文件
            if file_path.suffix == ".gz" and file_path.suffixes[-2] == ".pkl":
                key = file_path.name.replace(".pkl.gz", "")
                with gzip.open(file_path, "rb") as f:
                    loaded_detections[key] = pickle.load(f)
            elif file_path.suffix == ".npz":
                loaded_detections[file_path.stem] = np.load(file_path)["arr_0"]
            elif file_path.suffix == ".jpg":
                continue
            else:
                raise ValueError(f"{file_path} is not a .pkl.gz or .npz file!")

        self.curr_results = loaded_detections

    def calculate_observations(
        self,
    ) -> None:
        """根据当前检测结果计算观测。"""
        # 如果没有检测结果，直接返回
        if not self.curr_results:
            logger.warning("[Detector] 没有检测结果，无法计算观测")
            self.curr_observations = []
            return

        # Traverse all the detections(遍历所有检测结果)
        N, _, _ = self.curr_results["masks"].shape

        # for debugging only(仅用于调试)
        # bbox_hl_mapping = []
        for i in range(N):

            if self.masked_points[i] is None:
                continue

            # Create pointcloud(创建点云)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.masked_points[i])
            pcd.colors = o3d.utility.Vector3dVector(self.masked_colors[i])
            pcd.transform(self.curr_data.pose)

            # Get bbox(获取包围盒)
            bbox = safe_create_bbox(pcd)
            # bbox = pcd.get_axis_aligned_bounding_box()

            if self.cfg.filter_ceiling:
                z = bbox.get_center()[2]

                # check z is close to ceiling_height(检查z是否接近天花板高度)
                if abs(z - self.cfg.ceiling_height) < self.cfg.ceiling_threshold:  # 0.1
                    # If z close ceiling_height， skip this observation
                    # 如果z接近天花板高度，跳过此观测
                    continue

            # Get Mobility(获取移动性)
            # class_name = self.obj_classes.get_classes_arr()[self.curr_results['class_id'][i]]
            # Get distance(获取距离)
            distance = self.get_distance(bbox, self.curr_data.pose)

            # Init observation(初始化观测)
            curr_obs = LocalObservation()

            # Set observation info(设置观测信息)
            curr_obs.idx = self.curr_data.idx
            curr_obs.class_id = self.curr_results["class_id"][i]
            curr_obs.mask = self.curr_results["masks"][i]

            curr_obs.xyxy = self.curr_results["xyxy"][i]
            curr_obs.conf = self.curr_results["confidence"][i]

            if self.cfg.use_weighted_feature:
                curr_obs.clip_ft = self.get_weighted_feature(idx=i)
            else:
                curr_obs.clip_ft = self.curr_results["image_feats"][i]

            curr_obs.pcd = pcd
            curr_obs.bbox = bbox
            curr_obs.distance = distance

            # judge if low mobility according to the clip feature
            # 根据clip特征判断是否为低移动性
            curr_obs.is_low_mobility = self.is_low_mobility(
                curr_obs.clip_ft
            )  # , hl_debug, hl_idx
            # for debugging only(仅用于调试)
            # bbox_hl_mapping.append([self.curr_results['xyxy'][i], hl_debug, hl_idx])

            # if curr_obs classid is desk set as low mobility
            # 如果当前观测的类别ID是desk，则设置为低移动性
            if (
                self.obj_classes.get_classes_arr()[curr_obs.class_id]
                in self.cfg.lm_examples
            ):
                curr_obs.is_low_mobility = True

            if self.cfg.save_cropped:
                whole_image = self.curr_data.color

                # crop image by xyxy
                # 根据xyxy裁剪图像
                x1, y1, x2, y2 = map(int, curr_obs.xyxy)
                cropped_image = whole_image[y1:y2, x1:x2]
                cropped_mask = curr_obs.mask[y1:y2, x1:x2].astype(np.uint8) * 255

                masked_image = cv2.bitwise_and(
                    cropped_image, cropped_image, mask=cropped_mask
                )

                curr_obs.masked_image = masked_image
                curr_obs.cropped_image = cropped_image

            # Add observation to the list(将观测添加到列表)
            self.curr_observations.append(curr_obs)

        logger.info(
            f"[Detector] cur observeation num: {len(self.curr_observations)}"
        )

    def get_weighted_feature(self, idx):
        """获取加权特征。"""
        image_feat = self.curr_results["image_feats"][idx]
        text_feat = self.curr_results["text_feats"][idx]

        w_image = self.cfg.image_weight
        w_text = 1 - w_image

        weighted_feature = w_image * image_feat + w_text * text_feat

        norm = np.linalg.norm(weighted_feature)
        if norm > 0:
            weighted_feature /= norm

        return weighted_feature

    def visualize_time(
        self,
        elapsed_time,
    ) -> None:
        """可视化时间"""
        logger.info(f"[Detector][Visualize] Elapsed time: {elapsed_time:.4f} seconds")
        self.visualizer.log(
            "plot_time/frame_elapsed_time",
            self.visualizer.Scalar(elapsed_time),
            self.visualizer.SeriesLine(width=2.5, color=[255, 0, 0]),  # Red color
        )

    def visualize_memory(
        self,
        memory_usage,
    ) -> None:
        """可视化内存使用情况。"""
        logger.info(f"[Detector][Visualize] Memory usage: {memory_usage:.2f} MB")
        self.visualizer.log(
            "plot_memory/memory_usage",
            self.visualizer.Scalar(memory_usage),
            self.visualizer.SeriesLine(width=2.5, color=[0, 255, 0]),  # Green color
        )

    def visualize_detection(
        self,
    ) -> None:
        """可视化检测结果。"""

        if self.annotated_image is not None:
            self.visualizer.log(
                "world/camera/rgb_image_annotated",
                self.visualizer.Image(self.annotated_image),
            )

        if self.cfg.show_local_entities:
            self.visualizer.log(
                "world/camera_raw/rgb_image",
                self.visualizer.Image(self.curr_data.color),
            )

            if self.annotated_image_fs is not None:
                self.visualizer.log(
                    "world/camera_fs/rgb_image_annotated",
                    self.visualizer.Image(self.annotated_image_fs),
                )

            if self.annotated_image_fs_after is not None:
                self.visualizer.log(
                    "world/camera_fs_after/rgb_image_annotated",
                    self.visualizer.Image(self.annotated_image_fs_after),
                )

        # Visualize camera traj(可视化相机轨迹)
        focal_length = [
            self.curr_data.intrinsics[0, 0].item(),
            self.curr_data.intrinsics[1, 1].item(),
        ]
        principal_point = [
            self.curr_data.intrinsics[0, 2].item(),
            self.curr_data.intrinsics[1, 2].item(),
        ]
        height, width = self.curr_data.color.shape[:2]
        resolution = [width, height]
        self.visualizer.log(
            "world/camera",
            self.visualizer.Pinhole(
                resolution=resolution,
                focal_length=focal_length,
                principal_point=principal_point,
            ),
        )

        translation = self.curr_data.pose[:3, 3].tolist()

        # change the rotation mat to axis-angle(将旋转矩阵转换为轴角)
        axis, angle = self.visualizer.rotation_matrix_to_axis_angle(
            self.curr_data.pose[:3, :3]
        )
        self.visualizer.log(
            "world/camera",
            self.visualizer.Transform3D(
                translation=translation,
                rotation=self.visualizer.RotationAxisAngle(axis=axis, angle=angle),
                from_parent=False,
            ),
        )

        # follower camera for recording
        # Visualize camera traj
        # 用于录制的跟随相机
        # 可视化相机轨迹

        f = [
            self.curr_data.intrinsics[0, 0].item(),
            self.curr_data.intrinsics[1, 1].item(),
        ]
        p = [
            self.curr_data.intrinsics[0, 2].item(),
            self.curr_data.intrinsics[1, 2].item(),
        ]
        h, w = self.curr_data.color.shape[:2]
        r = [w, h]
        self.visualizer.log(
            "world/follower_camera",
            self.visualizer.Pinhole(resolution=r, focal_length=f, principal_point=p),
        )
        self.visualizer.log(
            "world/follower_camera_2",
            self.visualizer.Pinhole(resolution=r, focal_length=f, principal_point=p),
        )

        pose_current = self.curr_data.pose
        cam2_to_cam1 = self.create_camera2_to_camera1_transform()
        cam2_to_cam1_2 = self.create_camera2_to_camera1_transform2()
        pose_new = pose_current @ cam2_to_cam1
        pose_new_2 = pose_current @ cam2_to_cam1_2

        translation = pose_new[:3, 3].tolist()
        # change the rotation mat to axis-angle(将旋转矩阵转换为轴角)
        axis, angle = self.visualizer.rotation_matrix_to_axis_angle(pose_new[:3, :3])
        self.visualizer.log(
            "world/follower_camera",
            self.visualizer.Transform3D(
                translation=translation,
                rotation=self.visualizer.RotationAxisAngle(axis=axis, angle=angle),
                from_parent=False,
            ),
        )

        # Using for visualization(用于可视化)
        pose_smooth = self.pose_filter_follower.update(pose_new_2)
        translation = pose_smooth[:3, 3].tolist()
        axis, angle = self.visualizer.rotation_matrix_to_axis_angle(pose_smooth[:3, :3])

        self.visualizer.log(
            "world/follower_camera_2",
            self.visualizer.Transform3D(
                translation=translation,
                rotation=self.visualizer.RotationAxisAngle(axis=axis, angle=angle),
                from_parent=False,
            ),
        )

        if self.prev_data is not None:
            prev_translation = self.prev_data.pose[:3, 3].tolist()
            prev_quaternion = self.visualizer.rotation_matrix_to_quaternion(
                self.prev_data.pose[:3, :3]
            )

            # # Log a line strip from the previous to the current camera pose
            # # 记录从前一个相机位姿到当前相机位姿的线段
            # self.visualizer.log(
            #     f"world/camera_trajectory/{self.curr_data.idx}",
            #     self.visualizer.LineStrips3D(
            #         [np.vstack([prev_translation, translation]).tolist()],
            #         colors=[[255, 0, 0]]  # Red color for the trajectory line
            #         colors=[[255, 0, 0]]  # 轨迹线为红色
            #     )
            # )

        if self.cfg.show_debug_entities:
            layout_pointcloud = self.get_layout_pointcloud()
            positions = layout_pointcloud.points
            pcd_entity = "world/layout"
            self.visualizer.log(pcd_entity, self.visualizer.Points3D(positions))

    def create_camera2_to_camera1_transform(self):
        """创建从相机2到相机1的变换矩阵。"""
        # 定义平移向量：相机2相对于相机1的平移
        # Define translation vector: translation of camera 2 relative to camera 1
        translation = np.array(self.cfg.follower_translation)  # Up 0.2m, back -0.2m

        # Rotation angles in degrees(旋转角度/度)
        angle_roll = self.cfg.follower_roll  # Rotation around X axis(绕x轴旋转)
        angle_pitch = self.cfg.follower_pitch  # Rotation around Y axis(绕y轴旋转)
        angle_yaw = self.cfg.follower_yaw  # Rotation around Z axis(绕z轴旋转)

        # Convert angles to radians(将角度转换为弧度)
        angle_roll_rad = np.radians(angle_roll)
        angle_pitch_rad = np.radians(angle_pitch)
        angle_yaw_rad = np.radians(angle_yaw)

        # 绕X轴的旋转矩阵（roll）
        rotation_roll = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle_roll_rad), -np.sin(angle_roll_rad)],
                [0, np.sin(angle_roll_rad), np.cos(angle_roll_rad)],
            ]
        )

        # 绕Y轴的旋转矩阵（pitch）
        rotation_pitch = np.array(
            [
                [np.cos(angle_pitch_rad), 0, np.sin(angle_pitch_rad)],
                [0, 1, 0],
                [-np.sin(angle_pitch_rad), 0, np.cos(angle_pitch_rad)],
            ]
        )

        # 绕Z轴的旋转矩阵（yaw）
        rotation_yaw = np.array(
            [
                [np.cos(angle_yaw_rad), -np.sin(angle_yaw_rad), 0],
                [np.sin(angle_yaw_rad), np.cos(angle_yaw_rad), 0],
                [0, 0, 1],
            ]
        )

        # 组合旋转矩阵：顺序为Z, Y, X
        rotation_matrix = rotation_yaw @ rotation_pitch @ rotation_roll

        # 创建4x4变换矩阵
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix  # Fill rotation matrix(填充旋转矩阵)
        transform[:3, 3] = translation  # Fill translation vector(创建平移向量)

        return transform

    def create_camera2_to_camera1_transform2(self):
        """创建从相机2到相机1的第二个变换矩阵。"""
        # 定义平移向量：相机2相对于相机1的平移
        # Define translation vector: translation of camera 2 relative to camera 1
        translation = np.array(self.cfg.follower_translation2)  # Up 0.2m, back -0.2m

        # Rotation angles in degrees(旋转角度/度)
        angle_roll = self.cfg.follower_roll2  # Rotation around X axis
        angle_pitch = self.cfg.follower_pitch2  # Rotation around Y axis
        angle_yaw = self.cfg.follower_yaw2  # Rotation around Z axis

        # Convert angles to radians(将角度转换为弧度)
        angle_roll_rad = np.radians(angle_roll)
        angle_pitch_rad = np.radians(angle_pitch)
        angle_yaw_rad = np.radians(angle_yaw)

        # Rotation matrix around X axis (roll)
        rotation_roll = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle_roll_rad), -np.sin(angle_roll_rad)],
                [0, np.sin(angle_roll_rad), np.cos(angle_roll_rad)],
            ]
        )

        # Rotation matrix around Y axis (pitch)
        rotation_pitch = np.array(
            [
                [np.cos(angle_pitch_rad), 0, np.sin(angle_pitch_rad)],
                [0, 1, 0],
                [-np.sin(angle_pitch_rad), 0, np.cos(angle_pitch_rad)],
            ]
        )

        # Rotation matrix around Z axis (yaw)
        rotation_yaw = np.array(
            [
                [np.cos(angle_yaw_rad), -np.sin(angle_yaw_rad), 0],
                [np.sin(angle_yaw_rad), np.cos(angle_yaw_rad), 0],
                [0, 0, 1],
            ]
        )

        # 组合旋转矩阵：顺序为Z, Y, X
        rotation_matrix = rotation_yaw @ rotation_pitch @ rotation_roll

        # Create 4x4 transformation matrix(创建4x4变换矩阵)
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix  # Fill rotation matrix
        transform[:3, 3] = translation  # Fill translation vector

        return transform

    def is_low_mobility(self, clip_feat) -> bool:
        """判断一个物体是否为低移动性。"""
        # 计算clip特征和原型之间的余弦相似度
        clip_feat = clip_feat.reshape(1, -1)
        sim = cosine_similarity(clip_feat, self.proto_feats)
        sim = sim.reshape(-1)

        sim_lm = np.max(sim[: self.num_examples[0]])
        sim_hm = np.max(
            sim[self.num_examples[0] : (self.num_examples[0] + self.num_examples[1])]
        )
        sim_lm_des = np.max(sim[(self.num_examples[0] + self.num_examples[1]):])

        # for debugging only
        # lm_idx = np.argmax(sim[:self.num_examples[0]])
        # hm_idx = np.argmax(sim[self.num_examples[0] : (self.num_examples[0]+self.num_examples[1])])
        # lm_des_idx = np.argmax(sim[(self.num_examples[0]+self.num_examples[1]):])
        # hl_debug = np.array([sim_lm, sim_hm, sim_lm_des])
        # hl_idx = np.array([lm_idx, hm_idx, lm_des_idx])

        # Use configurable thresholds(使用可配置的阈值)
        similarity_delta = self.cfg.mobility.similarity_delta
        descriptor_threshold = self.cfg.mobility.descriptor_threshold

        if sim_lm > sim_hm + similarity_delta:
            res = True
        elif sim_lm + similarity_delta < sim_hm:
            res = False
        else:
            res = sim_lm_des > descriptor_threshold
        return res  # , hl_debug, hl_idx

    def get_distance(self, bbox, pose) -> float:
        # Get the center of the bounding box
        # 计算包围盒的中心
        bbox_center = np.array(bbox.get_center())

        # Get the translation part of the pose (assuming it's a 4x4 matrix)
        pose_translation = np.array(pose[:3, 3])  # Extract translation (x, y, z)

        # Calculate the Euclidean distance between the pose translation and the bbox center
        # FIXME: 这里是在算啥?
        distance = np.linalg.norm(bbox_center - pose_translation)

        return distance

    def compute_clip_features_batched(
        self,
        image,
        detections,
        clip_model,
        clip_tokenizer,
        clip_preprocess,
        device,
        classes,
    ):
        """批量计算CLIP特征。"""
        # 将图像转换为PIL图像
        image = Image.fromarray(image)

        # Set the padding for cropping(设置裁剪的填充)
        padding = 20

        # Initialize lists to store the cropped images and features(初始化列表以存储裁剪的图像和特征)
        image_crops = []
        image_feats = []
        text_feats = []

        # Initialize lists to store preprocessed images and text tokens for batch processing
        # 初始化列表以存储用于批处理的预处理图像和文本标记
        preprocessed_images = []
        text_tokens = []

        # Prepare data for batch processing
        # 准备批处理数据
        for idx in range(len(detections.xyxy)):
            x_min, y_min, x_max, y_max = detections.xyxy[idx]
            image_width, image_height = image.size

            # Calculate the padding for each side of the bounding box
            # 计算边界框每边的填充
            left_padding = min(padding, x_min)
            top_padding = min(padding, y_min)
            right_padding = min(padding, image_width - x_max)
            bottom_padding = min(padding, image_height - y_max)

            # Adjust the bounding box coordinates based on the padding
            # 根据填充调整边界框坐标
            x_min -= left_padding
            y_min -= top_padding
            x_max += right_padding
            y_max += bottom_padding

            # Crop the image
            # 裁剪图像
            cropped_image = image.crop((x_min, y_min, x_max, y_max))

            # Preprocess the cropped image
            # 预处理裁剪的图像
            preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0)
            preprocessed_images.append(preprocessed_image)

            # Get the class id for the detection
            # 获取检测的类别ID
            class_id = detections.class_id[idx]

            # Append the class name to the text tokens list
            # 将类别名称附加到文本标记列表
            text_tokens.append(classes[class_id])

            # Append the cropped image to the image crops list
            # 将裁剪的图像附加到图像裁剪列表
            image_crops.append(cropped_image)

        # Convert lists to batches
        # 将列表转换为批次
        preprocessed_images_batch = torch.cat(preprocessed_images, dim=0).to(device)
        text_tokens_batch = clip_tokenizer(text_tokens).to(device)

        # Perform batch inference
        # 执行批处理推断
        with torch.no_grad():
            # Encode the images using the CLIP model
            # 使用CLIP模型编码图像
            image_features = clip_model.encode_image(preprocessed_images_batch)

            # Normalize the image features
            # 归一化图像特征
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Encode the text tokens using the CLIP model
            # 使用CLIP模型编码文本标记
            text_features = clip_model.encode_text(text_tokens_batch)

            # Normalize the text features
            # 归一化文本特征
            text_features /= text_features.norm(dim=-1, keepdim=True)

        # Convert the image and text features to numpy arrays
        # 将图像和文本特征转换为numpy数组
        image_feats = image_features.cpu().numpy()
        text_feats = text_features.cpu().numpy()

        if self.cfg.use_avg_feat_for_unknown:
            count = 0
            for idx, class_id in enumerate(detections.class_id):
                if class_id == self.unknown_class_id:
                    count += 1
                    # Modify the text_feats for the unknown class
                    # 修改未知类别的文本特征
                    text_feats[idx] = (
                        self.class_feats_mean
                    )
                    # You can modify how you update the text_feats here
                    # 你可以在这里修改如何更新文本特征
                    # random_feats = np.random.rand(*self.class_feats_mean.shape)
                    # random_feats /=     np.linalg.norm(random_feats)

                    # text_feats[idx] = random_feats

            logger.info(
                f"[Detector] 已将 {count} 个未知类别文本特征更新为平均值。"
            )
        else:
            for idx, class_id in enumerate(detections.class_id):
                if class_id == self.unknown_class_id:
                    count += 1
                    # 修改未知类别的文本特征
                    # Modify the text_feats for the unknown class
                    # text_feats[idx] = self.class_feats_mean  # You can modify how you update the text_feats here

                    random_feats = np.random.rand(*self.class_feats_mean.shape)
                    random_feats /= np.linalg.norm(random_feats)

                    text_feats[idx] = random_feats

        # 返回裁剪的图像、图像特征和文本特征
        # Return the cropped images, image features, and text features
        return image_crops, image_feats, text_feats

class Filter:
    """用于过滤检测结果的类。"""

    def __init__(
        self,
        classes,
        iou_th: float = 0.80,
        proximity_th: float = 0.95,
        keep_larger: bool = True,
        small_mask_size: int = 200,
        skip_refinement: bool = False,
    ):

        self.confidence = None
        self.class_id = None
        self.xyxy = None
        self.masks = None
        self.color = None
        self.masks_size = None
        self.inter_np = None

        self.skip_refinement = skip_refinement
        self.classes = classes
        self.iou_th = iou_th
        self.proximity_th = proximity_th
        self.keep_larger = keep_larger
        self.small_mask_size = small_mask_size

        self.device = "cpu"

    def update_detections(self, detections: sv.Detections, color: np.array):
        """更新检测结果以进行过滤。"""
        with timing_context("update_detections", self):
            self.color = color

            self.confidence = detections.confidence
            self.class_id = detections.class_id
            self.xyxy = detections.xyxy
            self.masks = detections.mask

            self.masks_size = np.sum(self.masks, axis=(1, 2))

            # Compute intersection every time the detections are updated
            # 每次更新检测时计算交集
            N = self.get_len()
            # Convert masks to PyTorch tensors to accelerate computation
            # Compute pairwise intersection using matrix operations
            # 将掩码转换为PyTorch张量以加速计算
            # 使用矩阵运算计算成对交集

            device = self.device
            masks = torch.tensor(self.masks, dtype=torch.float32).to(device)
            intersection = torch.matmul(masks.view(N, -1), masks.view(N, -1).T)

            self.inter_np = intersection.cpu().numpy()

    def set_device(self, device):
        """设置计算设备。"""
        self.device = device

    def run_filter(self):
        """运行所有过滤步骤。"""
        original_num = self.get_len()
        if self.confidence is None or original_num == 0:
            logger.warning("[Detector][Filter] 没有要过滤的检测结果。")
            return

        keep = self.filter_by_mask_size()
        self.set_detections(keep)

        if not self.skip_refinement:
            keep = self.filter_by_iou()
            self.set_detections(keep)

            keep = self.filter_by_proximity()
            self.set_detections(keep)

            self.overlap_check()

        keep = self.filter_by_bg()
        self.set_detections(keep)

        keep = self.filter_by_mask_size()
        self.set_detections(keep)

        if self.get_len() == 0:
            logger.warning(
                "[Detector][Filter] 过滤后，没有检测结果剩下..."
            )
            return None
        logger.info(
            f"[Detector][Filter] 从 {original_num} 个中过滤出 {self.get_len()} 个"
        )

        # create new detections object and return
        # 创建新的检测对象并返回
        filtered_detections = sv.Detections(
            class_id=np.array(self.class_id, dtype=np.int64),
            confidence=np.array(self.confidence, dtype=np.float32),
            xyxy=np.array(self.xyxy, dtype=np.float32),
            mask=np.array(self.masks, dtype=np.bool_),
        )
        return filtered_detections

    def get_len(self):
        """获取检测结果的数量。"""
        return len(self.confidence)

    def filter_by_mask_size(self):
        """根据掩码大小过滤检测结果。"""
        keep = self.masks_size >= self.small_mask_size
        for idx, is_keep in enumerate(keep):
            if not is_keep:
                class_name = self.classes.get_classes_arr()[self.class_id[idx]]
                logger.info(
                    f"[Detector][Filter] 移除 {class_name} 因为掩码太小"
                )
        return keep

    def set_detections(self, keep):
        """根据保留掩码设置检测结果。"""
        if len(keep) != self.get_len():
            logger.warning(
                "[Detector][Filter] The boolean list should be as long as the detections."
            )
            return

        self.confidence, self.class_id, self.xyxy, self.masks, self.masks_size = (
            self.confidence[keep],
            self.class_id[keep],
            self.xyxy[keep],
            self.masks[keep],
            self.masks_size[keep],
        )
        self.inter_np = self.inter_np[keep][:, keep]

    def filter_by_iou(self):
        """根据IoU过滤检测结果。"""
        N = self.get_len()
        # 使用布尔列表来控制
        if N == 0:
            return np.array([], dtype=bool)

        masks = self.masks
        masks_size = self.masks_size

        # Compute pairwise IoU matrix using matrix operations for acceleration
        # 使用矩阵运算加速计算成对IoU矩阵
        intersection = self.inter_np
        area = masks.reshape(N, -1).sum(axis=1)
        union = area[:, None] + area[None, :] - intersection
        iou_matrix = intersection / union

        # Initialize keep mask(初始化保留掩码)
        keep = np.ones(N, dtype=bool)

        # Apply IoU threshold and keep larger/smaller masks(应用IoU阈值并保留较大/较小的掩码)
        for i in range(N):
            if not keep[i]:
                continue
            for j in range(i + 1, N):
                if iou_matrix[i, j] > self.iou_th:
                    if ((masks_size[i] > masks_size[j]) and self.keep_larger) or (
                        (masks_size[i] < masks_size[j]) and not self.keep_larger
                    ):
                        keep[j] = False
                    else:
                        keep[i] = False
                        break

        logger.info(
            f"[Detector][Filter] 原始检测数量: {N}, 掩码IoU过滤后: {np.sum(keep)}"
        )
        return keep

    def filter_by_proximity(self):
        """根据邻近度过滤检测结果。"""
        if self.color is None:
            logger.warning("[Detector][Filter] No color image is provided.")
            return
        N = self.get_len()
        if N == 0:
            return np.array([], dtype=bool)
        # check if mask overlaps with each other(检查掩码是否相互重叠)
        overlap = self.inter_np
        overlap = overlap > 0
        np.fill_diagonal(overlap, False)

        N = self.get_len()
        # Initialize keep mask(初始化保留掩码)
        keep = np.ones(N, dtype=bool)
        masks_size = self.masks_size

        # Get all the cropped images first to accelerate computation
        # 首先获取所有裁剪图像以加速计算
        cropped_images = []
        cropped_masks = []
        for i in range(N):
            x1, y1, x2, y2 = map(int, self.xyxy[i])
            cropped_image = self.color[y1:y2, x1:x2]
            cropped_images.append(cropped_image)
            cropped_mask = self.masks[i][y1:y2, x1:x2].astype(bool)
            cropped_masks.append(cropped_mask)

        for i in range(N):
            if not keep[i]:
                continue
            for j in range(i + 1, N):
                # if overlapped, crop the images and check if they have the same distribution
                # 如果重叠，裁剪图像并检查它们是否具有相同的分布
                if overlap[i, j]:
                    from_same_dis = if_same_distribution(
                        cropped_images[i],
                        cropped_images[j],
                        cropped_masks[i],
                        cropped_masks[j],
                        self.proximity_th,
                    )

                    if from_same_dis:
                        class_i = self.classes.get_classes_arr()[self.class_id[i]]
                        class_j = self.classes.get_classes_arr()[self.class_id[j]]
                        if ((masks_size[i] > masks_size[j]) and self.keep_larger) or (
                            (masks_size[i] < masks_size[j]) and not self.keep_larger
                        ):
                            keep[j] = False
                            self.merge_detections(j, i)
                            logger.info(
                                f"[Detector][Filter] Merging {class_j} into {class_i}"
                            )
                        else:
                            keep[i] = False
                            self.merge_detections(i, j)
                            logger.info(
                                f"[Detector][Filter] Merging {class_i} into {class_j}"
                            )

        logger.info(
            f"[Detector][Filter] Original number of detections: {N}, after proximity filter: {np.sum(keep)}"
        )
        return keep

    def overlap_check(self):
        """检查并处理重叠的掩码。"""
        N = self.get_len()
        if N == 0:
            return
        masks_size = self.masks_size

        # check if mask overlaps with each other(检查掩码是否相互重叠)
        overlap = self.inter_np
        overlap = overlap > 0
        np.fill_diagonal(overlap, False)

        for i in range(N):
            for j in range(i + 1, N):
                if overlap[i, j]:

                    if masks_size[i] > masks_size[j]:
                        self.masks[i] = self.masks[i] & (~self.masks[j])
                        self.xyxy[i] = update_bbox(self.masks[i])
                    else:
                        self.masks[j] = self.masks[j] & (~self.masks[i])
                        self.xyxy[j] = update_bbox(self.masks[j])

    def filter_by_bg(self):
        """根据背景类别过滤检测结果。"""
        N = self.get_len()
        keep = np.ones(N, dtype=bool)

        for idx, class_id in enumerate(self.class_id):
            if self.classes.get_classes_arr()[class_id] in self.classes.bg_classes:
                logger.info(
                    f"[Detector][Filter] Removing {self.classes.get_classes_arr()[class_id]} because it is a background class."
                )
                keep[idx] = False
        return keep

    def merge_detections(self, det, target):
        """将一个检测结果合并到另一个。"""
        # 将det合并到目标检测中
        self.masks[target] = np.logical_or(self.masks[target], self.masks[det])
        x_i1, y_i1, x_i2, y_i2 = map(int, self.xyxy[target])
        x_j1, y_j1, x_j2, y_j2 = map(int, self.xyxy[det])
        y1 = min(y_i1, y_j1)
        y2 = max(y_i2, y_j2)
        x1 = min(x_i1, x_j1)
        x2 = max(x_i2, x_j2)
        self.xyxy[target, :] = x1, y1, x2, y2

def update_bbox(mask):
    """根据掩码更新边界框。"""
    y, x = np.nonzero(mask)
    return np.min(x), np.min(y), np.max(x), np.max(y)

def if_same_distribution(img1, img2, mask1, mask2, sim_threshold):
    """检查两个掩码区域的颜色分布是否相似。"""
    # Separate the image into three channels
    b1, g1, r1 = cv2.split(img1)
    b2, g2, r2 = cv2.split(img2)

    b1, g1, r1 = b1[mask1], g1[mask1], r1[mask1]
    b2, g2, r2 = b2[mask2], g2[mask2], r2[mask2]

    # Compute histograms for each channel
    num_batches = 16
    hist_b1, _ = np.histogram(b1, bins=num_batches, range=(0, 256))
    hist_g1, _ = np.histogram(g1, bins=num_batches, range=(0, 256))
    hist_r1, _ = np.histogram(r1, bins=num_batches, range=(0, 256))
    hist_b2, _ = np.histogram(b2, bins=num_batches, range=(0, 256))
    hist_g2, _ = np.histogram(g2, bins=num_batches, range=(0, 256))
    hist_r2, _ = np.histogram(r2, bins=num_batches, range=(0, 256))

    # Normalize histograms
    hist_b1 = hist_b1 / np.linalg.norm(hist_b1)
    hist_g1 = hist_g1 / np.linalg.norm(hist_g1)
    hist_r1 = hist_r1 / np.linalg.norm(hist_r1)
    hist_b2 = hist_b2 / np.linalg.norm(hist_b2)
    hist_g2 = hist_g2 / np.linalg.norm(hist_g2)
    hist_r2 = hist_r2 / np.linalg.norm(hist_r2)

    # Concatenate histograms
    hist1 = np.concatenate([hist_b1, hist_g1, hist_r1])
    hist2 = np.concatenate([hist_b2, hist_g2, hist_r2])

    # Compute cosine similarity
    cos_sim = cosine_similarity([hist1], [hist2])[0][0]

    return cos_sim > sim_threshold

def get_text_features(
    class_names: list, clip_model, clip_tokenizer, device, clip_length, batch_size=64
) -> np.ndarray:

    multiple_templates = [
        "{}",
        "There is the {} in the scene.",
    ]

    # Get all the prompted sequences
    class_name_prompts = [
        x.format(lm) for lm in class_names for x in multiple_templates
    ]

    # Get tokens
    text_tokens = clip_tokenizer(class_name_prompts).to(device)
    # Get Output features
    text_feats = np.zeros((len(class_name_prompts), clip_length), dtype=np.float32)
    # Get the text feature batch by batch
    text_id = 0
    while text_id < len(class_name_prompts):
        # Get batch size
        batch_size = min(len(class_name_prompts) - text_id, batch_size)
        # Get text prompts based on batch size
        text_batch = text_tokens[text_id : text_id + batch_size]
        with torch.no_grad():
            batch_feats = clip_model.encode_text(text_batch).float()

        batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
        batch_feats = np.float32(batch_feats.cpu())
        # move the calculated batch into the Ouput features
        text_feats[text_id : text_id + batch_size, :] = batch_feats
        # Move on and Move on
        text_id += batch_size

    # shrink the output text features into classes names size
    text_feats = text_feats.reshape((-1, len(multiple_templates), text_feats.shape[-1]))
    text_feats = np.mean(text_feats, axis=1)

    # TODO: Should we do normalization? Answer should be YES
    norms = np.linalg.norm(text_feats, axis=1, keepdims=True)
    text_feats /= norms

    return text_feats

def save_hilow_debug(bbox_hl_mapping, output_image, frame_idx):
    for item in bbox_hl_mapping:
        bbox = item[0]
        hl_debug = item[1]
        hl_idx = item[2]
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{hl_debug[0]:.3f}_{hl_idx[0]}, {hl_debug[1]:.3f}_{hl_idx[1]}, {hl_debug[2]:.3f}_{hl_idx[2]}"
        cv2.putText(
            output_image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    output_image_path = f"./debug/{frame_idx}_bbox_hl_mapping.jpg"
    cv2.imwrite(output_image_path, output_image)
