import copy
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import open3d as o3d

@dataclass
class DataInput:
    """
    数据输入的数据类。

    Attributes:
        idx (int): 帧索引
        time_stamp (float): 时间戳
        color (np.ndarray): 彩色图像 (H, W, 3), uint8类型
        depth (np.ndarray): 深度图像 (H, W), float32类型
        color_name (str): 彩色图像的文件名
        intrinsics (np.ndarray): 相机内参矩阵 (3x3)
        pose (np.ndarray): 相机位姿矩阵 (4x4)，从相机到世界的变换
    """

    idx: int = 0
    time_stamp: float = 0.0
    color: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0, 3), dtype=np.uint8)
    )
    # 深度图，形状为 H, W, 1
    depth: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float32)
    )
    color_name: str = ""
    # 3*3 的内参矩阵
    intrinsics: np.ndarray = field(default_factory=lambda: np.eye(3))
    pose: np.ndarray = field(default_factory=lambda: np.eye(4))

    def clear(self) -> None:
        """清空所有字段为默认值。"""
        self.idx = 0
        self.time_stamp = 0.0
        self.color = np.empty((0, 0, 3), dtype=np.uint8)
        self.depth = np.empty((0, 0), dtype=np.float32)
        self.color_name = ""
        self.intrinsics = np.eye(3)
        self.pose = np.eye(4)

    def copy(self):
        """返回对象的深拷贝。"""
        return copy.deepcopy(self)

@dataclass
class Observation:
    """
    观测信息的数据类

    Attributes:
        class_id (int): 类别ID
        pcd (o3d.geometry.PointCloud): 点云
        bbox (o3d.geometry.AxisAlignedBoundingBox): 轴对齐包围盒
        clip_ft (np.ndarray): CLIP特征向量
        matched_obj_uid (None): 匹配到的对象UID
        matched_obj_score (float): 匹配得分
        matched_obj_idx (int): 匹配到的对象索引
    """

    class_id: int = 0
    class_name: str = ""
    pcd: o3d.geometry.PointCloud = field(default_factory=o3d.geometry.PointCloud)
    bbox: o3d.geometry.AxisAlignedBoundingBox = field(
        default_factory=o3d.geometry.AxisAlignedBoundingBox
    )
    clip_ft: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))

    # 匹配信息
    matched_obj_uid: None = None
    matched_obj_score: float = 0.0
    matched_obj_idx: int = -1

@dataclass
class LocalObservation(Observation):
    """
    局部观测信息的数据类, 继承自Observation

    Attributes:
        idx (int): 帧索引
        mask (np.ndarray): 分割掩码
        xyxy (np.ndarray): 边界框坐标 (x1, y1, x2, y2)
        conf (float): 置信度
        distance (float): 物体到相机的距离
        is_low_mobility (bool): 是否为低移动性物体
        masked_image (np.ndarray): 掩码后的图像（用于调试）
        cropped_image (np.ndarray): 裁剪后的图像（用于调试）
    """

    idx: int = 0
    mask: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.uint8))
    xyxy: np.ndarray = field(default_factory=lambda: np.empty((0, 4), dtype=np.float32))
    conf: float = 0.0
    distance: float = 0.0

    is_low_mobility: bool = False

    # 以下属性用于调试
    masked_image: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0, 3), dtype=np.uint8)
    )
    cropped_image: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0, 3), dtype=np.uint8)
    )

@dataclass
class GlobalObservation(Observation):
    """
    全局观测信息的数据类, 继承自Observation。

    Attributes:
        uid (uuid.UUID): 唯一标识符
        pcd_2d (o3d.geometry.PointCloud): 2D点云
        bbox_2d (o3d.geometry.AxisAlignedBoundingBox): 2D轴对齐包围盒
        related_objs (list): 相关对象的CLIP特征列表
        related_bbox (list): 相关对象的包围盒列表（用于演示）
        related_color (list): 相关对象的颜色列表（用于演示）
        masked_image (np.ndarray): 单个掩码后的图像（用于调试）
        cropped_image (np.ndarray): 单个裁剪后的图像（用于调试）
        masked_images (list): 掩码后的图像列表，每个元素为 np.ndarray (H, W, 3), uint8类型（用于调试）
        cropped_images (list): 裁剪后的图像列表，每个元素为 np.ndarray (H, W, 3), uint8类型（用于调试）
    """

    uid: uuid.UUID = field(default_factory=uuid.uuid4)
    pcd_2d: o3d.geometry.PointCloud = field(default_factory=o3d.geometry.PointCloud)
    bbox_2d: o3d.geometry.AxisAlignedBoundingBox = field(
        default_factory=o3d.geometry.AxisAlignedBoundingBox
    )
    # 相关对象
    # 当前我们“只保存CLIP特征” <-- 请注意！
    related_objs: list = field(default_factory=list)
    # 仅用于更好的演示
    related_bbox: list = field(default_factory=list)
    related_color: list = field(default_factory=list)

    # 以下属性用于调试
    masked_image: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0, 3), dtype=np.uint8)
    )
    cropped_image: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0, 3), dtype=np.uint8)
    )
    # 以下属性用于调试
    masked_images: list = field(default_factory=list)
    cropped_images: list = field(default_factory=list)

class ObjectClasses:
    """
    管理对象类别及其关联颜色

    为类别文件创建颜色映射, 同时管理背景类型的显示与否, 默认背景类别为 [floor, wall, ceiling] 地板, 墙和天花板

    Attributes:
        classes_file_path (str): 包含类别名称的文件路径，每行一个

    Usage:
        obj_classes = ObjectClasses(classes_file_path, skip_bg=True)
        model.set_classes(obj_classes.get_classes_arr())
        some_class_color = obj_classes.get_class_color(index or class_name)
    """

    def __init__(self, classes_file_path, bg_classes, skip_bg):
        self.classes_file_path = Path(classes_file_path)
        self.bg_classes = bg_classes
        self.skip_bg = skip_bg
        self.classes, self.class_to_color = self._load_or_create_colors(
            selection_ratio=1.0
        )

    def _load_or_create_colors(self, selection_ratio=1.0):
        """
        创建类别和颜色的关键函数
        需要外部类别文件用于检测部分
        """
        with open(self.classes_file_path, "r") as f:
            all_class = [cls.strip() for cls in f.readlines()]

        # 根据 skip_bg 标志过滤所有类别
        if self.skip_bg:
            classes = [cls for cls in all_class if cls not in self.bg_classes]
        else:
            classes = all_class

        # 为每个类别添加颜色
        # 加载颜色路径
        color_file_path = (
            self.classes_file_path.parent / f"{self.classes_file_path.stem}_colors.json"
        )

        id_color_file_path = (
            self.classes_file_path.parent
            / f"{self.classes_file_path.stem}_id_colors.json"
        )

        if color_file_path.exists():
            with open(color_file_path, "r") as f:
                class_to_color = json.load(f)
            # 构建一个字典映射 {class, color}
            class_to_color = {
                cls: class_to_color[cls] for cls in classes if cls in class_to_color
            }
        else:
            class_to_color = {
                class_name: list(np.random.rand(3).tolist()) for class_name in classes
            }
            # 生成相应的 id_to_color 映射
            id_to_color = {str(i): class_to_color[cls] for i, cls in enumerate(classes)}
            # 将新字典转储到json
            with open(color_file_path, "w") as f:
                json.dump(class_to_color, f)

            with open(id_color_file_path, "w") as f:
                json.dump(id_to_color, f)

        if selection_ratio == 1.0:
            return classes, class_to_color

        import random

        # 根据比例随机选择一部分类别及其颜色
        num_selected = max(
            1, int(len(classes) * selection_ratio)
        )  # 至少选择一个
        selected_classes = random.sample(classes, num_selected)
        selected_class_to_color = {cls: class_to_color[cls] for cls in selected_classes}

        return selected_classes, selected_class_to_color

    def get_classes_arr(self):
        """
        返回类别名称列表
        """
        return self.classes

    def get_bg_classes_arr(self):
        """
        返回被跳过的类别名称列表
        """
        return self.bg_classes

    def get_class_color(self, key):
        """
        获取与给定类别名称或索引关联的颜色

        Args:
            key (str or int): 类别的索引或名称

        Returns:
            list: 与类别关联的RGB颜色
        """
        if isinstance(key, int):
            if key < 0 or key >= len(self.classes):
                raise ValueError(f"Invalid class index out of range: {key}")
            return self.class_to_color[self.classes[key]]
        elif isinstance(key, str):
            if key not in self.classes:
                raise ValueError(f"Invalid class name: {key}")
            return self.class_to_color[key]
        else:
            raise TypeError(f"Invalid type for key: {type(key)}")

    def get_class_color_dict_by_index(self):
        """
        Return a dictionary mapping class index to color, indexed by class index
        """
        return {
            str(i): self.class_to_color[self.classes[i]]
            for i in range(len(self.classes))
        }

class GoalMode(Enum):
    RANDOM = "random"
    CLICK = "click"
    INQUIRY = "inquiry"
