from collections import Counter

import numpy as np
import open3d as o3d
import torch


def mask_depth_to_points(
    depth: torch.Tensor,
    image: torch.Tensor,
    cam_K: torch.Tensor,
    masks: torch.Tensor,
    device: str = "cuda",
):
    """
    根据深度图、相机内参和实例 mask, 将像素坐标转换为 3D 点云, 并为每个点附上颜色

    参数说明:
        depth: (H, W)              单通道深度图, 单位通常为米, 0 或负值表示无效深度
        image: (H, W, 3) 或 None   RGB 图像, 像素值范围通常为 [0, 1] 或 [0, 255]
        cam_K: (3, 3)              相机内参矩阵
                                   [ [fx,  0, cx],
                                     [ 0, fy, cy],
                                     [ 0,  0,  1] ]
        masks: (N, H, W)           N 个二值/概率 mask, 每个 mask 对应一个前景对象或区域
                                   - 若为二值: 1 表示属于该对象, 0 表示不属于
                                   - 若为浮点概率：值越大表示该像素越属于该对象
        device: str                计算设备, 如 "cuda" 或 "cpu"

    返回:
        points: (N, H, W, 3)       每个 mask 下的 3D 点坐标 (X, Y, Z),
                                   无效点为 0 (通过 valid 掩掉)
        colors: (N, H, W, 3)       每个点的 RGB 颜色, 与 points 对应,
                                   无效点颜色为 0; 若无 image, 则为为每个 mask 随机分配的颜色
    """
    N, H, W = masks.shape

    fx, fy, cx, cy = cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2]

    # 构建像素网格坐标 (x, y)，维度为 (H, W)
    # y: 行索引 [0, 1, ..., H-1]
    # x: 列索引 [0, 1, ..., W-1]
    # indexing="ij" 表示第一维对应 y(行)，第二维对应 x(列)
    y, x = torch.meshgrid(
        torch.arange(0, H, device=device),
        torch.arange(0, W, device=device),
        indexing="ij",
    )

    # 将单通道深度图扩展到 N 个通道，对应 N 个 mask：
    # depth:    (H, W)
    # depth.repeat(N, 1, 1): (N, H, W)，为每个 mask 复制一份深度图
    # z: (N, H, W)，mask 后的深度，只保留属于该 mask 的区域
    #   - 若 masks 是 {0,1}，则 z 在非 mask 区域为 0
    #   - 若 masks 是权重/概率，则 z 相当于被加权
    z = depth.repeat(N, 1, 1) * masks

    # 生成有效深度的掩码：
    # valid: (N, H, W)，z > 0 的位置为 1，其余为 0
    # 用 float 是为了后续可以直接做逐元素乘法作为 mask
    valid = (z > 0).float()

    # 根据针孔相机模型，从像素坐标和深度恢复相机坐标系下的 3D 点:
    # 像素坐标 (u, v) ≈ (x, y)，深度为 z 时，
    # X = (u - cx) * z / fx
    # Y = (v - cy) * z / fy
    #
    # 这里 x, y 原本是 (H, W)，通过广播与 z (N, H, W) 运算后，
    # 会自动扩展成 (N, H, W)，即为每个 mask 都计算一份 3D 坐标
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy

    # 将 (X, Y, Z) 堆叠到最后一个维度，得到点云：
    # x, y, z:      (N, H, W)
    # stack 后为：  (N, H, W, 3)
    points = torch.stack((x, y, z), dim=-1) * valid.unsqueeze(-1)
    # valid.unsqueeze(-1): (N, H, W, 1)
    # 与 points 相乘后，将无效点（z <= 0 处）统一置为 0

    if image is not None:
        # 若提供了原始 RGB 图像，则使用对应像素的颜色

        # 将 image 扩展到 N 个对象:
        # image: (H, W, 3)
        # image.repeat(N, 1, 1, 1): (N, H, W, 3)
        # 再与 masks (N, H, W) 相乘，仅保留当前对象的区域颜色

        rgb = image.repeat(N, 1, 1, 1) * masks.unsqueeze(-1)
        # 再次乘以 valid，将深度无效处（z <= 0）也置零
        colors = rgb * valid.unsqueeze(-1)
    else:
        # 若没有提供 RGB 图像，则为每个对象随机分配一个颜色
        print("No RGB image provided, assigning random colors to objects")
        # Generate a random color for each mask
        # 为每个 mask 生成一个随机 RGB 颜色，范围 [0,1]
        # random_colors: (N, 3)，每一行是一个对象的 RGB 颜色
        random_colors = (
            torch.randint(0, 256, (N, 3), device=device, dtype=torch.float32) / 255.0
        )  # RGB colors in [0, 1]
        # Expand dims to match (N, H, W, 3) and apply to valid points

        # 将 (N, 3) 扩展为 (N, H, W, 3)，使每个对象在其 mask 区域内是同一种颜色
        # unsqueeze(1).unsqueeze(1): (N, 1, 1, 3)
        # expand(-1, H, W, -1):      (N, H, W, 3)
        colors = random_colors.unsqueeze(1).unsqueeze(1).expand(
            -1, H, W, -1
        ) * valid.unsqueeze(-1)
        # 再乘以 valid，将无效点的颜色置为 0
        
    # 返回每个 mask 对应的点云和颜色
    return points, colors


def init_pcd_denoise_dbscan(
    pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10
) -> o3d.geometry.PointCloud:
    ## Remove noise via clustering
    pcd_clusters = pcd.cluster_dbscan(  # inint
        eps=eps,
        min_points=min_points,
    )

    # Convert to numpy arrays
    obj_points = np.asarray(pcd.points)
    obj_colors = np.asarray(pcd.colors)
    pcd_clusters = np.array(pcd_clusters)

    # Count all labels in the cluster
    counter = Counter(pcd_clusters)

    # Remove the noise label
    if counter and (-1 in counter):
        del counter[-1]

    if counter:
        # Find the label of the largest cluster
        most_common_label, _ = counter.most_common(1)[0]

        # Create mask for points in the largest cluster
        largest_mask = pcd_clusters == most_common_label

        # Apply mask
        largest_cluster_points = obj_points[largest_mask]
        largest_cluster_colors = obj_colors[largest_mask]

        # If the largest cluster is too small, return the original point cloud
        if len(largest_cluster_points) < 5:
            return pcd

        # Create a new PointCloud object
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)

        pcd = largest_cluster_pcd

    return pcd


def refine_points_with_clustering(points, colors, eps=0.05, min_points=10):
    """
    Cluster the point cloud using Open3D's DBSCAN and extract the largest cluster.

    Args:
    - points: Point cloud coordinates (torch.Tensor, Nx3).
    - colors: Point cloud colors (torch.Tensor, Nx3).
    - eps: DBSCAN neighborhood radius.
    - min_points: Minimum number of points for DBSCAN.

    Returns:
    - refined_points: Filtered point cloud coordinates (numpy.ndarray).
    - refined_colors: Filtered point cloud colors (numpy.ndarray).
    """
    # Convert to numpy format
    points_np = points.cpu().numpy()
    colors_np = colors.cpu().numpy()

    # If there are no points, return empty arrays to avoid further processing
    if points_np.shape[0] == 0:
        # print("No points found in the input point cloud.")
        # FIXED: [KDTreeFlann::SetRawData] Failed due to no data warning
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.colors = o3d.utility.Vector3dVector(colors_np)

    # Use Open3D's DBSCAN clustering
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))

    # Get the size of each cluster
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Remove noise points (-1)
    if -1 in unique_labels:
        mask_noise = labels != -1
        labels = labels[mask_noise]
        points_np = points_np[mask_noise]
        colors_np = colors_np[mask_noise]
        unique_labels, counts = np.unique(
            labels, return_counts=True
        )  # Recalculate clustering information

    # Check if there are still clusters
    if len(unique_labels) == 0:
        # print("No valid clusters found after removing noise.")
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    # Find the largest cluster
    max_label = unique_labels[np.argmax(counts)]

    # Select points in the largest cluster
    mask = labels == max_label
    refined_points_np = points_np[mask]
    refined_colors_np = colors_np[mask]

    # Return as numpy arrays
    return refined_points_np, refined_colors_np


def pcd_dbscan(
    pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10
) -> o3d.geometry.PointCloud:
    ## Remove noise via clustering
    # Convert point cloud to numpy arrays
    points_np = np.asarray(pcd.points)
    colors_np = np.asarray(pcd.colors) if len(pcd.colors) > 0 else None

    # DBSCAN clustering
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))

    if len(labels) == 0:
        print("No clusters found!")
        return o3d.geometry.PointCloud()  # Return empty point cloud

    # Get cluster sizes
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Find largest cluster
    max_label = unique_labels[np.argmax(counts)]

    # Print cluster info
    print(
        f"Found {len(unique_labels)} clusters, selecting cluster with label {max_label} (size: {counts.max()})"
    )

    # Select points in largest cluster
    mask = labels == max_label
    refined_points = points_np[mask]

    if colors_np is not None:
        refined_colors = colors_np[mask]
    else:
        refined_colors = None

    # Create new point cloud
    refined_pcd = o3d.geometry.PointCloud()
    refined_pcd.points = o3d.utility.Vector3dVector(refined_points)
    if refined_colors is not None:
        refined_pcd.colors = o3d.utility.Vector3dVector(refined_colors)

    return refined_pcd


def safe_create_bbox(
    pcd: o3d.geometry.PointCloud,
) -> o3d.geometry.AxisAlignedBoundingBox:
    """
    Safely compute the axis-aligned bounding box of a point cloud.
    If the point cloud is empty, min_bound and max_bound are both [0,0,0].
    """
    points = np.asarray(pcd.points)
    if points.shape[0] == 0:
        # Return empty bounding box
        return o3d.geometry.AxisAlignedBoundingBox(
            np.array([0, 0, 0]), np.array([0, 0, 0])
        )
    else:
        bbox = pcd.get_axis_aligned_bounding_box()
        return bbox
