"""
完成task02和task03：找到相关物体 & 动态避障
NOTE: 不包含 semantic hazard的处理部分，默认task 23没有需要避障的语义物体（地毯）
dualmap 主机端执行：订阅目标/相关物体/房间等，基于离线 local map 查询位置；
并通过 Nav2 NavigateToPose 导航到目标点，并支持面向目标的旋转与recovery流程。
"""

import os
os.environ["DISPLAY"] = ""

import sys
import time
import math
import cv2
from cv_bridge import CvBridge
from PIL import Image as PILImage
import base64
from io import BytesIO
import yaml
import threading
import json
import requests
from typing import Optional, Tuple

import numpy as np
import open_clip
import torch
import torch.nn.functional as F

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import PoseStamped, PoseArray
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import Image
from action_msgs.msg import GoalStatus

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))        # applications/
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)                    # DualMap/
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.path.append("/home/tang123/Documents/DualMap/applications/utils")
from utils.object import BaseObject
import datetime

LOG_FILE = "nav_result.txt"


def write_log(message):
    """
    记录带有时间戳的日志到文件
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    log_entry = f"[{timestamp}] {message}\n"
    # 输出到控制台方便调试，也可注释掉
    print(log_entry.strip())
    with open(LOG_FILE, "a") as f:
        f.write(log_entry)


STATUS_NAME = {
    GoalStatus.STATUS_UNKNOWN: "UNKNOWN",
    GoalStatus.STATUS_ACCEPTED: "ACCEPTED",
    GoalStatus.STATUS_EXECUTING: "EXECUTING",
    GoalStatus.STATUS_CANCELING: "CANCELING",
    GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
    GoalStatus.STATUS_CANCELED: "CANCELED",
    GoalStatus.STATUS_ABORTED: "ABORTED",
}


def yaw_to_quaternion(yaw: float):
    qz = math.sin(yaw * 0.5)
    qw = math.cos(yaw * 0.5)
    return 0.0, 0.0, qz, qw


def quat_to_yaw(qx, qy, qz, qw) -> float:
    # yaw (Z) from quaternion
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        qw * qw + qx * qx - qy * qy - qz * qz
    )


class TaskSubscriber(Node):
    def __init__(self, cfg_path: str):
        super().__init__("nav2_goal_sender")

        with open(cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.load_dir = self.cfg["load_dir"]

        # ====== callback group：允许并发回调（配合 MultiThreadedExecutor）======
        self._cbg = ReentrantCallbackGroup()

        self.position_sub = self.create_subscription(
            Odometry, "/odom", self.position_callback, 10, callback_group=self._cbg
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, "/map", self.map_callback, 10, callback_group=self._cbg
        )
        self.rgb_sub = self.create_subscription(
            Image, "/camera/color/image_raw", self.rgb_callback, 10, callback_group=self._cbg
        )

        # SJQ: 此处订阅DUALMAP传回的检索结果
        self.dualmp_sub = self.create_subscription(
            PoseStamped, "/remap_target_position", self.remap_target_callback, 10, callback_group=self._cbg
        )
        # SJQ: publisher 发布相关物体的 bbox 信息，供 dualmap 端接收，对应的函数在下方标记
        self.related_bbox_pub = self.create_publisher(PoseStamped, '/related_bbox_junqi_only', 10)

        # ====== Nav2 Action Client ======
        self._action_name = "/navigate_to_pose"
        self._client = ActionClient(self, NavigateToPose, self._action_name, callback_group=self._cbg)

        # ====== 运行数据 ======
        self.target_name: Optional[str] = None
        self.related_object_name: Optional[str] = None
        self.obj_map = None

        self.load_results()

        # DEBUG: 如果想看物体class和位置等信息，取消下面这句的注释，会先打印出来
        # self.debug_all_objs()
        self.clip_model_path = self.cfg["clip_dir"]
        self.init_clip()

        # 机器人当前位姿（来自 /odom）
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self._last_odom_print_t = 0.0

        # 房间与过滤
        self.room = None
        self.room_bbox = None
        self.is_room_ready = False

        # ===== 任务检查相关 ======
        self.rgb_check_end = False
        self.bridge = CvBridge()
        self.latest_image = None
        self.image_lock = threading.Lock()

        # ===== VLM recover 大保底 ======
        self.vlm_recover = False
        self.turn_around_imgs = {}

        # ====== 从配置文件读取房间边界 ======
        self.room_edges = {}
        if "room_edges" in self.cfg:
            self.room_edges = self.cfg["room_edges"]
            self.get_logger().info(f"Loaded room edges from config: {list(self.room_edges.keys())}")
        self.room_anchors = {}
        if "room_anchors" in self.cfg:
            self.room_anchors = self.cfg["room_anchors"]
            self.get_logger().info(f"Loaded room anchors from config: {list(self.room_anchors.keys())}")


        # ====== 流程控制（线程安全） ======
        self._lock = threading.Lock()
        self._task_event = threading.Event()
        self._shutdown_event = threading.Event()

        # 最新计算出的目标位置
        self.target_x = None
        self.target_y = None

        # ====== 地图变量 ======
        self.map_data = None
        self.map_info = None
        self.map_received = False

        self.require_room_filter = True   # 强制要求目标必须在房间bbox内
        self.room_wait_timeout = 3.0      # 等 room topic 的最大时间（秒）
        # 到达目标点距离，ok就转向它
        self._arrived_dist = 1.0

        # 主线程，开始检索，分配状态机
        self._worker = threading.Thread(target=self._task_worker, daemon=True)
        self._worker.start()

        self.get_logger().info("TaskSubscriber initialized. Waiting for topics...")

    # ====================== 回调：只做轻量更新 ======================

    def map_callback(self, msg: OccupancyGrid):
        """处理map数据，更新 map_data 和 map_info"""
        try:
            width = msg.info.width
            height = msg.info.height
            resolution = msg.info.resolution
            origin_x = msg.info.origin.position.x
            origin_y = msg.info.origin.position.y

            data = np.array(msg.data, dtype=np.int8).reshape((height, width))

            with self._lock:
                self.map_data = data
                self.map_info = {
                    "width": width,
                    "height": height,
                    "resolution": resolution,
                    "origin_x": origin_x,
                    "origin_y": origin_y
                }
                self.map_received = True

        except Exception as e:
            self.get_logger().error(f"Error in map_callback: {e}")

    def world_to_map(self, x: float, y: float) -> Tuple[int, int]:
        """世界坐标 -> 地图栅格坐标"""
        if self.map_info is None:
            return 0, 0
        resolution = self.map_info["resolution"]
        origin_x = self.map_info["origin_x"]
        origin_y = self.map_info["origin_y"]
        mx = int((x - origin_x) / resolution)
        my = int((y - origin_y) / resolution)
        return mx, my

    def map_to_world(self, mx: int, my: int) -> Tuple[float, float]:
        """地图栅格坐标 -> 世界坐标"""
        if self.map_info is None:
            return 0.0, 0.0
        resolution = self.map_info["resolution"]
        origin_x = self.map_info["origin_x"]
        origin_y = self.map_info["origin_y"]
        wx = mx * resolution + origin_x
        wy = my * resolution + origin_y
        return wx, wy

    def find_optimal_free_point_by_room_center(
        self, target_x: float, target_y: float, search_radius: float = 0.8
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        寻找最优空闲点：优先选择距离房间中心“曼哈顿距离”最小的空闲栅格点
        """
        with self._lock:
            if (not self.map_received) or (self.map_data is None) or (self.map_info is None):
                self.get_logger().warn("Map not received yet, cannot find free point")
                return target_x, target_y

            map_data = self.map_data.copy()
            map_info = self.map_info.copy()

            # 获取房间边界
            room_bbox = self.room_bbox
            is_room_ready = self.is_room_ready

        # 检查房间是否准备好
        if not is_room_ready or room_bbox is None:
            self.get_logger().warn("Room not ready, using original target point")
            return target_x, target_y

        min_x, max_x, min_y, max_y = room_bbox
        room_center_x = (min_x + max_x) / 2.0
        room_center_y = (min_y + max_y) / 2.0

        resolution = map_info["resolution"]
        target_mx, target_my = self.world_to_map(target_x, target_y)
        room_center_mx, room_center_my = self.world_to_map(room_center_x, room_center_y)
        search_cells = int(search_radius / resolution)

        candidate_points = []
        # 搜索整个半径内的所有点
        for radius in range(0, search_cells + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx * dx + dy * dy > radius * radius:
                        continue
                    mx = target_mx + dx
                    my = target_my + dy
                    if 0 <= mx < map_info["width"] and 0 <= my < map_info["height"]:
                        if map_data[my, mx] == 0:
                            manhattan_to_room_center = abs(mx - room_center_mx) + abs(my - room_center_my)
                            euclidean_to_target = math.sqrt(dx * dx + dy * dy) * resolution
                            candidate_points.append((manhattan_to_room_center, euclidean_to_target, mx, my))

        # 如果有候选点，找到距离房间中心曼哈顿距离最小的点
        if candidate_points:
            # 首先按距离房间中心的曼哈顿距离从小到大排序（主要排序条件）
            # 如果曼哈顿距离相同，再按距离目标点的欧氏距离从小到大排序（次要条件，选择更接近目标的点）
            candidate_points.sort(key=lambda x: (x[0], x[1]))
            best_manhattan, best_euclidean, best_mx, best_my = candidate_points[0]
            free_x, free_y = self.map_to_world(best_mx, best_my)

            # 计算实际世界距离
            distance_to_target = math.sqrt((free_x - target_x) ** 2 + (free_y - target_y) ** 2)
            distance_to_room_center = abs(free_x - room_center_x) + abs(free_y - room_center_y)

            self.get_logger().info(
                f"Found optimal free point at ({free_x:.3f}, {free_y:.3f}), "
                f"distance to target: {distance_to_target:.3f}m, "
                f"Manhattan to room center: {distance_to_room_center:.3f}m, "
                f"map cells Manhattan: {best_manhattan}"
            )
            return free_x, free_y

        # 没有找到空闲点，返回原始目标点
        self.get_logger().warn(f"No free point found within {search_radius}m radius, using original target")
        return target_x, target_y

    def position_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        o = msg.pose.pose.orientation
        yaw = quat_to_yaw(o.x, o.y, o.z, o.w)

        self.current_x = x
        self.current_y = y
        self.current_yaw = yaw

        t = time.time()
        if t - self._last_odom_print_t > 0.5:
            self._last_odom_print_t = t

    # ====================== ROOM TARGET RELATED 回调，由 main 直接调用，不走ros ======================

    def _room_cb(self, room: str):
        """回调函数：接收目标房间，匹配与指令最相似的房间"""
        bbox, anchor_pt = self.query_room_callback(room)
        if bbox is None:
            self.get_logger().warn(f"Room '{room}' not matched; will not apply room filter.")
            return

        with self._lock:
            self.room = room
            self.room_bbox = bbox
            self.room_anchor_pt = anchor_pt
            self.is_room_ready = True

        self.get_logger().info(f"Room ready: {room} bbox={bbox} anchor={anchor_pt}")

    def _target_cb(self, target_name: str):
        with self._lock:
            self.target_name = target_name
        self.get_logger().info(f"Received target name: {self.target_name}")
        # 触发 worker 执行任务
        self._task_event.set()

    def _related_obj_cb(self, related_name: str):
        with self._lock:
            self.related_object_name = related_name
        self.get_logger().info(f"Received related object name: {self.related_object_name}")
        # 触发 task_worker：如果已有 target_name，则走 related->target 流程
        self._task_event.set()

    # def _hazard_cb(self, hazard: str):
    #     """存储障碍的边界到yaml中"""
    #     self.get_logger().info(f"Received hazard name: {hazard}")
    #     corners = self.query_callback(hazard)
    #     hazard_path = self.cfg["hazard_yaml_path"]
    #     if corners is not None:
    #         pop_hazard2yaml(hazard_path, corners)

    # ====================== Worker：执行导航流程 ======================

    def request_exit(self, reason: str = ""):
        self.get_logger().warn(f"Request exit. reason={reason}")
        self._shutdown_event.set()
        try:
            rclpy.shutdown()
        except Exception:
            pass

    def _task_worker(self):
        """
        后台线程：等待 target/related 事件，然后执行：
        - 有 related：先去 related，再去/转向 target
        -  失败则进入 recovery（转N圈 + 重试）
        """
        while not self._shutdown_event.is_set():
            self._task_event.wait(timeout=0.2)
            if self._shutdown_event.is_set():
                break
            if not self._task_event.is_set():
                continue

            # 取一次快照
            with self._lock:
                target_name = self.target_name
                related_name = self.related_object_name

            self._task_event.clear()

            # TODO: 没解析到target，不对劲，退出
            if not target_name:
                self.get_logger().warn("Worker triggered but target_name is empty. Skip.")
                continue

            try:
                # ===== 如果要求目标必须在 room 内：先等待 room_ready =====
                if getattr(self, "require_room_filter", True):
                    t0 = time.time()
                    while True:
                        with self._lock:
                            room_ready = self.is_room_ready
                            room_bbox = self.room_bbox
                            room_name = getattr(self, "room", None)

                        if room_ready and room_bbox is not None:
                            self.get_logger().info(f"[room] ready: {room_name} bbox={room_bbox}")
                            break

                        if time.time() - t0 > getattr(self, "room_wait_timeout", 3.0):
                            self.get_logger().error(
                                f"[room] require_room_filter=True but room not ready within "
                                f"{getattr(self, 'room_wait_timeout', 3.0)}s. Publish /target_room first."
                            )
                            raise RuntimeError("room not ready")

                        time.sleep(0.05)

                # 查询 target 位置
                corners = self.query_callback(target_name)
                if corners is None:
                    self.get_logger().error(f"Target '{target_name}' not found in map.")
                    continue

                target_pos = np.array(corners).mean(axis=0)
                target_x, target_y = float(target_pos[0]), float(target_pos[1])

                # FLAG: 查找距离最近的空闲点
                # NOTE: 1.0是 target 距离阈值，根据bbox大小调整
                free_x, free_y = self.find_optimal_free_point_by_room_center(target_x, target_y, 1.0)

                with self._lock:
                    self.target_x = target_x
                    self.target_y = target_y

                self.get_logger().info(f"[query] target '{target_name}' -> ({target_x:.3f}, {target_y:.3f})")


                # related 优先导航,不再去 target的位置，target给recovery做，此处只去related的位置/房间的锚点
                if related_name != "None":
                    # NOTE: 有相关物体
                    rcorners = self.query_callback(related_name)
                    if rcorners is None:
                        # NOTE: 没有检索到相关物体的位置，直接去房间锚点
                        if self.room_anchor_pt is not None:
                            anchor_x, anchor_y = self.room_anchor_pt
                            self.get_logger().info(f"No related object, go to room anchor point: ({anchor_x:.3f}, {anchor_y:.3f})")
                            ok = self._goto_point(anchor_x, anchor_y, yaw=0.0, frame_id="map", wait_timeout=5.0)
                        else:
                            # TODO: 没有相关物体，也没有房间锚点，随便给个锚点
                            pass

                    else:
                        # NOTE: 找到相关物体位置，先去相关物体点
                        rpos = np.array(rcorners).mean(axis=0)
                        rx, ry = float(rpos[0]), float(rpos[1])
                        delta_rx = rcorners[1][0] - rcorners[0][0]
                        delta_ry = rcorners[2][1] - rcorners[1][1]

                        # FLAG: 导航到最近的相关空闲点
                        # NOTE: 1.2是 related 距离阈值，根据bbox大小调整
                        free_rx, free_ry = self.find_optimal_free_point_by_room_center(rx, ry, 1.2)

                        self.get_logger().info(f"[query] related '{related_name}' -> ({rx:.3f}, {ry:.3f})")
                        self.get_logger().warn(f"goto related point: ({free_rx:.3f}, {free_ry:.3f})")

                        # ok = self._goto_point(free_rx, free_ry, yaw=0.0, frame_id="map", wait_timeout=5.0)
                        # DEBUG: 朝向相关物体
                        ok = self._goto_and_face_target(free_rx, free_ry, rx, ry)

                        # if ok:
                        #     # 如果已经足够接近 target，直接 face；否则去 target
                        #     dist_to_target = math.sqrt(
                        #         (self.current_x - target_x) ** 2 + (self.current_y - target_y) ** 2
                        #     )
                        #     if dist_to_target < self._arrived_dist:
                        #         ok = self._face_target(target_x, target_y)
                        #     else:
                        #         self.get_logger().warn("Too far away from target, fallback to direct target.")
                        #         ok = self._goto_and_face_target(free_x, free_y, target_x, target_y)
                        # else:
                        #     self.get_logger().warn("Goto related failed, fallback to direct target.")
                        #     ok = self._goto_and_face_target(free_x, free_y, target_x, target_y)
                else:
                    # NOTE: 没有相关物体，直接去房间锚点
                    if self.room_anchor_pt is not None:
                        anchor_x, anchor_y = self.room_anchor_pt
                        self.get_logger().info(f"No related object, go to room anchor point: ({anchor_x:.3f}, {anchor_y:.3f})")
                        ok = self._goto_point(anchor_x, anchor_y, yaw=0.0, frame_id="map", wait_timeout=5.0)
                    else:
                        # TODO: 没有相关物体，也没有房间锚点，随便给个锚点？
                        pass

                # NOTE: 已经到达锚点位置，recovery 检查任务完成
                if ok:
                    is_complete = self.check_task()
                    print("Task is", is_complete)
                    if not is_complete:
                        # 任务未完成，尝试重新导航，发给DUALMAP 查东西
                        print("@@@@@@@@@@@@@@@@@ PASS TO DUALMAP REMAP @@@@@@@@@@@@@@@@@")
                        if related_name != "None":
                            self.get_logger().info(f"[EXIST RELATED] publish RELATED bbox")
                            # NOTE: 这里是发布相关物体的 bbox 信息，中心x，y 和 宽高
                            print("[RECOVER] related center and box size:", rx, ry, delta_rx, delta_ry)
                            self.publish_related_bbox(rx, ry, delta_rx, delta_ry, self.target_name)
                        else:
                            self.get_logger().info(f"[NO RELATED] publish ROOM bbox")
                            # NOTE: 这里是发布房间的 bbox 信息，中心x，y 和 宽高
                            room_cx = (self.room_bbox[0] + self.room_bbox[1]) / 2.0
                            room_cy = (self.room_bbox[2] + self.room_bbox[3]) / 2.0
                            room_w = self.room_bbox[1] - self.room_bbox[0]
                            room_h = self.room_bbox[3] - self.room_bbox[2]
                            print("[RECOVER] room center and box size:", room_cx, room_cy, room_w, room_h)
                            self.publish_related_bbox(room_cx, room_cy, room_w, room_h, self.target_name)

                        self.target_x = None
                        self.target_y = None

                        print("@@@@@@@@@@@@@@@@@ PASS TO DUALMAP REMAP @@@@@@@@@@@@@@@@@")

                        self.run_recovery()
                        self.request_exit("Recovery finished after task incomplete! ")
                    else:
                        self.request_exit("task complete")

                ok = self._goto_point(0.0, 0.0, yaw=0.0, frame_id="map", wait_timeout=5.0)
                # LOG
                if ok:
                    write_log("Returned to origin successfully")

            except Exception as e:
                self.get_logger().error(f"Worker exception: {repr(e)}")


    def publish_related_bbox(self, cx, cy, width, height, label):
        # SJQ: 此处为发布相关物体的 bbox 信息[name, center_x, center_y, width, height]，供 dualmap 端接收
        """临时发布边界框（使用PoseStamped）"""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        # frame_id 填充 target_name 的 str
        msg.header.frame_id = self.target_name

        # 用第一个位姿表示中心
        from geometry_msgs.msg import Pose
        center_pose = Pose()
        center_pose.position.x = cx
        center_pose.position.y = cy
        center_pose.position.z = 0.0

        # orientation xy 存储 width 和 height
        center_pose.orientation.x = width
        center_pose.orientation.y = height

        msg.pose = center_pose
        self.related_bbox_pub.publish(msg)
        self.get_logger().info(f"发布临时bbox: {label} at ({cx}, {cy})")

    def _goto_and_face_target(self, free_x: float, free_y: float, tx: float, ty: float) -> bool:
        # 先去目标点，再原地转向
        ok = self._goto_point(free_x, free_y, yaw=0.0, frame_id="map", wait_timeout=5.0)
        if not ok:
            return False
        return self._face_target(tx, ty)

    def _face_target(self, tx: float, ty: float) -> bool:
        delta_yaw = self.calculate_yaw_to_target(tx, ty)
        target_yaw = self.current_yaw + delta_yaw
        self.get_logger().info(f"Face target: delta_yaw={delta_yaw:.3f} -> target_yaw={target_yaw:.3f}")
        return self._goto_point(self.current_x, self.current_y, yaw=target_yaw, frame_id="map", wait_timeout=5.0)


    def remap_target_callback(self, msg: PoseStamped):
        # SJQ: 这里解析位置，赋值给 self.target_x, self.target_y
        """收到目标物体位置后，发布该位置"""
        remap_status = msg.header.frame_id
        if remap_status == "success":
            self.target_x = msg.pose.position.x
            self.target_y = msg.pose.position.y
            self.get_logger().info(f"收到目标物体位置: x={self.target_x:.3f}, y={self.target_y:.3f}")

        else:
            self.target_x = 0
            self.target_y = 0
            self.get_logger().info("Fail to get target, failed, QAQ.")


    # ====================== Nav2 Action：异步 + Event 等待 ======================

    def _goto_point(self, x: float, y: float, yaw: float, frame_id: str, wait_timeout: float) -> bool:
        """
        发送 NavigateToPose 并等待 result（在 worker 线程里 wait，不阻塞 ROS 回调线程）。
        FLAG: 不再做任何提前截断/cancel逻辑，完全交给 Nav2 自己的容忍度。
        """
        if not self._client.wait_for_server(timeout_sec=wait_timeout):
            self.get_logger().error(
                f"NavigateToPose server not available: '{self._action_name}' (waited {wait_timeout}s)"
            )
            return False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = frame_id
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        qx, qy, qz, qw = yaw_to_quaternion(float(yaw))
        goal_msg.pose.pose.orientation.x = qx
        goal_msg.pose.pose.orientation.y = qy
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw

        self.get_logger().info(f"Send goal: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f} ({frame_id})")

        done_evt = threading.Event()
        result_holder = {"status": None, "accepted": None}

        def _on_goal_response(fut):
            try:
                gh = fut.result()
                if gh is None or (not gh.accepted):
                    result_holder["accepted"] = False
                    done_evt.set()
                    return

                result_holder["accepted"] = True
                self.get_logger().info("Goal accepted. Waiting result...")

                rfut = gh.get_result_async()

                def _on_result(rf):
                    try:
                        res = rf.result()
                        result_holder["status"] = None if res is None else int(res.status)
                    finally:
                        done_evt.set()

                rfut.add_done_callback(_on_result)

            except Exception as e:
                self.get_logger().error(f"Goal response exception: {repr(e)}")
                result_holder["accepted"] = False
                done_evt.set()

        send_future = self._client.send_goal_async(goal_msg)
        send_future.add_done_callback(_on_goal_response)

        nav_timeout = 300.0
        ok = done_evt.wait(timeout=nav_timeout)
        if not ok:
            self.get_logger().error(f"Navigation timeout after {nav_timeout}s.")
            return False

        if result_holder["accepted"] is not True:
            self.get_logger().error("Goal rejected / no goal_handle.")
            return False

        status = result_holder["status"]
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Navigation SUCCEEDED.")
            return True

        self.get_logger().warn(f"Navigation finished with status={status} ({STATUS_NAME.get(status, '???')})")
        return False

    # ====================== Room query & CLIP query ======================

    def query_room_callback(self, room_name: str, thresh: float = 0.25):
        db = self.room_edges
        anchors = self.room_anchors
        names = list(db.keys())
        if len(names) == 0:
            self.get_logger().warn("[room-sem] ROOM_DB is empty.")
            return None

        device = "cpu"
        tokens = self.clip_tokenizer(names).to(device)
        query_tokens = self.clip_tokenizer([room_name]).to(device)

        with torch.no_grad():
            ft = self.clip_model.encode_text(tokens)
            ft = ft / ft.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            q = self.clip_model.encode_text(query_tokens)
            q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            sims = (ft @ q.squeeze(0))
            best_score, best_idx = torch.max(sims, dim=0)

        best_score = float(best_score.item())
        best_idx = int(best_idx.item())

        if best_score < thresh:
            best_name = names[best_idx] if 0 <= best_idx < len(names) else None
            self.get_logger().warn(
                f"[room-sem] No good match for '{room_name}', best='{best_name}', score={best_score:.3f}, thresh={thresh:.3f}"
            )
            return None

        self.get_logger().info(
            f"[room-sem] Match '{room_name}' -> '{names[best_idx]}' (idx={best_idx}, score={best_score:.3f})"
        )
        return db[names[best_idx]], anchors.get(names[best_idx], None)


    def debug_all_objs(self):
        for obj in self.obj_map:
            self.get_logger().info(f"Obj: {obj.class_name}, clip_ft shape: {obj.clip_ft.shape}")
            obj_min_x = obj.bbox_2d.min_bound[0]
            obj_min_y = obj.bbox_2d.min_bound[1]
            obj_max_x = obj.bbox_2d.max_bound[0]
            obj_max_y = obj.bbox_2d.max_bound[1]

            left_down_map = np.array([obj_min_x, obj_min_y])
            right_down_map = np.array([obj_max_x, obj_min_y])
            left_up_map = np.array([obj_max_x, obj_max_y])
            right_up_map = np.array([obj_min_x, obj_max_y])

            corner_list = [left_down_map, right_down_map, left_up_map, right_up_map]

            obj_pos = np.mean(corner_list, axis=0)
            self.get_logger().info(f"Obj position: {obj_pos}")

            print("============================")

    def query_callback(self, instance_query: str):
        text_queries = [instance_query]
        text_queries_tokenized = self.clip_tokenizer(text_queries).to("cpu")
        text_query_ft = self.clip_model.encode_text(text_queries_tokenized)
        text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)
        text_query_ft = text_query_ft.squeeze()

        values = []
        for obj in self.obj_map:
            values.append(torch.from_numpy(obj.clip_ft))
        map_clip_fts = torch.stack(values, dim=0).to("cpu")

        cos_sim = F.cosine_similarity(text_query_ft.unsqueeze(0), map_clip_fts, dim=-1)
        sorted_cos_sim, sorted_idx = torch.sort(cos_sim, dim=0, descending=True)

        # room filter
        with self._lock:
            room_ready = self.is_room_ready
            room_bbox = self.room_bbox

        if room_ready and room_bbox is not None:
            min_x, max_x, min_y, max_y = room_bbox
        else:
            min_x, max_x, min_y, max_y = -100, 100, -100, 100

        for cos_val, idx in zip(sorted_cos_sim.tolist(), sorted_idx.tolist()):
            obj_min_x = self.obj_map[idx].bbox_2d.min_bound[0]
            obj_min_y = self.obj_map[idx].bbox_2d.min_bound[1]
            obj_max_x = self.obj_map[idx].bbox_2d.max_bound[0]
            obj_max_y = self.obj_map[idx].bbox_2d.max_bound[1]

            if not (min_x <= obj_min_x <= max_x and
                    min_y <= obj_min_y <= max_y and
                    min_x <= obj_max_x <= max_x and
                    min_y <= obj_max_y <= max_y):
                continue

            left_down_map = np.array([obj_min_x, obj_min_y])
            right_down_map = np.array([obj_max_x, obj_min_y])
            left_up_map = np.array([obj_max_x, obj_max_y])
            right_up_map = np.array([obj_min_x, obj_max_y])

            corner_list = [left_down_map, right_down_map, left_up_map, right_up_map]
            self.get_logger().info(
                f"[query] '{instance_query}' hit idx={idx} sim={cos_val:.3f} name={self.obj_map[idx].class_name}"
            )
            return corner_list

        self.get_logger().warn(f"[query] '{instance_query}' found nothing after room filter.")
        return None

    def load_results(self):
        load_dir = self.load_dir
        if not os.path.exists(load_dir):
            self.get_logger().error(f"{load_dir} does not exist.")
            sys.exit(1)

        self.get_logger().info(f"Loading saved obj results from: {load_dir}")

        obj_map = []
        pkl_files = sorted([f for f in os.listdir(self.load_dir) if f.endswith(".pkl")])
        for file in pkl_files:
            obj_results_path = os.path.join(self.load_dir, file)
            loaded_obj = BaseObject.load_from_disk(obj_results_path)
            obj_map.append(loaded_obj)

        self.get_logger().info(f"Successfully loaded {len(obj_map)} objects")
        self.obj_map = obj_map

    def init_clip(self):
        self.get_logger().info("Loading CLIP model")
        clip_model_name = "ViT-B-32"
        pretrained_path = self.clip_model_path  # 修改为实际路径
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=pretrained_path, device="cpu"
        )
        device = "cpu"
        self.clip_model = self.clip_model.to(device)
        self.clip_model.eval()
        self.clip_tokenizer = open_clip.get_tokenizer(clip_model_name)
        self.get_logger().info(f"Using device: {device}, Done initializing CLIP model.")

    # ====================== 任务检查 & recovery ======================

    def rgb_callback(self, msg: Image):
        if self.rgb_check_end is not True:
            # 不是任务检查，直接丢弃
            return

        self.get_logger().info("Checking rgb image!")
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # 任务检查
        with self.image_lock:
            self.latest_image = cv_image.copy()
            self.get_logger().info("Updated latest image for task checking!")
            self.rgb_check_end = False


    def send_image_to_vlm(self, cv_image: np.ndarray, query: str, system_prompt: str = None) -> dict:
        """
        将图像发送到VLM并获取响应

        Args:
            cv_image: OpenCV图像 (BGR格式)
            query: 查询文本
            system_prompt: 系统提示词

        Returns:
            VLM的响应字典
        """
        try:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb_image)

            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # 构建VLM请求
            if system_prompt is None:
                system_prompt = "You are a helpful vision-language assistant. Analyze the image and answer questions about it."

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                    ],
                },
            ]

            payload = {
                "model": "qwen-vl-plus",  # VLM
                "messages": messages,
                "max_tokens": 1000,
                "temperature": 0.1,
            }

            self.get_logger().info(f"Sending image ({cv_image.shape}) and query to VLM...")
            self.get_logger().info(f"Query: {query}")

            start_time = time.time()
            base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            api_key = self.cfg["api_key"]
            if not api_key:
                raise ValueError("请在 config 里设置 api_key")

            url = f"{base_url}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            response = requests.post(url, headers=headers, json=payload, timeout=30)
            elapsed_time = time.time() - start_time

            if response.status_code != 200:
                return {"success": False, "error": f"HTTP {response.status_code}", "response_text": response.text}

            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                self.get_logger().info(f"VLM response received in {elapsed_time:.2f}s")
                self.get_logger().info(f"VLM Response: {content}")
                return {
                    "success": True,
                    "response": content,
                    "raw_response": result,
                    "processing_time": elapsed_time,
                    "image_shape": cv_image.shape,
                }

            return {"success": False, "error": "Invalid response format", "raw_response": result}

        except requests.exceptions.Timeout:
            return {"success": False, "error": "Request timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _sanitize_filename(self, s: str) -> str:
        # 避免中文/空格/特殊字符导致的问题
        s = str(s) if s is not None else "None"
        keep = []
        for ch in s:
            if ch.isalnum() or ch in ("-", "_"):
                keep.append(ch)
            elif ch.isspace():
                keep.append("_")
            else:
                keep.append("_")
        return "".join(keep)[:80]

    def _save_rgb_snapshot(self, cv_image: np.ndarray, prefix: str = "check") -> Optional[str]:
        """
        保存一张 BGR OpenCV 图像到本地，返回保存路径；失败返回 None
        """
        try:
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            ms = int((time.time() % 1) * 1000)
            tgt = self._sanitize_filename(getattr(self, "target_name", "None"))
            room = self._sanitize_filename(getattr(self, "room", "None"))

            filename = f"{prefix}_{ts}_{ms:03d}_room-{room}_tgt-{tgt}.jpg"
            save_path = os.path.join("/data/DualMap/applications", filename)

            # 直接写 BGR 即可
            ok = cv2.imwrite(save_path, cv_image)
            if not ok:
                self.get_logger().error(f"cv2.imwrite failed: {save_path}")
                return None

            self.get_logger().info(f"Saved RGB snapshot: {save_path}")
            return save_path
        except Exception as e:
            self.get_logger().error(f"Save RGB snapshot exception: {repr(e)}")
            return None

    def check_task(self) -> bool:
        """
        vlm 检查的容器，但在 worker 线程里执行，不会阻塞 ROS executor。
        """
        print("+++++++++++++++++++ Task Checking +++++++++++++++++++++++")

        # 触发一次 rgb 更新
        if self.latest_image is None:
            self.rgb_check_end = True

        while self.latest_image is None:
            self.get_logger().warn("Waiting for latest image for task checking...")
            time.sleep(0.5)

        cv_image = self.latest_image.copy()
        self._save_rgb_snapshot(cv_image, prefix="check")
        print("Save!")

        query = (
            f"Target object is {self.target_name}. "
            "Is the target object in the image? Answer 'Yes' or 'No'"
        )

        vlm_res = self.send_image_to_vlm(cv_image, query)
        if not vlm_res.get("success", False):
            self.get_logger().error(f"VLM check failed: {vlm_res.get('error')}")
            return False

        ans = str(vlm_res.get("response", "")).strip().lower()
        return ans == "yes" or ans == "yes."

    def turn_around(self, turn_cnt: int):
        yaw_list = [0.0, math.pi / 2, math.pi, -math.pi / 2]
        cx, cy = self.current_x, self.current_y
        for yaw in yaw_list:
            ok = self._goto_point(cx, cy, yaw=yaw, frame_id="map", wait_timeout=5.0)
            if turn_cnt == 0:
                # NOTE: 只在第一次转圈时保留rgb，给VLM做推理
                self.get_logger().info(f"Turn to yaw={yaw:.2f}, collecting image...")

                # 图像采集
                self.rgb_check_end = True
                time.sleep(0.5)

                with self.image_lock:
                    if self.latest_image is not None:
                        cv_image = self.latest_image.copy()
                        self.turn_around_imgs[yaw] = cv_image

                self.rgb_check_end = False

            if not ok:
                self.get_logger().warn("Turn around failed, continue...")
            time.sleep(1.0)

    def run_recovery(self):

        # NOTE: recovery 策略：转10圈结束/成功获得目标位置，VLM分析一次，如果没有dualmap结果，返回VLM的yaw
        self.get_logger().warn("Running recovery...TURNING...")

        success = False
        turn_cnt = 0
        self.turn_around_imgs.clear()

        while turn_cnt < 10:
            self.turn_around(turn_cnt)

            if turn_cnt == 0:
                # NOTE: 尝试用VLM分析图像，找目标物体
                vlm_res = self.get_vlm_answer()
                print("VLM RES:", vlm_res)

            turn_cnt += 1

            if self.target_x is not None and self.target_y is not None:
                success = True
                break

        if not success:
            if vlm_res is None:
                # 把存储的图像发给 VLM 看看，转向看见的角度
                ok = self._goto_point(self.current_x, self.current_y, yaw=vlm_res, frame_id="map", wait_timeout=5.0)
                # LOG2：Recovery 找到物体记录 <<<
                if ok:
                    write_log(f"Object Found: {self.target_name} ")

            else:
                # NOTE: recovery 失败
                self.get_logger().warn("Recovery failed, return to origin.")
            return
        else:
            # SJQ: 成功获得目标位置，前往目标位置并面向目标
            self.get_logger().warn(f"Recovery: target_x {self.target_x:.1f} target_y {self.target_y:.1f} from dualmap.")
            ok = self._goto_and_face_target(self.target_x, self.target_y, self.current_x, self.current_y)
            # LOG2：Recovery 找到物体记录 <<<
            if ok:
                write_log(f"Object Found: {self.target_name} ")
        # while True:
        #     if self.target_x is not None and self.target_y is not None:
        #         break
        #     time.sleep(0.5)
        # if abs(self.target_x - 0.0) < 1e-3 and abs(self.target_y - 0.0) < 1e-3:
        #     # NOTE: 如果recover失败应当返回0,0，不如此返回，改为转10圈失败
        #     # fail 了，返回原点
        #     self.get_logger().warn("Recovery failed, return to origin.")
        #     self._goto_point(0.0, 0.0, yaw=0.0, frame_id="map", wait_timeout=5.0)
        #     return


    def get_vlm_answer(self):
        if self.turn_around_imgs:

            self.get_logger().info(f"Analyzing {len(self.turn_around_imgs)} images from different angles...")

            # 分析每个角度的图像
            for yaw, img in self.turn_around_imgs.items():
                if self.target_name is not None:
                    # 使用 VLM 分析图像中是否有目标物体
                    query = f"Is the {self.target_name} clearly visible in this image? Answer 'Yes' or 'No'."
                    result = self.send_image_to_vlm(img, query)

                    if result.get("success", False):
                        response = str(result.get("response", "")).strip().lower()
                        self.get_logger().info(f"Angle {math.degrees(yaw):.0f}° VLM response: {response}")

                        # 解析 VLM 响应
                        if "yes" in response or "yes." in response:
                            # 尝试从响应中提取置信度
                            angle = yaw
                            self.get_logger().info(f"Found target at angle {math.degrees(yaw):.0f} rad")

                            return angle
                else:
                    self.get_logger().info(f"Target name is None, skip VLM analysis.")
                    return



    def calculate_yaw_to_target(self, target_x, target_y) -> float:
        dx = target_x - self.current_x
        dy = target_y - self.current_y
        target_angle = math.atan2(dy, dx)
        angle_diff = target_angle - self.current_yaw
        return math.atan2(math.sin(angle_diff), math.cos(angle_diff))

    def destroy_node(self):
        self._shutdown_event.set()
        self._task_event.set()
        super().destroy_node()


# def pop_hazard2yaml(hazard_path: str, corners: list):
#     left_down, right_down, left_up, right_up = corners

#     yaml_content = """bboxes:
#   - frame: map
#     corners:
#       - [{:.1f}, {:.1f}]
#       - [{:.1f}, {:.1f}]
#       - [{:.1f}, {:.1f}]
#       - [{:.1f}, {:.1f}]
# resolution: 0.01
# topic: /keepout_filter_mask
# publish_rate: 0.1
# target_frame: map
# max_cells: 400000
#     """.format(
#             left_down[0], left_down[1],
#             right_down[0], right_down[1],
#             left_up[0], left_up[1],
#             right_up[0], right_up[1],
#         )

#     with open(hazard_path, "w") as f:
#         f.write(yaml_content)

#     print(f"[query] Pumped semantic hazard to yaml: {hazard_path}")


def parse_command_with_qwen(cfg_path:str, user_query: str):
    """
    使用 Qwen 官方 API 解析用户指令，提取导航参数。

    Args:
        user_query: 用户输入的指令文本

    Returns:
        包含 target_room, target_object, related_object 的字典
    """
    # 从cfg读取API密钥和基础URL
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    api_key = cfg["api_key"]
    base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    if not api_key:
        raise ValueError("请设置 QWEN_API_KEY 环境变量")

    # 构建与OpenAI兼容的请求格式
    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    #  prompt 确保解析格式一致
    prompt = f"""
请从以下用户指令中提取三个关键要素：
用户指令：“{user_query}”
请提取：
1. **目标房间** (target_room)：要去的房间类型（如卧室、厨房、客厅等）
2. **相关物体** (related_object)：与目标物体相关的物体，可能是家具的类型（如床、桌子等）
3. **寻找物品** (target_object)：需要在目标房间找到的物品
规则：
- 如果某项信息不明确或不存在，请返回 "None"
- 物品名称应该是具体的（如"被子"而不是"那个被子"），一定会有需要找到的物体！！！
- 相关物体的意思是，例如"去卧室拿床上的被子"，相关物体就是“床”，如果没有相关物体，请返回 "None"，相关物体如果存在一定是在命令中提到的
- 有可能不存在相关物体！！比如去书房找瓶子，就没有相关物体，你应当对 related_object 返回"None"!!!
- 只返回JSON格式，不要有其他文本
- 房间只可能是bedroom，childroom，kitchen，livingroom 中的一个，名称必须原样返回 4者中的一个，如 bedroom！！！
- 返回的物体名称需要是英文的类型，比如输出的指令是“床”，你应当返回“bed”
输出格式：
{{
    "target_room": "房间名称",
    "related_object": "物品名称",
    "target_object": "物品名称"
}}
现在请生成JSON：
"""

    # 构建请求体
    payload = {
        "model": "qwen-max",  # MODEL qwen-turbo, qwen-plus
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,  # 低温度以保证输出稳定性
        "top_p": 0.8,
        "stream": False,
        "max_tokens": 1024
    }

    try:
        # 发送请求
        response = requests.post(url,
                                 headers=headers,
                                 data=json.dumps(payload),
                                 timeout=30)
        response.raise_for_status()  # 检查HTTP错误

        result = response.json()

        # 解析响应
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]

            # 清理响应内容，提取JSON部分
            content = content.strip()

            # 查找JSON对象
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                json_str = json_match.group()
                parsed_data = json.loads(json_str)

                # 确保所有键都存在，缺失的键设为"None"
                required_keys = ["target_room", "related_object", "target_object"]
                for key in required_keys:
                    if key not in parsed_data:
                        parsed_data[key] = "None"

                return parsed_data
            else:
                raise ValueError("API响应中未找到有效的JSON格式")
        else:
            raise ValueError("API响应格式异常")

    except requests.exceptions.RequestException as e:
        print(f"API请求失败: {e}")
        # 返回默认值或抛出异常，根据你的错误处理策略决定
        return {
            "target_room": "None",
            "related_object": "None",
            "target_object": "None"
        }
    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {e}")
        return {
            "target_room": "None",
            "related_object": "None",
            "target_object": "None"
        }


def main():
    # 从配置读取
    cfg_path = "/home/tang123/Documents/DualMap/config/query/query_task_2_3.yaml"

    # 读取指令
    query_text = input("请输入指令：")
    # LOG1
    write_log(f"Start: Command received - '{query_text}'")
    qwen_result = parse_command_with_qwen(cfg_path, query_text)


    # DEBUG: 免解析指令
    # target_room = "bed room"
    # target_name = "bed"
    # related_object = "bed"
    # avoid_hazard = "None"

    target_room = qwen_result["target_room"]
    target_name = qwen_result["target_object"]
    related_object = qwen_result["related_object"]
    # avoid_hazard = qwen_result["avoid_object"]



    # 初始化ROS和Node
    rclpy.init()
    node = TaskSubscriber(cfg_path)

    # 等待初始化
    time.sleep(1)

    # if avoid_hazard != "None":
    #     node._hazard_cb(avoid_hazard)
    #     print(f"poped HAZARD: {avoid_hazard}")
    # print("************************************************")

    # 设置目标信息
    if target_room != "None":
        node.room = target_room
        node._room_cb(target_room)
        print(f"目标房间: {target_room}")

        # 等待房间准备完成
        wait_start = time.time()
        while not node.is_room_ready:
            if time.time() - wait_start > 5.0:
                print("等待房间准备超时！")
                break
            time.sleep(0.1)
        print("ROOM READY!")

    print("************************************************")

    if related_object == "None":
        node.related_object_name = "None"
        node.target_name = target_name
        node._target_cb(target_name)
    else:
        node.related_object_name = related_object
        node.target_name = target_name
        node._related_obj_cb(related_object)

    # 启动执行器
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()