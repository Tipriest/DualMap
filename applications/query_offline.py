"""
dualmap 主机端执行：订阅目标/相关物体/房间等，基于离线 local map 查询位置；
并通过 Nav2 NavigateToPose 导航到目标点，并支持面向目标的旋转与recovery流程。
"""

import os
os.environ["DISPLAY"] = ""

import sys
import time
import math
import yaml
import threading
from typing import Dict, Optional, Tuple

import numpy as np
import open_clip
import torch
import torch.nn.functional as F

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry
from action_msgs.msg import GoalStatus, GoalStatusArray

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))        # applications/
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)                    # DualMap/
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.object import BaseObject
from mobileclip.modules.common.mobileone import reparameterize_model


def uuid_to_hex(goal_id_msg) -> str:
    try:
        return bytes(goal_id_msg.uuid).hex()
    except Exception:
        return str(goal_id_msg)


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
    def __init__(self, cfg_path: str, load_dir: str):
        super().__init__("nav2_goal_sender")

        with open(cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # ====== callback group：允许并发回调（配合 MultiThreadedExecutor）======
        self._cbg = ReentrantCallbackGroup()

        # ====== 订阅：回调里只更新数据，不做阻塞 ======
        self.room_subscription = self.create_subscription(
            String, "target_room", self._room_cb, 10, callback_group=self._cbg
        )
        self.related_obj_subscription = self.create_subscription(
            String, "related_object", self._related_obj_cb, 10, callback_group=self._cbg
        )
        self.subscription = self.create_subscription(
            String, "target_name", self._target_cb, 10, callback_group=self._cbg
        )
        self.hazard_subscription = self.create_subscription(
            String, "semantic_hazard", self._hazard_cb, 10, callback_group=self._cbg
        )
        self.position_sub = self.create_subscription(
            Odometry, "/odom", self.position_callback, 10, callback_group=self._cbg
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, "/map", self.map_callback, 10, callback_group=self._cbg
        )

        # ====== Nav2 Action Client ======
        self._action_name = "/navigate_to_pose"
        self._client = ActionClient(self, NavigateToPose, self._action_name, callback_group=self._cbg)


        # ====== 运行数据 ======
        self.load_dir = load_dir
        self.target_name: Optional[str] = None
        self.related_object_name: Optional[str] = None
        self.obj_map = None
        self.latest_costmap = None

        self.load_results()
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

        self.room_edges = {
            "bedroom": [1.65, 6.45, -4.4, -0.8],
            "studyroom": [1.8, 6.6, 0.7, 3.1]
        }

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


        # 主线程，开始检索，分配状态机
        self._worker = threading.Thread(target=self._task_worker, daemon=True)
        self._worker.start()

        self.get_logger().info("TaskSubscriber initialized. Waiting for topics...")

    # ====================== 回调：只做轻量更新 ======================

    def map_callback(self, msg: OccupancyGrid):
        '''
        Docstring for map_callback
        
        处理map数据，更新 map_data 和 map_info
        '''
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
        """
        将世界坐标转换为地图坐标
        """
        if self.map_info is None:
            return 0, 0
            
        resolution = self.map_info['resolution']
        origin_x = self.map_info['origin_x']
        origin_y = self.map_info['origin_y']
        
        mx = int((x - origin_x) / resolution)
        my = int((y - origin_y) / resolution)
        
        return mx, my
    
    def map_to_world(self, mx: int, my: int) -> Tuple[float, float]:
        """
        将地图坐标转换为世界坐标
        """
        if self.map_info is None:
            return 0.0, 0.0
            
        resolution = self.map_info['resolution']
        origin_x = self.map_info['origin_x']
        origin_y = self.map_info['origin_y']
        
        wx = mx * resolution + origin_x
        wy = my * resolution + origin_y
        
        return wx, wy
    
    def is_free_cell(self, mx: int, my: int) -> bool:
        """
        判断地图上的某个点是否空闲
        0: 空闲, -1: 未知, 100: 障碍物
        """
        if self.map_data is None:
            return False
            
        height, width = self.map_data.shape
        if 0 <= mx < width and 0 <= my < height:
            value = self.map_data[my, mx]
            # 0表示空闲，其他值表示障碍物或未知
            return value == 0
        return False
    
    def find_nearest_free_point(self, target_x: float, target_y: float, search_radius: float = 0.8) -> Tuple[Optional[float], Optional[float]]:
        """
        在地图上寻找距离目标点指定半径内的最近空闲点
        """
        with self._lock:
            if not self.map_received or self.map_data is None or self.map_info is None:
                self.get_logger().warn("Map not received yet, cannot find free point")
                return target_x, target_y  # 返回原始目标点
            
            map_data = self.map_data.copy()
            map_info = self.map_info.copy()
        
        resolution = map_info['resolution']
        
        # 转换目标点到地图坐标
        target_mx, target_my = self.world_to_map(target_x, target_y)
        
        # 计算搜索半径对应的地图格子数
        search_cells = int(search_radius / resolution)
        
        # 从内向外搜索空闲点
        for radius in range(0, search_cells + 1):
            free_points = []
            
            # 搜索当前半径内的所有点
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    # 检查是否在半径内
                    if dx*dx + dy*dy > radius*radius:
                        continue
                    
                    mx = target_mx + dx
                    my = target_my + dy
                    
                    # 检查是否在地图范围内
                    if 0 <= mx < map_info['width'] and 0 <= my < map_info['height']:
                        if map_data[my, mx] == 0:  # 空闲点
                            free_points.append((mx, my))
            
            # 如果有空闲点，找到最近的一个
            if free_points:
                # 计算每个点到目标点的距离，选择最近的
                distances = []
                for mx, my in free_points:
                    dist = math.sqrt((mx - target_mx)**2 + (my - target_my)**2)
                    distances.append((dist, mx, my))
                
                # 按距离排序
                distances.sort(key=lambda x: x[0])
                nearest_mx, nearest_my = distances[0][1], distances[0][2]
                
                # 转换回世界坐标
                free_x, free_y = self.map_to_world(nearest_mx, nearest_my)
                
                distance_world = math.sqrt((free_x - target_x)**2 + (free_y - target_y)**2)
                self.get_logger().info(f"Found free point at ({free_x:.3f}, {free_y:.3f}), "
                                      f"distance from target: {distance_world:.3f}m")
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


    def _room_cb(self, msg: String):
        room = msg.data
        bbox = self.query_room_callback(room)
        if bbox is None:
            self.get_logger().warn(f"Room '{room}' not matched; will not apply room filter.")
            return

        with self._lock:
            self.room = room
            self.room_bbox = bbox
            self.is_room_ready = True

        self.get_logger().info(f"Room ready: {room} bbox={bbox}")

    def _target_cb(self, msg: String):
        with self._lock:
            self.target_name = msg.data
        self.get_logger().info(f"Received target name: {self.target_name}")

        # 触发 worker 执行任务（目标为主触发）
        self._task_event.set()

    def _related_obj_cb(self, msg: String):
        with self._lock:
            self.related_object_name = msg.data
        self.get_logger().info(f"Received related object name: {self.related_object_name}")

        # 触发 worker：如果已有 target_name，则走 related->target 流程
        self._task_event.set()

    def _hazard_cb(self, msg: String):
        hazard = msg.data
        self.get_logger().info(f"Received hazard name: {hazard}")
        corners = self.query_callback(hazard)
        if corners is not None:
            pop_hazard2yaml(corners)

    # ====================== Worker：执行导航流程 ======================

    def request_exit(self, reason: str = ""):
        self.get_logger().warn(f"Request exit. reason={reason}")
        self._shutdown_event.set()
        # 让 spin() 退出
        try:
            rclpy.shutdown()
        except Exception:
            pass

    def _task_worker(self):
        """
        后台线程：等待 target/related 事件，然后执行：
        - 有 related：先去 related，再转向 target
        - 无 related：直接去 target，再转向 target
        - 失败则进入 recovery
        """
        while not self._shutdown_event.is_set():
            # 等事件
            self._task_event.wait(timeout=0.2)
            if self._shutdown_event.is_set():
                break
            if not self._task_event.is_set():
                continue

            # 取一次快照
            with self._lock:
                target_name = self.target_name
                related_name = self.related_object_name
                room_ready = self.is_room_ready
                room_bbox = self.room_bbox

            # 清事件，本轮开始执行
            self._task_event.clear()

            if not target_name:
                # 只有 related 没有 target，不执行导航
                self.get_logger().warn("Worker triggered but target_name is empty. Skip.")
                continue

            try:
                # ===== [ADD-1] 如果要求目标必须在 room 内：先等待 room_ready =====
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
                                f"{getattr(self, 'room_wait_timeout', 3.0)}s. "
                                f"Publish /target_room first."
                            )
                            # 本轮放弃，不做全局检索（避免跑到错误房间）
                            raise RuntimeError("room not ready")

                        time.sleep(0.05)

                #  1. 查询 target 位置（耗时操作放到 worker 线程里）
                corners = self.query_callback(target_name)
                if corners is None:
                    self.get_logger().error(f"Target '{target_name}' not found in map.")
                    continue

                target_pos = np.array(corners).mean(axis=0)
                target_x, target_y = float(target_pos[0]), float(target_pos[1])

                # FLAG: 查找距离最近的空闲点
                free_x, free_y = self.find_nearest_free_point(target_x, target_y, 0.8)

                with self._lock:
                    self.target_x = target_x
                    self.target_y = target_y

                self.get_logger().info(f"[query] target '{target_name}' -> ({target_x:.3f}, {target_y:.3f})")

                # 2. 如果有 related，先去 related
                if related_name:
                    rcorners = self.query_callback(related_name)
                    if rcorners is None:
                        self.get_logger().warn(f"Related '{related_name}' not found, fallback to direct target.")
                        
                        # FLAG: 导航到最近的点
                        # ok = self._goto_and_face_target(target_x, target_y)
                        ok = self._goto_and_face_target(free_x, free_y, target_x, target_y)
                    else:
                        rpos = np.array(rcorners).mean(axis=0)
                        rx, ry = float(rpos[0]), float(rpos[1])

                        # FLAG: 导航到最近的相关空闲点
                        free_rx, free_ry = self.find_nearest_free_point(rx, ry, 0.8)

                        # DEBUG: 可以在此处硬赋值related 坐标
                        self.get_logger().info(f"[query] related '{related_name}' -> ({rx:.3f}, {ry:.3f})")
                        self.get_logger().warn(f"goto related point: ({free_rx:.3f}, {free_ry:.3f})")

                        # FLAG: 导航到最近的相关空闲点
                        # ok = self._goto_point(rx, ry, yaw=0, frame_id="map", wait_timeout=5.0)
                        ok = self._goto_point(free_rx, free_ry, yaw=0, frame_id="map", wait_timeout=5.0)

                        if ok:
                            if np.sqrt(np.sum((self.current_x - self.target_x) ** 2 + (self.current_y - self.target_y) ** 2)) < 0.8:
                                # 已经到达范围
                                ok = self._face_target(target_x, target_y)
                            else:
                                self.get_logger().warn("Too far away from target, fallback to direct target.")
                                ok = self._goto_and_face_target(free_x, free_y, target_x, target_y)

                        else:
                            self.get_logger().warn("Goto related failed, fallback to direct target.")
                            ok = self._goto_and_face_target(free_x, free_y, target_x, target_y)

                else:
                    ok = self._goto_and_face_target(free_x, free_y, target_x, target_y)

                # 3. 检查任务完成（非阻塞：在 worker 线程里允许 input）
                if ok:
                    is_complete = self.check_task()
                    if not is_complete:
                        self.run_recovery(target_name)
                        self.request_exit("recovery finished after task incomplete")
                    else:
                        self.request_exit("task complete")
                else:
                    self.get_logger().warn("Navigation failed -> enter recovery")
                    self.run_recovery(target_name)
                    self.request_exit("recovery finished after nav failed")

            except Exception as e:
                self.get_logger().error(f"Worker exception: {repr(e)}")

    def _goto_and_face_target(self, free_x:float, free_y:float, tx: float, ty: float) -> bool:
        # 先去目标点，再原地转向
        ok = self._goto_point(free_x, free_y, yaw=0.0, frame_id="map", wait_timeout=5.0)
        if not ok:
            return False
        return self._face_target(tx, ty)

    def _face_target(self, tx: float, ty: float) -> bool:
        delta_yaw = self.calculate_yaw_to_target(tx, ty)
        # 原地转向：用当前位姿 x,y
        cx, cy, cyaw = self.current_x, self.current_y, self.current_yaw
        target_yaw = cyaw + delta_yaw
        self.get_logger().info(f"Face target: delta_yaw={delta_yaw:.3f} -> target_yaw={target_yaw:.3f}")
        return self._goto_point(cx, cy, yaw=target_yaw, frame_id="map", wait_timeout=5.0)

    # ====================== Nav2 Action：异步 + Event 等待 ======================

    def _goto_point(self, x: float, y: float, yaw: float, frame_id: str, wait_timeout: float) -> bool:
        """
        发送 NavigateToPose 并等待 result（线程里 wait，不阻塞 ROS 回调线程）。
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

        self.get_logger().info(
            f"Send goal: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f} ({frame_id})"
        )

        done_evt = threading.Event()
        result_holder = {"status": None, "accepted": None}

        def _on_goal_response(fut):
            try:
                gh = fut.result()
                if gh is None:
                    result_holder["accepted"] = False
                    done_evt.set()
                    return
                if not gh.accepted:
                    result_holder["accepted"] = False
                    done_evt.set()
                    return

                result_holder["accepted"] = True
                self.get_logger().info("Goal accepted. Waiting result...")

                rfut = gh.get_result_async()

                def _on_result(rf):
                    try:
                        res = rf.result()
                        if res is None:
                            result_holder["status"] = None
                        else:
                            result_holder["status"] = int(res.status)
                    finally:
                        done_evt.set()

                rfut.add_done_callback(_on_result)

            except Exception as e:
                self.get_logger().error(f"Goal response exception: {repr(e)}")
                result_holder["accepted"] = False
                done_evt.set()

        send_future = self._client.send_goal_async(goal_msg)
        send_future.add_done_callback(_on_goal_response)

        # 等待结果（这里在线程里等待，不影响 ROS 回调执行）
        nav_timeout = 300.0
        ok = done_evt.wait(timeout=nav_timeout)
        if not ok:
            self.get_logger().error(f"Navigation timeout after {nav_timeout}s.")
            return False

        if result_holder["accepted"] is not True:
            self.get_logger().error("Goal rejected / no goal_handle.")
            return False

        status = result_holder["status"]
        print(f"Navigation result status: {status} ============= ")
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Navigation SUCCEEDED.")
            return True

        self.get_logger().warn(f"Navigation finished with status={status} ({STATUS_NAME.get(status, '???')})")
        return False

    # ====================== Room query & CLIP query ======================

    def query_room_callback(self, room_name: str, thresh: float = 0.25):
        db = self.room_edges
        names = list(db.keys())
        if len(names) == 0:
            self.get_logger().warn("[room-sem] ROOM_DB is empty.")
            return None

        device = "cuda"
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
        return db[names[best_idx]]

    def query_callback(self, instance_query: str):
        text_queries = [instance_query]
        text_queries_tokenized = self.clip_tokenizer(text_queries).to("cuda")
        text_query_ft = self.clip_model.encode_text(text_queries_tokenized)
        text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)
        text_query_ft = text_query_ft.squeeze()

        values = []
        for obj in self.obj_map:
            values.append(torch.from_numpy(obj.clip_ft))
        map_clip_fts = torch.stack(values, dim=0).to("cuda")

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
            self.get_logger().info(f"[query] '{instance_query}' hit idx={idx} sim={cos_val:.3f} name={self.obj_map[idx].class_name}")
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
        clip_model_name = "MobileCLIP-S2"
        pretrained = "datacompdr"
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=pretrained
        )
        device = "cuda"
        self.clip_model = self.clip_model.to(device)
        self.clip_model.eval()
        self.clip_model = reparameterize_model(self.clip_model)
        self.clip_tokenizer = open_clip.get_tokenizer(clip_model_name)
        self.get_logger().info("Done initializing CLIP model.")

    # ====================== 任务检查 & recovery ======================

    def check_task(self) -> bool:
        """
        TODO:
        这里后续会换成 VLM 判定。当前保留交互，但在 worker 线程里执行，不会阻塞 ROS executor。
        """
        try:
            complete = input("If task complete, enter '1/0': ").strip()
            return complete == "1"
        except Exception:
            return False
        
    def turn_around(self):
        yaw_list = [0.0, math.pi/2, math.pi, -math.pi/2]
        cx, cy = self.current_x, self.current_y
        for yaw in yaw_list:
            ok = self._goto_point(cx, cy, yaw=yaw, frame_id="map", wait_timeout=5.0)
            if not ok:
                self.get_logger().warn("Turn around failed, continue...")
            time.sleep(1.0)

    def run_recovery(self, target_name: str):
        self.get_logger().warn("Running recovery...")

        self.turn_around()
        
        cnt = 0
        while cnt < 3:
        # 重新尝试检索+导航若干次
            self.get_logger().warn(f"Recovery try {cnt+1}/3 ...")
            # TODO: 重新load remap后的结果
            corners = self.query_callback(target_name)
            if corners is None:
                self.turn_around()
                cnt += 1
                continue
            else:

                pos = np.array(corners).mean(axis=0)
                tx, ty = float(pos[0]), float(pos[1])

                # 重新导航到目标位置
                free_recovery_x, free_recovery_y = self.find_nearest_free_point(tx, ty, 0.8)
                self.get_logger().info(f"Recovery to {free_recovery_x:.1f}, {free_recovery_y:.1f}")

                ok = self._goto_and_face_target(free_recovery_x, free_recovery_y, tx, ty)
                if ok:
                    self.get_logger().info("Recovery navigation success.")
                    return
                else:
                    self.get_logger().warn("Recovery navigation failed, retry...")
                    cnt += 1

        self.get_logger().error("Recovery failed. (No sys.exit here; keep node alive for next task.)")

    # ====================== 计算面向目标 yaw ======================

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


def pop_hazard2yaml(corners: list):
    yaml_path = "keepout_bboxes.yaml"
    left_down, right_down, left_up, right_up = corners

    yaml_content = """bboxes:
  - frame: map
    corners:
      - [{:.1f}, {:.1f}]
      - [{:.1f}, {:.1f}]
      - [{:.1f}, {:.1f}]
      - [{:.1f}, {:.1f}]
""".format(
        left_down[0], left_down[1],
        right_down[0], right_down[1],
        left_up[0], left_up[1],
        right_up[0], right_up[1],
    )

    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"[query] Pumped semantic hazard to yaml: {yaml_path}")


def main(cfg_path: str):
    rclpy.init()
    load_dir = "/home/cycl/code_workspace/DualMap/output/20260107_220514/global_map"
    node = TaskSubscriber(cfg_path, load_dir)

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    cfg_path = "/home/cycl/code_workspace/DualMap/config/query_config.yaml"
    main(cfg_path)