"""
Docstring for applications.offline_local_map_query
dualmap 主机端执行，订阅目标物体名称，基于离线构建的local map进行目标位置查询
发布目标位置，避障物包围盒
"""

import os

os.environ["DISPLAY"] = ""
import sys
import time

import yaml
import numpy as np
import open_clip
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))        # applications/
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)                    # DualMap/
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.object import BaseObject
from mobileclip.modules.common.mobileone import reparameterize_model

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
import math
from nav_msgs.msg import OccupancyGrid, Odometry

from typing import Dict, Optional
from action_msgs.msg import GoalStatus, GoalStatusArray


def uuid_to_hex(goal_id_msg) -> str:
    # goal_id_msg is unique_identifier_msgs/msg/UUID
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


class Nav2GoalStatusMonitor(Node):
    def __init__(self, action_name: str):
        super().__init__("nav2_goal_status_monitor")
        self._action_name = action_name.rstrip("/")
        self._status_topic = f"{self._action_name}/_action/status"
        self._last_status_by_goal: Dict[str, int] = {}

        self.create_subscription(GoalStatusArray, self._status_topic, self._on_status, 10)
        self.get_logger().info(f"Monitoring Nav2 action status topic: {self._status_topic}")

    def _on_status(self, msg: GoalStatusArray) -> None:
        if not msg.status_list:
            # Still print occasionally so user knows it's alive
            self.get_logger().info("No active goals.")
            return

        for st in msg.status_list:
            goal_hex = uuid_to_hex(st.goal_info.goal_id)
            status_code = int(st.status)
            prev = self._last_status_by_goal.get(goal_hex)
            if prev == status_code:
                continue

            self._last_status_by_goal[goal_hex] = status_code
            name = STATUS_NAME.get(status_code, str(status_code))
            self.get_logger().info(f"goal_id={goal_hex} status={name}")


def yaw_to_quaternion(yaw: float):
    # Assuming roll=pitch=0
    qz = math.sin(yaw * 0.5)
    qw = math.cos(yaw * 0.5)
    return 0.0, 0.0, qz, qw



class TaskSubscriber(Node):
    def __init__(self, cfg_path: str):
        super().__init__("nav2_goal_sender")
        with open(cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # 读取房间名称
        self.room_subscription = self.create_subscription(String, "target_room",self._room_cb, 10)

        self.related_obj_subscription = self.create_subscription(String, "related_object", self.get_related_obj_position, 10)
        self.subscription = self.create_subscription(String, "target_name", self.get_target_position, 10)

        self.hazard_subscription = self.create_subscription(String, "semantic_hazard", self.get_hazard_position, 10)

        self.costmap_sub = self.create_subscription(OccupancyGrid, "/global_map/cost_map", self.costmap_callback, 10)

        self.position_sub = self.create_subscription(Odometry, "/odom", self.position_callback, 10)

        self._action_name = "/navigate_to_pose"
        self._client = ActionClient(self, NavigateToPose, self._action_name)

        # monitor nav2 status
        self._status_topic = f"{self._action_name}/_action/status"
        self._last_status_by_goal: Dict[str, int] = {}

        self.create_subscription(GoalStatusArray, self._status_topic, self._on_status, 10)
        self.get_logger().info(f"Monitoring Nav2 action status topic: {self._status_topic}")

        self.goal_id = None

        self.load_dir = None
        self.target_name = None
        self.obj_map = None
        self.latest_costmap = None
        self.recovery_cnt = 0

        self.load_results()
        self.init_clip()

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0


        self.is_room_ready = False
        self.is_related_position = False
        self.is_arrived = False

        self.is_task_complete = False

        # 输入格式：min_x, max_x, min_y, max_y
        self.room_edges = {
            "bedroom": [1.65, 6.45, -4.4, -0.8],
            "studyroom": [1.8, 6.6, 0.7, 3.1]
        }

        # 手动给定目标，测试本地clip部分
        # self.test_clip_offline("bed Room", "chair")
        # print("test end!")


    def _on_status(self, msg: GoalStatusArray) -> None:
        if not msg.status_list:
            # Still print occasionally so user knows it's alive
            self.get_logger().info("No active goals.")
            return

        for st in msg.status_list:
            goal_hex = uuid_to_hex(st.goal_info.goal_id)
            status_code = int(st.status)
            prev = self._last_status_by_goal.get(goal_hex)
            if prev == status_code:
                continue

            self._last_status_by_goal[goal_hex] = status_code
            name = STATUS_NAME.get(status_code, str(status_code))

            # flag 标识是否到达目标位置
            if status_code == GoalStatus.SUCCEEDED:
                self.is_arrived = True

            self.get_logger().info(f"goal_id={goal_hex} status={name}")

    def costmap_callback(self, msg: OccupancyGrid):
        self.latest_costmap = msg

    def position_callback(self, msg: Odometry):
        # 获取x, y坐标
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        # 从四元数提取yaw角
        orientation = msg.pose.pose.orientation
        qx = orientation.x
        qy = orientation.y
        qz = orientation.z
        qw = orientation.w
        
        # 计算yaw角（绕Z轴的旋转）
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 
                        qw * qw + qx * qx - qy * qy - qz * qz)
        
        # 存储到类变量中
        self.current_x = x
        self.current_y = y
        self.current_yaw = yaw
    
        # 打印调试信息
        print(f"Robot Position: x={x:.3f}, y={y:.3f}, yaw={math.degrees(yaw):.2f}°")

    def _room_cb(self, msg: String):
        self.room = msg.data

        target_room_bbox = self.query_room_callback(self.room)

        self.room_bbox = target_room_bbox

        self.is_room_ready = True
        print("room ready!")
        print(self.room_bbox)
  


    def query_room_callback(self, room_name: str, thresh: float = 0.25):
        """
        语义匹配版本：用 CLIP text embedding 匹配 ROOM_DB['room_name']
        输入: room_name (topic 字符串)
        输出: (room_bbox[best_idx])；失败返回 (None)
        """
        db = self.room_edges
        names = list(db.keys())
        print(names)
        print("---------------------------")

        if len(names) == 0:
            self.get_logger().warn("[room-sem] ROOM_DB is empty.")
            return None, -1, -1.0

        device = "cuda"  # 你当前代码就是固定 cuda
        # 1) 编码所有候选 room name
        tokens = self.clip_tokenizer(names).to(device)
        # 2) 编码 query
        query_tokens = self.clip_tokenizer([room_name]).to(device)

        with torch.no_grad():
            ft = self.clip_model.encode_text(tokens)                 # (N, D)
            ft = ft / ft.norm(dim=-1, keepdim=True).clamp_min(1e-6)  # normalize
            q = self.clip_model.encode_text(query_tokens)            # (1, D)
            q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)

            sims = (ft @ q.squeeze(0))                               # (N,)
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

    def send_goal(self, x: float, y: float, yaw: float, frame_id: str, wait_timeout: float) -> bool:
        if not self._client.wait_for_server(timeout_sec=wait_timeout):
            self.get_logger().error(
                f"NavigateToPose action server not available: '{self._action_name}'. "
                f"(waited {wait_timeout}s)"
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
            f"Sending Nav2 goal to '{self._action_name}': x={x:.3f}, y={y:.3f}, yaw={yaw:.3f} ({frame_id})"
        )

        send_future = self._client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()

        if goal_handle is None:
            self.get_logger().error("Failed to send goal (no goal_handle).")
            return False

        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected by server.")
            return False

        self.get_logger().info("Goal accepted. Waiting for result...")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()

        if result is None:
            self.get_logger().error("No result returned.")
            return False

        status = result.status
        # 4 == SUCCEEDED (action_msgs/msg/GoalStatus)
        if status == 4:
            self.get_logger().info("Navigation SUCCEEDED.")
            return True

        self.get_logger().warn(f"Navigation finished with status={status}.")
        return False


    def load_results(self):

        load_dir = (
            "/home/cycl/code_workspace/DualMap/output/20260107_220514/global_map"
        )
        if not os.path.exists(load_dir):
            print(f"Error: {load_dir} does not exist.")
            sys.exit(1)

        print(("Loading saved obj results from: {}".format(load_dir)))
        self.load_dir = load_dir

        obj_map = []
        pkl_files = sorted([f for f in os.listdir(self.load_dir) if f.endswith('.pkl')])

        for file in pkl_files:
            if file.endswith(".pkl"):
                obj_results_path = os.path.join(self.load_dir, file)
                # object construction
                loaded_obj = BaseObject.load_from_disk(obj_results_path)
                obj_map.append(loaded_obj)
        print(f"Successfully loaded {len(obj_map)} objects")
        self.obj_map = obj_map
        print(f"Obj Map length: %d" % len(obj_map))

        # for obj in obj_map:
        #     print(obj.class_name)


    def init_clip(self):
        # traverse the .pkl in the directory to get constructed maps

        ### Init of CLIP
        print("Loading CLIP model")

        clip_model_name = "MobileCLIP-S2"
        pretrained = "datacompdr"
        self.clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=pretrained
        )
        device = "cuda"
        self.clip_model = self.clip_model.to(device)
        self.clip_model.eval()
        # Only reparameterize if the model is MobileCLIP
        # if "MobileCLIP" in self.cfg.clip.model_name:
        print("==> Opening mobileclip")
        self.clip_model = reparameterize_model(self.clip_model)

        self.clip_tokenizer = open_clip.get_tokenizer(clip_model_name)
        print("Done initializing CLIP model.")


    def get_target_position(self, msg):
        """获取目标物体位置考虑是否发布"""

        self.target_name = msg.data
        print(f"Received target name: {self.target_name}")

        print("==> target object")

        corner_list = self.query_callback(self.target_name)
        target_position = np.array(corner_list).mean(axis=0)
        print(f"[query] target position: {target_position}")
        target_x = target_position[0]
        target_y = target_position[1]
        target_yaw = 0.0  
        frame_id = "map"
        wait_timeout = 5.0

        self.target_x = target_x
        self.target_y = target_y

        # 如果没有相关物体，就直接发布目标物体位置
        if not self.is_related_position:
            self.is_arrived = False
            self.send_goal(target_x, target_y, target_yaw, frame_id, wait_timeout)

            # 到地点就转向
            while not self.is_arrived:
                time.sleep(1)
                print("waiting for TARGET object goal to be reached...")

            print("target object position", self.target_x, self.target_y)
            # 获取目标物体位置
            delta_yaw = self.calculate_yaw_to_target(self.target_x, self.target_y)

            print("turning to target", delta_yaw)
            self.is_arrived = False
            self.send_goal(0, 0, delta_yaw, frame_id, wait_timeout)

            while not self.is_arrived:
                time.sleep(1)
                print("TURNING to TARGET object goal...")

            is_complete = self.check_task()

            if not is_complete:
                self.run_recovery()


        print("========== TARGET OBJ GOAL SEND END ===========")


    def recover_get_target_position(self, target_name):
        """重新获取目标物体位置考虑是否发布"""

        self.target_name = target_name
        print(f"Received RECOVERY target name: {self.target_name}")

        print("==> RECOVERY target object")

        corner_list = self.query_callback(self.target_name)
        target_position = np.array(corner_list).mean(axis=0)
        print(f"[query] RECOVERY target position: {target_position}")
        target_x = target_position[0]
        target_y = target_position[1]
        target_yaw = 0.0  
        frame_id = "map"
        wait_timeout = 5.0

        self.target_x = target_x
        self.target_y = target_y

        # 如果没有相关物体，就直接发布目标物体位置
        if not self.is_related_position:
            self.is_arrived = False
            self.send_goal(target_x, target_y, target_yaw, frame_id, wait_timeout)

            # 到地点就转向
            while not self.is_arrived:
                time.sleep(1)
                print("waiting for RECOVERY TARGET object goal to be reached...")

            print("RECOVERY target object position", self.target_x, self.target_y)
            # 获取目标物体位置
            delta_yaw = self.calculate_yaw_to_target(self.target_x, self.target_y)

            print("turning to target", delta_yaw)
            self.is_arrived = False
            self.send_goal(0, 0, delta_yaw, frame_id, wait_timeout)

            while not self.is_arrived:
                time.sleep(1)
                print("TURNING to RECOVERY TARGET object goal...")

            is_complete = self.check_task()

            return is_complete

        print("========== RECOVERY TARGET OBJ GOAL SEND END ===========")
        

    def check_task(self):
        """检查任务是否完成，如果没完成进入recovery"""

        # TODO: 改成VLM判定
        complete = input("If task complete, enter '1/0': ")

        self.is_task_complete = True if complete == '1' else False

        if not self.is_task_complete:
            return False

        else:
            print("Task complete, exiting...")

            # TODO: 改成回到原点
            sys.exit(0)

    def run_recovery(self):
        """运行recovery流程"""
        print("Running recovery...")

        self.yaw_list = [0.0, np.pi/2, np.pi, -np.pi/2]

        for yaw in self.yaw_list:

            # 原地自转，重新建图
            self.is_arrived = False
            
            self.send_goal(self.current_x, self.current_y, yaw, "map", 5.0)
            while not self.is_arrived:
                time.sleep(1)
                print("TURNING TO recovery mapping goal...")

        self.cnt_recovery = 0
        while self.cnt_recovery < 3:
            self.is_related_position = False

            is_recovery_success = self.recover_get_target_position(self.target_name)

            if not is_recovery_success:
                self.cnt_recovery += 1
                print("Recovery failed, try again...")
                continue

        print("Recovery failed, exiting...")
        sys.exit(0)
            

    def test_room_offline(self, room_name: String):
        self.room = room_name

        target_room_bbox = self.query_room_callback(self.room)

        self.room_bbox = target_room_bbox

        self.is_room_ready = True
        print("room ready!")
        print(self.room_bbox)


    def test_clip_offline(self, room_name:str, target: str):
        """
        手动输入目标测试检索流程
        """

        print("start offline test!")
        if room_name:
            self.test_room_offline(room_name)

        corner_list = self.query_callback(target)
        target_position = np.array(corner_list).mean(axis=0)
        print(f"[query] target position: {target_position}")
        
        target_x = target_position[0]
        target_y = target_position[1]
        target_yaw = 0.0  
        frame_id = "map"
        wait_timeout = 5.0

        # self.send_goal(target_x, target_y, target_yaw, frame_id, wait_timeout)

        print("==============================")

    def get_related_obj_position(self, msg):

        self.related_object_name = msg.data
        print(f"Received related object name: {self.related_object_name}")

        print("==> related object")
        related_corner_list = self.query_callback(self.related_object_name)
        print(f"[query] related object: {self.related_object_name}")
        related_position = np.array(related_corner_list).mean(axis=0)

        self.is_related_position = True

        related_x = related_position[0]
        related_y = related_position[1]
        related_yaw = 0.0
        frame_id = "map"
        wait_timeout = 5.0

        # 发布相关物体位置
        self.is_arrived = False
        self.send_goal(related_x, related_y, related_yaw, frame_id, wait_timeout)

        while not self.is_arrived:
            time.sleep(1)
            print("waiting for RELATED object goal to be reached...")

        print("target object position", self.target_x, self.target_y)
        # 获取目标物体位置
        delta_yaw = self.calculate_yaw_to_target(self.target_x, self.target_y)

        print("turning to target", delta_yaw)
        self.is_arrived = False
        self.send_goal(0, 0, delta_yaw, frame_id, wait_timeout)

        print("======== RELATED OBJ GOAL SEND END ============")

        while not self.is_arrived:
            time.sleep(1)
            print("TURNING to TARGET object goal...")

        is_complete = self.check_task()

        if not is_complete:
            self.run_recovery()


    def calculate_yaw_to_target(self, target_x, target_y):
        """
        计算机器人需要旋转多少角度才能面向目标点
        """
        if not hasattr(self, 'current_x'):
            return 0
        
        # 计算目标点相对于机器人的方向
        dx = target_x - self.current_x
        dy = target_y - self.current_y
        
        # 计算目标点相对于机器人的角度（全局坐标系）
        target_angle = math.atan2(dy, dx)
        
        # 计算需要旋转的角度（当前yaw到目标角度的差）
        angle_diff = target_angle - self.current_yaw
        
        # 将角度标准化到[-π, π]范围内
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
        
        return angle_diff


    def get_hazard_position(self, msg):

        # TODO: 后面接口还没配上
        self.hazard_name = msg.data
        print(f"Received hazard name: {self.hazard_name}")

        print("==> avoid object")

        avoid_corner_list = self.query_callback(self.hazard_name)
        pop_hazard2yaml(avoid_corner_list)
        print("==============================")


    def query_callback(self, instance_query):

        text_queries = [instance_query]

        text_queries_tokenized = self.clip_tokenizer(text_queries).to("cuda")
        text_query_ft = self.clip_model.encode_text(text_queries_tokenized)
        text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)
        text_query_ft = text_query_ft.squeeze()

        ## Get stacked clip feats from the map
        values = []
        for obj in self.obj_map:
            values.append(torch.from_numpy(obj.clip_ft))
        map_clip_fts = torch.stack(values, dim=0).to("cuda")

        ## claculate the cos sim between text clip and map clips
        cos_sim = F.cosine_similarity(text_query_ft.unsqueeze(0), map_clip_fts, dim=-1)

        sorted_cos_sim, sorted_idx = torch.sort(cos_sim, dim=0, descending=True)
        print("Sorted cos sim:")


        # 如果有room要求，就根据room要求筛选，否则输出全局最大可能的物体
        if self.is_room_ready:
            min_x, max_x, min_y, max_y = self.room_bbox
        else:
            min_x = -100
            max_x = 100
            min_y = -100
            max_y = 100
        
        for i, (cos_val, idx) in enumerate(zip(sorted_cos_sim.tolist(), sorted_idx.tolist())):
            
            obj_min_x = self.obj_map[idx].bbox_2d.min_bound[0]
            obj_min_y = self.obj_map[idx].bbox_2d.min_bound[1]

            obj_max_x = self.obj_map[idx].bbox_2d.max_bound[0]
            obj_max_y = self.obj_map[idx].bbox_2d.max_bound[1]

            if not (min_x <= obj_min_x <= max_x and
                    min_y <= obj_min_y <= max_y and
                    min_x <= obj_max_x <= max_x and
                    min_y <= obj_max_y <= max_y):
                continue

            else:
                print(f"{i+1}. No. {idx} : {cos_val:.3f}")
                print(
                    f"{self.obj_map[idx].class_name}, position {self.obj_map[idx].bbox_2d}, path {self.obj_map[idx].save_path}"
                )
                print("corners")

                left_down_map = np.array([obj_min_x, obj_min_y])
                right_down_map = np.array([obj_max_x, obj_min_y])
                left_up_map = np.array([obj_max_x, obj_max_y])
                right_up_map = np.array([obj_min_x, obj_max_y])

                print("===> Object index is", idx)
                print(left_down_map)
                print(right_down_map)
                print(left_up_map)
                print(right_up_map)
                corner_list = [left_down_map, right_down_map, left_up_map, right_up_map]

                return corner_list



def pop_hazard2yaml(corners: list):
    """
    pump semantic hazard to yaml, pass to navigation modual
    """
    # FLAG: 修改为目标yaml位置
    yaml_path = "keepout_bboxes.yaml"

    left_down = corners[0]
    right_down = corners[1]
    left_up = corners[2]
    right_up = corners[3]

    yaml_content = """bboxes:
  - frame: map
    corners:
      - [{:.1f}, {:.1f}]  # 左下角
      - [{:.1f}, {:.1f}]   # 右下角
      - [{:.1f}, {:.1f}]    # 右上角
      - [{:.1f}, {:.1f}]   # 左上角
    """.format(
        left_down[0], left_down[1], right_down[0], right_down[1], left_up[0], left_up[1], right_up[0], right_up[1]
    )

    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"[query] Pumped semantic hazard to yaml: {yaml_path}")


def main(cfg_path: str):

    rclpy.init()

    target_subscriber = TaskSubscriber(cfg_path)

    rclpy.spin(target_subscriber)


if __name__ == "__main__":
    import yaml
    from pathlib import Path

    cfg_path = "/home/cycl/code_workspace/DualMap/config/query_config.yaml"
    try:
        main(cfg_path)
    finally:
        rclpy.shutdown()