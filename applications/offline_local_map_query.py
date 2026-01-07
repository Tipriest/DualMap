"""
Docstring for applications.offline_local_map_query
dualmap 主机端执行，订阅目标物体名称，基于离线构建的local map进行目标位置查询
发布目标位置，避障物包围盒
"""

import os

os.environ["DISPLAY"] = ""
import sys

import yaml
import numpy as np
import open_clip
import torch
import torch.nn.functional as F

sys.path.append("/home/tipriest/Documents/DualMap/3rdparty")

from utils.object import BaseObject
from mobileclip.modules.common.mobileone import reparameterize_model

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
import math


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
        self.subscription = self.create_subscription(String, "target_name", self.get_target_position, 10)
        self.related_obj_subscription = self.create_subscription(String, "related_object", self.get_related_obj_position, 10)
        self.hazard_subscription = self.create_subscription(String, "semantic_hazard", self.get_hazard_position, 10)


        self._action_name = "/navigate_to_pose"
        self._client = ActionClient(self, NavigateToPose, self._action_name)

        self.load_dir = None
        self.target_name = None
        self.obj_map = None

        self.load_results()
        self.init_clip()

        # 手动给定目标，测试本地clip部分
        # self.test_clip_offline("tv")
        # print("test end!")

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
        ### Loading saved results
        # if map_dir is not provided, use the default path
        # FLAG: 移植实机需处理目录
        # if os.path.exists(self.cfg.test_map_dir):
        #     load_dir = self.cfg.test_map_dir
        # else:
        #     load_dir = os.path.join(
        #         self.cfg.output_path, f"{self.cfg.dataset_name}_{self.cfg.scene_id}", "map"
        #     )
        load_dir = (
            "/home/tipriest/Documents/DualMap/output/map_results/hm3d_00829-QaLdnwvtxbs/20251225_210242/global_map"
        )
        if not os.path.exists(load_dir):
            print(f"Error: {load_dir} does not exist.")
            sys.exit(1)

        print(("Loading saved obj results from: {}".format(load_dir)))
        self.load_dir = load_dir

    def init_clip(self):
        # traverse the .pkl in the directory to get constructed maps
        obj_map = []
        for file in os.listdir(self.load_dir):
            if file.endswith(".pkl"):
                obj_results_path = os.path.join(self.load_dir, file)
                # object construction
                loaded_obj = BaseObject.load_from_disk(obj_results_path)
                obj_map.append(loaded_obj)
        print(f"Successfully loaded {len(obj_map)} objects")
        self.obj_map = obj_map

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

        print(f"Obj Map length: %d" % len(obj_map))

    def get_target_position(self, msg):
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

        self.send_goal(target_x, target_y, target_yaw, frame_id, wait_timeout)

        print("==============================")

    def test_clip_offline(self, target: str):
        """
        手动输入目标测试检索流程
        """

        print("start offline test!")
        corner_list = self.query_callback(target)
        target_position = np.array(corner_list).mean(axis=0)
        print(f"[query] target position: {target_position}")

        target_x = target_position[0]
        target_y = target_position[1]
        target_yaw = 0.0
        frame_id = "map"
        wait_timeout = 5.0

        self.send_goal(target_x, target_y, target_yaw, frame_id, wait_timeout)

        print("==============================")

    def get_related_obj_position(self, msg):
        self.related_object_name = msg.data
        print(f"Received related object name: {self.related_object_name}")

        print("==> related object")
        related_corner_list = self.query_callback(self.related_object_name)
        print(f"[query] related object: {self.related_object_name}")
        related_position = np.array(related_corner_list).mean(axis=0)

        # TODO:

        print("==============================")

    def get_hazard_position(self, msg):
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
        cos_sim = F.cosine_similarity(
            text_query_ft.unsqueeze(0), map_clip_fts, dim=-1
        )

        ## Get top k candidates
        top_k = 1
        top_k_cos_sim, top_k_idx = torch.topk(cos_sim, top_k, dim=0)
        print("Most similar object:")
        for i, (cos_val, idx) in enumerate(zip(top_k_cos_sim.tolist(), top_k_idx.tolist())):

            bbox_2d = self.obj_map[idx].bbox_2d
            min_x = bbox_2d.min_bound[0]
            min_y = bbox_2d.min_bound[1]

            max_x = bbox_2d.max_bound[0]
            max_y = bbox_2d.max_bound[1]

            left_down_map = transfromPos(np.array([min_x, min_y]))
            right_down_map = transfromPos(np.array([max_x, min_y]))
            left_up_map = transfromPos(np.array([max_x, max_y]))
            right_up_map = transfromPos(np.array([min_x, max_y]))

            print(f"{i+1}. No. {idx} : {cos_val:.3f}")
            print(
                f"{self.obj_map[idx].class_name}, position {self.obj_map[idx].bbox_2d}, path {self. obj_map[idx].save_path}"
            )

            print("corners")
            print(left_down_map)
            print(right_down_map)
            print(left_up_map)
            print(right_up_map)
            corner_list = [left_down_map, right_down_map, left_up_map, right_up_map]

            return corner_list


def transfromPos(position: np.array) -> np.array:
    """
    输入为dualmap世界坐标系读出的坐标，返回gazebo坐标系下的坐标
    """
    # return np.array([position[1], -position[0], -position[2]])
    return np.array([position[1], -position[0]])


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

    cfg_path = "/home/tipriest/Documents/DualMap/config/query_config.yaml"
    try:
        main(cfg_path)
    finally:
        rclpy.shutdown()
