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

sys.path.append("/home/tang123/DualMap/")

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

        self.subscription = self.create_subscription(String, "target_name", self.get_target_position, 10)
        self.related_obj_subscription = self.create_subscription(String, "related_object", self.get_related_obj_position, 10)
        self.hazard_subscription = self.create_subscription(String, "semantic_hazard", self.get_hazard_position, 10)

        self.costmap_sub = self.create_subscription(OccupancyGrid, "/global_map/cost_map", self.costmap_callback, 10)
        # TODO: odom话题是什么
        self.position_sub = self.create_subscription(Odometry, "/amcl_pose", self.position_callback, 10)

        self._action_name = "/navigate_to_pose"
        self._client = ActionClient(self, NavigateToPose, self._action_name)

        self.load_dir = None
        self.target_name = None
        self.obj_map = None
        self.latest_costmap = None
        self.recovery_cnt = 0

        self.load_results()
        self.init_clip()


        self.is_room_ready = False
        self.is_related_position = False

        # 输入格式：min_x, max_x, min_y, max_y
        self.room_edges = {
            "bedroom": [-6.45, -1.65, 0.8, 4.4],
            "studyroom": [-6.6, -1.8, -3.1, -0.7]
        }

        # 手动给定目标，测试本地clip部分
        self.test_clip_offline("bed Room", "chair")
        print("test end!")

    def costmap_callback(self, msg: OccupancyGrid):
        self.latest_costmap = msg

    def position_callback(self, msg: Odometry):
        self.latest_position = msg.data


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
            "/home/tang123/DualMap/output/map_results/20260107_212437/global_map"
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
        
        # TODO: 此处发布的是什么，要发布指向的yaw还是要发布位置呢？

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

        # self.send_goal(target_x, target_y, target_yaw, frame_id, wait_timeout)

        print("========== TARGET OBJ GOAL SEND END ===========")


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

        # self.send_goal(related_x, related_y, related_yaw, frame_id, wait_timeout)

        print("======== RELATED OBJ GOAL SEND END ============")

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
            
            obj_min_x = -self.obj_map[idx].bbox_2d.min_bound[0]
            obj_min_y = -self.obj_map[idx].bbox_2d.min_bound[1]

            obj_max_x = -self.obj_map[idx].bbox_2d.max_bound[0]
            obj_max_y = -self.obj_map[idx].bbox_2d.max_bound[1]

            # TODO: 确定世界坐标系的转换
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

        ## Get top k candidates
        # top_k = len(cos_sim)
        # top_k_cos_sim, top_k_idx = torch.topk(cos_sim, top_k, dim=0)
        # print("Most similar object:")
        # for i, (cos_val, idx) in enumerate(zip(top_k_cos_sim.tolist(), top_k_idx.tolist())):

        #     bbox_2d = self.obj_map[idx].bbox_2d
        #     min_x = bbox_2d.min_bound[0]
        #     min_y = bbox_2d.min_bound[1]

        #     max_x = bbox_2d.max_bound[0]
        #     max_y = bbox_2d.max_bound[1]

        #     left_down_map = transfromPos(np.array([min_x, min_y]))
        #     right_down_map = transfromPos(np.array([max_x, min_y]))
        #     left_up_map = transfromPos(np.array([max_x, max_y]))
        #     right_up_map = transfromPos(np.array([min_x, max_y]))

        #     print(f"{i+1}. No. {idx} : {cos_val:.3f}")
        #     print(
        #         f"{self.obj_map[idx].class_name}, position {self.obj_map[idx].bbox_2d}, path {self. obj_map[idx].save_path}"
        #     )

        #     print("corners")
        #     print(left_down_map)
        #     print(right_down_map)
        #     print(left_up_map)
        #     print(right_up_map)
        #     corner_list = [left_down_map, right_down_map, left_up_map, right_up_map]

        #     return corner_list
        
    def get_room_edge(self, room_class: str):
        room_edge = self.room_edges[room_class]
        points = []

        for edge in room_edge:
            x, y = edge
            map_x = int((x - self.latest_costmap.info.origin.position.x) / self.latest_costmap.info.resolution)
            map_y = int((y - self.latest_costmap.info.origin.position.y) / self.latest_costmap.info.resolution)
            points.append((map_x, map_y))
        
        # 获取房间区域内的所有点
        min_x = min(p[0] for p in points)
        max_x = max(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_y = max(p[1] for p in points)

        return [min_x, max_x, min_y, max_y]
    
    def get_free_space(self, edge_points):
        min_x, max_x, min_y, max_y = edge_points

        # 计算房间中心点
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # 提取房间区域内的 costmap 数据
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                idx = y * self.latest_costmap.info.width + x
                if idx < len(self.latest_costmap.data) and self.latest_costmap.data[idx] == 0:
                    # 将网格坐标转换回世界坐标
                    world_x = x * self.latest_costmap.info.resolution + self.latest_costmap.info.origin.position.x
                    world_y = y * self.latest_costmap.info.resolution + self.latest_costmap.info.origin.position.y

                    # 计算到中心点的距离
                    dist = (x - center_x) ** 2 + (y - center_y) ** 2
                    if dist < min_dist:
                        min_dist = dist
                        best_point = (world_x, world_y)
        
        return best_point
 
        
    def recovery_clip(self, room_class: str):
        self.recovery_cnt += 1
        if self.recovery_cnt > 3:
            return None
        
        edge_points = self.get_room_edge(room_class)
        
        self.position_x = self.latest_pose.pose.position.x
        self.position_y = self.latest_pose.pose.position.y

        min_x, max_x, min_y, max_y = edge_points

        if min_x <= self.position_x <= max_x and min_y <= self.position_y <= max_y:
            # 如果已经在房间内了
            anchor_point = [self.position_x, self.position_y]
        else:
            # 如果不在房间内，则寻找房间中心的一个点
            anchor_point = self.get_free_space(edge_points)

        print(f"==> recovery room name {room_class}")
        print(f"==> recovery object {anchor_point}")

        # TODO: 后续改为判定转圈结束
        is_remap = True
        if is_remap:
            self.load_results()

            self.get_recovery_target()

            if self.related_object_name != "None":
                self.get_recovery_related_obj()
            
        # TODO: 后续打包这两个物体结果给nav模块
        
        

    def get_recovery_target(self):

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


    def get_recovery_related_obj(self):

        related_corner_list = self.query_callback(self.related_object_name)
        print(f"[query] related object: {self.related_object_name}")
        related_position = np.array(related_corner_list).mean(axis=0)




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

    cfg_path = "/home/tang123/DualMap/config/query_config.yaml"
    try:
        main(cfg_path)
    finally:
        rclpy.shutdown()
