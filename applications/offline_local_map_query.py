'''
Docstring for applications.offline_local_map_query
dualmap 主机端执行，订阅目标物体名称，基于离线构建的local map进行目标位置查询
发布目标位置，避障物包围盒
'''

import os
os.environ['DISPLAY'] = ''
import sys

import yaml
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
sys.path.append("/home/tang123/DualMap")

from utils.object import BaseObject
from mobileclip.modules.common.mobileone import reparameterize_model
from task_extract import TaskExtractor

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

class TargetPositionPublisher(Node):
    def __init__(self):
        super().__init__('target_position_publisher')
        self.publisher_ = self.create_publisher(Float64MultiArray, 'target_position', 10)
        
    def publish_position(self, position):
        msg = Float64MultiArray()
        msg.data = position.tolist()
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published target position: {msg.data}')


class TaskSubscriber(Node):
    def __init__(self, cfg_path: str):
        super().__init__('task_subscriber')
        with open(cfg_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        self.subscription = self.create_subscription(
            String,
            'target_name',
            self.get_target_position,
            10)
        
        self.hazard_subscription = self.create_subscription(
            String,
            'semantic_hazard',
            self.get_hazard_position,
            10)
        
        self.position_pub = TargetPositionPublisher()
        
        self.load_dir = None
        self.target_name = None
        self.obj_map = None

        self.load_results()
        self.init_clip()
        
        
    def load_results(self):
        ### Loading saved results
        # if map_dir is not provided, use the default path
        # FLAG: 移植实机需处理目录 
        load_dir = None
        if os.path.exists(self.cfg.test_map_dir):
            load_dir = self.cfg.test_map_dir
        else:
            load_dir = os.path.join(
                self.cfg.output_path, f"{self.cfg.dataset_name}_{self.cfg.scene_id}", "map"
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

        self.clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            self.cfg.clip.model_name, pretrained=self.cfg.clip.pretrained
        )
        self.clip_model = self.clip_model.to(self.cfg.device)
        self.clip_model.eval()
        # Only reparameterize if the model is MobileCLIP
        if "MobileCLIP" in self.cfg.clip.model_name:
            print("==> Opening mobileclip")
            self.clip_model = reparameterize_model(self.clip_model)

        self.clip_tokenizer = open_clip.get_tokenizer(self.cfg.clip.model_name)
        print("Done initializing CLIP model.")

        print(f"Obj Map length: %d" % len(obj_map))
        
        

    def get_target_position(self, msg):
        self.target_name = msg.data
        print(f"Received target name: {self.target_name}")
        
        print("==> target object")
        
        corner_list = self.query_callback(self.target_name)
        target_position = np.array(corner_list).mean(axis=0)
        print(f"[query] target position: {target_position}")
        self.position_pub.publish_position(target_position)
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
        cos_sim = F.cosine_similarity(text_query_ft.unsqueeze(0), map_clip_fts, dim=-1)

        ## Get top k candidates
        top_k = 1
        top_k_cos_sim, top_k_idx = torch.topk(cos_sim, top_k, dim=0)
        print("Top 5 similar objects:")
        for i, (cos_val, idx) in enumerate(
            zip(top_k_cos_sim.tolist(), top_k_idx.tolist())
        ):

            bbox_2d = self.obj_map[idx].bbox_2d
            min_x = bbox_2d.min_bound[0]
            min_y = bbox_2d.min_bound[1]

            max_x = bbox_2d.max_bound[0]
            max_y = bbox_2d.max_bound[1]

            left_down_map = transfromPos(np.array([min_x, min_y]))
            right_down_map = transfromPos(np.array([max_x, min_y]))
            left_up_map = transfromPos(np.array([max_x, max_y]))
            right_up_map = transfromPos(np.array([min_x, max_y]))

            print(
                f"{i+1}. No. {idx} : {cos_val:.3f}"
            )
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
    '''
    输入为dualmap世界坐标系读出的坐标，返回gazebo坐标系下的坐标
    '''
    # return np.array([position[1], -position[0], -position[2]])
    return np.array([position[1], -position[0]])

def pop_hazard2yaml(corners: list):
    '''
    pump semantic hazard to yaml, pass to navigation modual
    '''
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
        left_down[0], left_down[1],
        right_down[0], right_down[1],
        left_up[0], left_up[1],
        right_up[0], right_up[1]
        )

    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"[query] Pumped semantic hazard to yaml: {yaml_path}")


def getInstances():

    text_query = input("Enter your query (type 'bye' to exit): ").strip()
    
    if text_query == "bye" or text_query == "quit":
        print("Exiting query loop...")
        return
    
    ai_scientist = TaskExtractor(text_query)
    extract_res = ai_scientist.extract_navigation_components()

    return extract_res



def main(cfg_path: str):

    rclpy.init()
    
    position_publisher = TargetPositionPublisher(cfg_path)
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
