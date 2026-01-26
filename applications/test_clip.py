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

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))        # applications/
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)                    # DualMap/
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from utils.object import BaseObject


class TestClip():
    def __init__(self, load_dir: str):
        self.load_dir = load_dir

        self.load_results()
        self.init_clip()


    def load_results(self):
        load_dir = self.load_dir
        if not os.path.exists(load_dir):
            print(f"{load_dir} does not exist.")
            sys.exit(1)

        print(f"Loading saved obj results from: {load_dir}")

        obj_map = []
        pkl_files = sorted([f for f in os.listdir(self.load_dir) if f.endswith(".pkl")])
        for file in pkl_files:
            obj_results_path = os.path.join(self.load_dir, file)
            loaded_obj = BaseObject.load_from_disk(obj_results_path)
            obj_map.append(loaded_obj)

        print(f"Successfully loaded {len(obj_map)} objects")
        self.obj_map = obj_map

    def init_clip(self):
        print("Loading CLIP model")
        clip_model_name = "ViT-B-32"
        pretrained_path = "/home/tang123/ViT-B-32.pt"  # 修改为实际路径
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=pretrained_path, device="cpu"
        )
        device = "cpu"
        self.clip_model = self.clip_model.to(device)
        self.clip_model.eval()
        self.clip_tokenizer = open_clip.get_tokenizer(clip_model_name)
        print(f"Using device: {device}, Done initializing CLIP model.")


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

        for cos_val, idx in zip(sorted_cos_sim.tolist(), sorted_idx.tolist()):
            obj_min_x = self.obj_map[idx].bbox_2d.min_bound[0]
            obj_min_y = self.obj_map[idx].bbox_2d.min_bound[1]
            obj_max_x = self.obj_map[idx].bbox_2d.max_bound[0]
            obj_max_y = self.obj_map[idx].bbox_2d.max_bound[1]

            left_down_map = np.array([obj_min_x, obj_min_y])
            right_down_map = np.array([obj_max_x, obj_min_y])
            left_up_map = np.array([obj_max_x, obj_max_y])
            right_up_map = np.array([obj_min_x, obj_max_y])

            corner_list = [left_down_map, right_down_map, left_up_map, right_up_map]
            print(
                f"[query] '{instance_query}' hit idx={idx} sim={cos_val:.3f} name={self.obj_map[idx].class_name}"
            )
            return corner_list

        print(f"[query] '{instance_query}' found nothing after room filter.")
        return None
    

tester = TestClip(load_dir="/home/tang123/DualMap/output/map_on_oe/20260120_054551/global_map")
result = tester.query_callback("笔记本")