import copy
import json
import os
os.environ['DISPLAY'] = ''
import sys
from pathlib import Path

import hydra
import yaml
import numpy as np
import open3d as o3d
import open_clip
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
sys.path.append("/home/tang123/DualMap")

from utils.object import BaseObject
from mobileclip.modules.common.mobileone import reparameterize_model
from task_extract import TaskExtractor


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
    yaml_path = "/home/tang123/Documents/vln_gazebo_simulator/src/TB3-Gazebo-Nav2-explore-lite/config/nav2/keepout_bboxes.yaml"

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



    


@hydra.main(version_base=None, config_path="../config/", config_name="query_config")
def main(cfg: DictConfig):

    ### Loading Color map: class id --> color Dict
    if cfg.yolo.use_given_classes:
        given_classes_path = cfg.yolo.given_classes_path
        # dir_path = os.path.dirname(given_classes_path)  # './model'
        dir_path = Path(given_classes_path).parent
        base_name = os.path.basename(given_classes_path)  # 'gpt_indoor_table.txt'
        file_root, _ = os.path.splitext(base_name)  # 'gpt_indoor_table'

        class_id_colors_path = os.path.join(dir_path, file_root + "_id_colors.json")
        print(class_id_colors_path)
    else:
        class_id_colors_path = os.path.join(
            cfg.output_path,
            f"{cfg.dataset_name}_{cfg.scene_id}",
            "classes_info",
            f"{cfg.dataset_name}_{cfg.scene_id}_id_colors.json",
        )

    print("Loading classes id --> colors from: {}".format(class_id_colors_path))

    if not os.path.exists(class_id_colors_path):
        raise FileNotFoundError(f"Error: File not found: {class_id_colors_path}")

    class_id_colors = {}
    with open(class_id_colors_path, "r") as file:
        class_id_colors = json.load(file)
    class_id_colors = {int(key): value for key, value in class_id_colors.items()}

    # Dict: class id --> name
    class_id_names = {}

    if cfg.yolo.use_given_classes:
        class_id_names_path = cfg.yolo.given_classes_path
        # Load class names from txt
        with open(class_id_names_path, "r") as f:
            class_list = [line.strip() for line in f if line.strip()]
        class_id_names = {i: name for i, name in enumerate(class_list)}
    else:
        class_id_names_path = os.path.join(
            cfg.output_path,
            f"{cfg.dataset_name}_{cfg.scene_id}",
            "classes_info",
            f"{cfg.dataset_name}_{cfg.scene_id}_id_names.json",
        )

        with open(class_id_names_path, "r") as file:
            class_id_names = json.load(file)
        class_id_names = {int(key): value for key, value in class_id_names.items()}

    print("Loading classes id --> names  from: {}".format(class_id_names_path))

    if not os.path.exists(class_id_names_path):
        raise FileNotFoundError(f"Error: File not found: {class_id_names_path}")

    ### Loading saved results
    # if map_dir is not provided, use the default path
    
    # FLAG: 移植实机需处理目录 
    load_dir = None
    if os.path.exists(cfg.test_map_dir):
        load_dir = cfg.test_map_dir
    else:
        load_dir = os.path.join(
            cfg.output_path, f"{cfg.dataset_name}_{cfg.scene_id}", "map"
        )

    if not os.path.exists(load_dir):
        print(f"Error: {load_dir} does not exist.")
        sys.exit(1)

    print(("Loading saved obj results from: {}".format(load_dir)))

    ### Loading viewpoint
    viewpoint_path = os.path.join(load_dir, "viewpoint.json")
    print(f"Loading viewpoint from: {viewpoint_path}")

    # traverse the .pkl in the directory to get constructed maps
    obj_map = []
    for file in os.listdir(load_dir):
        if file.endswith(".pkl"):
            obj_results_path = os.path.join(load_dir, file)
            # object construction
            loaded_obj = BaseObject.load_from_disk(obj_results_path)
            obj_map.append(loaded_obj)
    print(f"Successfully loaded {len(obj_map)} objects")

    ### Init of CLIP
    print("Loading CLIP model")
    # clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    #     "ViT-H-14", "laion2b_s32b_b79k"
    # )
    # clip_model = clip_model.to("cuda")
    # clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        cfg.clip.model_name, pretrained=cfg.clip.pretrained
    )
    clip_model = clip_model.to(cfg.device)
    clip_model.eval()

    # Only reparameterize if the model is MobileCLIP
    if "MobileCLIP" in cfg.clip.model_name:
        print("==> Opening mobileclip")
        clip_model = reparameterize_model(clip_model)

    clip_tokenizer = open_clip.get_tokenizer(cfg.clip.model_name)

    print("Done initializing CLIP model.")

    # cmap = matplotlib.colormaps.get_cmap("turbo")

    print(f"Obj Map length: %d" % len(obj_map))

    def query_callback(instance_query):

        text_queries = [instance_query]

        text_queries_tokenized = clip_tokenizer(text_queries).to("cuda")
        text_query_ft = clip_model.encode_text(text_queries_tokenized)
        text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)
        text_query_ft = text_query_ft.squeeze()

        ## Get stacked clip feats from the map
        values = []
        for obj in obj_map:
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

            bbox_2d = obj_map[idx].bbox_2d
            min_x = bbox_2d.min_bound[0]
            min_y = bbox_2d.min_bound[1]

            max_x = bbox_2d.max_bound[0]
            max_y = bbox_2d.max_bound[1]

            left_down_map = transfromPos(np.array([min_x, min_y]))
            right_down_map = transfromPos(np.array([max_x, min_y]))
            left_up_map = transfromPos(np.array([max_x, max_y]))
            right_up_map = transfromPos(np.array([min_x, max_y]))

            print(
                f"{i+1}. No. {idx} {class_id_names[obj_map[idx].class_id]}: {cos_val:.3f}"
            )
            print(
                f"{obj_map[idx].class_name}, position {obj_map[idx].bbox_2d}, path {obj_map[idx].save_path}"
            )

            print("corners")
            print(left_down_map)
            print(right_down_map)
            print(left_up_map)
            print(right_up_map)
            corner_list = [left_down_map, right_down_map, left_up_map, right_up_map]

            return corner_list
            

    def run():

        extract_res = getInstances()

        target_room = extract_res["target_room"]
        target_object = extract_res["target_object"]
        avoid_object = extract_res["avoid_object"]

        print(f"[offline] target room {target_room}, target object {target_object}\
                avoid_object {avoid_object}")
        
        print("----------------------------------")
        
        print("==> target object")
        target_corner_list = query_callback(target_object + "in" + target_room)
        print("==============================")
        if avoid_object != "None":
            print("==> avoid object")
            avoid_corner_list = query_callback(avoid_object)
            pop_hazard2yaml(avoid_corner_list)
            print("==============================")
            

        else:
            print("Nothing to avoid!")


    run()


if __name__ == "__main__":
    main()
