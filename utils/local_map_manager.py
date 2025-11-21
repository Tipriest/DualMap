import logging
import os
import pdb
import shutil
from collections import Counter
from typing import List

import networkx as nx
import numpy as np
from omegaconf import DictConfig

from utils.base_map_manager import BaseMapManager
from utils.navigation_helper import NavigationGraph
from utils.object import LocalObject, LocalObjStatus
from utils.types import GlobalObservation, GoalMode, Observation

# Set up the module-level logger
logger = logging.getLogger(__name__)


class LocalMapManager(BaseMapManager):
    """
    管理局部地图，包括对象跟踪、状态更新和关系图。
    """

    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        # 构造函数
        super().__init__(cfg)

        # idx
        self.curr_idx = 0

        # global observations, all are low mobility objects(全局观测，所有都是低移动性对象)
        self.global_observations = []

        # objects list(对象列表)
        self.local_map = []
        self.global_map = []

        self.graph = (
            nx.Graph()
        )  # Use undirected graph to manage object relationships(使用无向图管理对象关系)
        self.current_relations = set()

        # objects need to be eliminated in local map and graph(需要在局部地图和图中淘汰的对象)
        self.to_be_eliminated = set()

        # Local Object Config Init(局部对象配置初始化)
        LocalObject.initialize_config(cfg)

        # For navigation
        self.nav_graph = None
        self.inquiry = ""
        # If in Click mode, the goal by clicked(如果在点击模式下，目标由点击确定)
        self.click_goal = None
        # If in Inquiry mode, the goal by bbox(如果在查询模式下，目标由包围盒确定)
        self.global_bbox = None
        self.global_score = 0.0

    def set_click_goal(self, goal):
        """设置点击模式下的导航目标。"""
        self.click_goal = goal

    def set_global_bbox(self, bbox):
        """设置全局包围盒。"""
        self.global_bbox = bbox

    def set_global_score(self, score):
        """设置全局得分。"""
        self.global_score = score

    def set_global_map(self, global_map):
        """设置全局地图。"""
        self.global_map = global_map

    def has_local_map(self) -> bool:
        return len(self.local_map) > 0

    def set_curr_idx(
        self,
        idx: int,
    ) -> None:
        """设置当前帧索引。"""
        self.curr_idx = idx
        LocalObject.set_curr_idx(idx)

    def get_global_observations(self) -> list:
        """获取全局观测的副本。"""
        return self.global_observations.copy()

    def clear_global_observations(
        self,
    ) -> None:
        """清空全局观测。"""
        self.global_observations.clear()

    # TODO: Using list, finding object by UID requires traversal, which is inefficient
    # (使用列表，通过UID查找对象需要遍历，效率低下)
    def get_object(self, uid) -> LocalObject:
        """Helper function to get an object by UID from the list."""
        """辅助函数, 通过UID从列表中获取对象。"""
        for obj in self.local_map:
            if obj.uid == uid:
                return obj

    def set_relation(self, obj1_uid, obj2_uid):
        """Set a relation between two objects based on their UIDs."""
        # Check if both objects exist
        if self.get_object(obj1_uid) and self.get_object(obj2_uid):
            self.graph.add_edge(obj1_uid, obj2_uid)

    def has_relation(self, obj1_uid, obj2_uid) -> bool:
        """根据UID检查两个对象是否有关系。"""
        return self.graph.has_edge(obj1_uid, obj2_uid)

    def remove_relation(self, obj1_uid, obj2_uid):
        """根据UID移除两个对象之间的关系。"""
        if self.graph.has_edge(obj1_uid, obj2_uid):
            self.graph.remove_edge(obj1_uid, obj2_uid)

    def get_related_objects(self, obj_uid) -> list:
        """返回与给定对象UID相关的对象UID列表。"""
        if obj_uid in self.graph:
            related_uids = list(self.graph.neighbors(obj_uid))
            # Convert these UIDs to LocalObject objects and return(将这些UID转换为LocalObject对象并返回)
            related_objects = [
                self.get_object(uid) for uid in related_uids if self.get_object(uid)
            ]
            return related_objects
        else:
            return (
                []
            )  # If the UID is not in the graph, return an empty list(如果UID不在图中，返回空列表)

    # Main entry point
    # Update local map using observations
    # 主入口点
    # 使用观测更新局部地图
    def process_observations(
        self,
        curr_observations: List[Observation],
    ) -> None:
        """处理当前观测并更新局部地图。"""

        # if first, then just insert(如果是第一次，则直接插入)
        if self.is_initialized == False:
            # Init the local map
            logger.info("[LocalMap] Init Local Map by first observation")

            if len(curr_observations) == 0:
                logger.warning("[LocalMap] No observation in this frame")
                return

            self.init_from_observation(curr_observations)
            self.is_initialized = True
            return

        if len(curr_observations) == 0:
            logger.warning("[LocalMap] No observation in this frame")
            self.update_local_map(curr_observations)
            return

        # 如果不是第一次，则进行匹配(设置跟踪器参考)
        logger.info("[LocalMap] 匹配中")
        self.tracker.set_current_frame(curr_observations)

        # Set tracker reference
        self.tracker.set_ref_map(self.local_map)
        self.tracker.matching_map()

        # After matching map, current frame information will be updated
        # 匹配地图后，当前帧信息将被更新
        curr_observations = self.tracker.get_current_frame()

        # Update local map(更新局部地图)
        self.update_local_map(curr_observations)

    def init_from_observation(
        self,
        observations: List[Observation],
    ) -> list:
        """从初始观测初始化局部地图。"""

        for obs in observations:
            # Init
            local_obj = LocalObject()
            local_obj.add_observation(obs)
            local_obj.update_info()

            self.local_map.append(local_obj)
            self.graph.add_node(local_obj.uid)

    def update_local_map(self, curr_obs: List[Observation]) -> None:
        """使用最新的观测更新局部地图。"""
        # 更新局部地图与观测
        for obs in curr_obs:
            if obs.matched_obj_idx == -1:
                # Add new local object(添加新的局部对象)
                local_obj = LocalObject()
                local_obj.add_observation(obs)
                local_obj.update_info()

                self.local_map.append(local_obj)
                self.graph.add_node(local_obj.uid)
                # logger.info("[LMM] Add new local object!, current local map objs num: ", len(self.local_map))
            else:
                # Update existing local object(更新现有的局部对象)
                matched_obj = self.local_map[obs.matched_obj_idx]
                matched_obj.add_observation(obs)
                matched_obj.update_info()

        # traverse through the local map (遍历局部地图)
        # split the local obj with the split marker (根据分裂标记分裂局部对象)
        # Solve couch + pillow problem (解决沙发+枕头问题)

        for obj in self.local_map:
            if obj.should_split:
                self.split_local_object(obj)

        # update the graph and map for insertion and elimination
        # 更新地图和图以进行插入和淘汰
        self.update_map_and_graph()

        # traverse through the local map (遍历局部地图)
        # 1. check stability and update status (1. 检查稳定性并更新状态)
        # 2. do actions based on objects status (2. 根据对象状态执行操作)
        for obj in self.local_map:
            # Update the status of the current local object (更新当前局部对象的状态)
            obj.update_status()

            # do actions based on objects status (根据对象状态执行操作)
            self.status_actions(obj)

        # update the graph and map for insertion and elimination
        # 更新地图和图以进行插入和淘汰
        self.update_map_and_graph()

        logger.info(
            "[LocalMap] 当前我们有global observations的数量: "
            + str(len(self.global_observations))
        )

        if self.cfg.use_rerun:
            self.visualize_local_map()

    def end_process(
        self,
    ) -> None:
        """在处理结束时执行最终操作。"""

        for obj in self.local_map:
            # Update the status of the current local object(更新当前局部对象的状态)
            obj.update_status()

            # do actions based on objects status(根据对象状态执行操作)
            self.status_actions(obj)

        # update the graph and map for insertion and elimination
        # 更新地图和图以进行插入和淘汰
        self.update_map_and_graph()

        if self.cfg.use_rerun:
            self.visualize_local_map()

    def update_map_and_graph(
        self,
    ) -> None:
        # Manage the map and graph
        # 1. Update the graph with current relations
        # 2. Insertion and Elimination of local objects in local_map
        # 3. Insertion and Elimination of nodes in graph
        """
        管理地图和图。
        1. 使用当前关系更新图。
        2. 在local_map中插入和淘汰局部对象。
        3. 在图中插入和淘汰节点。
        """

        # 1. Update the graph with current relations
        if self.current_relations:
            for obj1_uid, obj2_uid in list(self.graph.edges):
                if (obj1_uid, obj2_uid) not in self.current_relations and (
                    obj2_uid,
                    obj1_uid,
                ) not in self.current_relations:
                    self.remove_relation(obj1_uid, obj2_uid)

            # clear the current relations
            self.current_relations.clear()

        # 2. Elimination of local objects in local_map and graph
        # (在local_map和图中淘汰局部对象)
        if self.to_be_eliminated:
            # local map deletion(局部地图删除)
            self.local_map = [
                obj for obj in self.local_map if obj.uid not in self.to_be_eliminated
            ]

            # graph deletion(图删除)
            for uid in self.to_be_eliminated:
                if self.graph.has_node(uid):
                    self.graph.remove_node(uid)

            # clear the to_be_eliminated(清空待淘汰列表)
            self.to_be_eliminated.clear()

    def status_actions(self, obj: LocalObject) -> None:
        """根据对象状态执行操作。"""

        # Set Relations
        # if the object is stable, then no matter in what status,
        # it will search and check all the objs with major plane info in the map
        # to check if it has the on relation
        # 设置关系
        # 如果对象是稳定的，那么无论处于何种状态，
        # 它都会搜索并检查地图中所有具有主平面信息的对象，
        # 以检查是否存在"on"关系。

        if obj.is_stable == True:
            # logger.info uid for debug
            # logger.info(f"Checking relations for {obj.uid}")

            # traverse all the other objects and check the relation with the current object
            # 遍历所有其他对象并检查与当前对象的关系
            for other_obj in self.local_map:
                # If the other obj meets the on relation with the current object
                # and not the obj itself
                # 如果其他对象与当前对象满足"on"关系并且不是对象本身
                if other_obj.uid != obj.uid and self.on_relation_check(obj, other_obj):
                    # set the relation in the graph(在图中设置关系)
                    self.set_relation(obj.uid, other_obj.uid)
                    # save the current valid relations(保存当前有效关系)
                    self.current_relations.add(
                        (min(obj.uid, other_obj.uid), max(obj.uid, other_obj.uid))
                    )
                    # logger.info the relation for debug
                    # logger.info(f"Relation: {obj.uid} - {other_obj.uid}")

        # ELIMINATION Actions(淘汰操作)
        if obj.status == LocalObjStatus.ELIMINATION:
            self.to_be_eliminated.add(obj.uid)
            return

        # HM_ELIMINATION Actions(高移动性淘汰操作)
        if obj.status == LocalObjStatus.HM_ELIMINATION:

            # if we run local mapping only, we keep the HM/LM_ELIMINATION objects
            # 如果我们只运行局部建图，我们保留HM/LM_ELIMINATION对象
            if self.cfg.run_local_mapping_only:
                return

            # Get all the related objects in the graph(获取图中所有相关对象)
            related_objs = self.get_related_objects(obj.uid)

            # if no related objects, delete the current object and return
            # 如果没有相关对象，删除当前对象并返回
            if len(related_objs) == 0:
                self.to_be_eliminated.add(obj.uid)
                return

            # else return, waiting for the LM actions
            # 否则返回，等待LM操作
            return

        # LM_ELIMINATION Actions
        # 低移动性淘汰操作
        if obj.status == LocalObjStatus.LM_ELIMINATION:

            # if we run local mapping only, we keep the HM/LM_ELIMINATION objects
            # 如果我们只运行局部建图，我们保留HM/LM_ELIMINATION对象
            if self.cfg.run_local_mapping_only:
                return

            # Get all the related objects in the graph
            # 获取图中所有相关对象
            related_objs = self.get_related_objects(obj.uid)

            # if no related objects, delete in local map and ready for global obs
            # 如果没有相关对象，在局部地图中删除并准备全局观测
            if len(related_objs) == 0:

                class_name = self.visualizer.obj_classes.get_classes_arr()[obj.class_id]

                # restrict unknown (限制未知标签)
                if self.cfg.restrict_unknown_labels and class_name == "unknown":
                    self.to_be_eliminated.add(obj.uid)
                    return

                # generate global observation and insert to global obs list
                # 生成全局观测并插入到全局观测列表
                global_obs = self.create_global_observation(obj)
                self.global_observations.append(global_obs)

                self.to_be_eliminated.add(obj.uid)

                return

            # If it has related objects
            # Check all the related objects status
            # 如果有相关对象, 检查所有相关对象的状态
            is_related_obj_ready = True
            for related_obj in related_objs:
                # if all the related object is not ready, than the LM will wait
                # 如果所有相关对象都未准备好，则LM将等待
                if (
                    related_obj.status == LocalObjStatus.UPDATING
                    or related_obj.status == LocalObjStatus.PENDING
                ):
                    is_related_obj_ready = False
                    break

            # If all the related object is ready, then generate global observation
            # And delete the current object and all related objects
            # 如果所有相关对象都准备好了，则生成全局观测，并删除当前对象和所有相关对象

            if is_related_obj_ready:

                class_name = self.visualizer.obj_classes.get_classes_arr()[obj.class_id]

                # restrict unknown(限制未知标签)
                if self.cfg.restrict_unknown_labels and class_name == "unknown":
                    self.to_be_eliminated.add(obj.uid)
                    for related_obj in related_objs:
                        self.to_be_eliminated.add(related_obj.uid)
                    return

                # generate global observation and insert to global obs list
                # 生成全局观测并插入到全局观测列表
                global_obs = self.create_global_observation(
                    obj, related_objs=related_objs
                )
                self.global_observations.append(global_obs)

                self.to_be_eliminated.add(obj.uid)

                # Delete all the related objs(删除所有相关对象)
                for related_obj in related_objs:
                    self.to_be_eliminated.add(related_obj.uid)

            return

    def on_relation_check(self, base_obj: LocalObject, test_obj: LocalObject) -> bool:
        # test whether the test_obj is related to the base_obj (with "on" relation)
        # return True if related, False otherwise
        """
        测试 test_obj 是否与 base_obj 相关（具有 "on" 关系）。
        如果相关则返回 True, 否则返回 False。
        """

        # If no bbox, return False(如果没有包围盒，返回 False)
        if base_obj.bbox is None or test_obj.bbox is None:
            return False

        # If both objs have no major plane info, return False
        # 如果两个对象都没有主平面信息，返回 False
        if base_obj.major_plane_info is None and test_obj.major_plane_info is None:
            return False

        # if both objs have major plane info, also return False
        # 如果两个对象都有主平面信息，也返回 False
        if (
            base_obj.major_plane_info is not None
            and test_obj.major_plane_info is not None
        ):
            return False

        base_center = base_obj.bbox.get_center()
        test_center = test_obj.bbox.get_center()
        if np.all(base_center == 0) or np.all(test_center == 0):
            return False

        # Here we have one obj has major plane info and the other has no major plane info
        # check which one has major plane info, and set that obj as base_obj
        # 这里我们有一个对象有主平面信息，另一个没有
        # 检查哪个有主平面信息，并将该对象设为 base_obj
        if base_obj.major_plane_info is None:
            # swap base_obj and test_obj
            base_obj, test_obj = test_obj, base_obj

        base_aabb = base_obj.bbox
        test_aabb = test_obj.bbox

        # Get base_obj and test_obj AABB bounds
        base_min_bound = base_aabb.get_min_bound()  # return [x_min, y_min, z_min]
        base_max_bound = base_aabb.get_max_bound()  # return [x_max, y_max, z_max]

        test_min_bound = test_aabb.get_min_bound()  # return [x_min, y_min, z_min]
        test_max_bound = test_aabb.get_max_bound()  # return [x_max, y_max, z_max]

        # Get AABB xy range
        base_x_min, base_y_min = base_min_bound[0], base_min_bound[1]
        base_x_max, base_y_max = base_max_bound[0], base_max_bound[1]

        test_x_min, test_y_min = test_min_bound[0], test_min_bound[1]
        test_x_max, test_y_max = test_max_bound[0], test_max_bound[1]

        # calculate AABB overlap area in xy plane(计算 xy 平面上的 AABB 重叠区域)
        overlap_x = max(0, min(base_x_max, test_x_max) - max(base_x_min, test_x_min))
        overlap_y = max(0, min(base_y_max, test_y_max) - max(base_y_min, test_y_min))

        # Calculate test_obj AABB area size in xy plane
        # 计算 test_obj AABB 在 xy 平面上的面积大小
        test_area = (test_x_max - test_x_min) * (test_y_max - test_y_min)

        # get overlap area size(获取重叠区域大小)
        overlap_area_size = overlap_x * overlap_y

        # calculate the ratio of overlap area to test_obj AABB area size
        # ensure the ratio is a float
        # 计算重叠区域与 test_obj AABB 面积大小的比率
        # 确保比率是浮点数
        overlap_ratio = overlap_area_size / test_area

        # if the ratio is lower than the threshold, return False
        # 如果比率低于阈值，返回 False
        if overlap_ratio < self.cfg.object_matching.overlap_ratio:
            return False

        # Check if the down z value of test_obj is near the base_obj's major plane
        # 检查 test_obj 的底部 z 值是否接近 base_obj 的主平面
        plane_distance = self.cfg.on_relation.plane_distance
        if not (
            test_min_bound[2] - plane_distance <= base_obj.major_plane_info
            and base_obj.major_plane_info <= test_min_bound[2] + (plane_distance * 2)
        ):
            return False

        return True

    def create_global_observation(
        self, obj: LocalObject, related_objs: List[LocalObject] = []
    ) -> Observation:
        # generate observations for global mapping
        """为全局建图生成观测。"""

        curr_obs = GlobalObservation()

        # 设置全局观测信息
        curr_obs.uid = obj.uid
        curr_obs.class_id = obj.class_id
        curr_obs.pcd = obj.pcd
        curr_obs.bbox = obj.pcd.get_axis_aligned_bounding_box()
        curr_obs.clip_ft = obj.clip_ft
        # Use configurable voxel size for downsampling
        # 使用可配置的体素大小进行下采样
        pcd_2d = obj.voxel_downsample_2d(obj.pcd, self.cfg.downsample_voxel_size)
        curr_obs.pcd_2d = pcd_2d
        curr_obs.bbox_2d = pcd_2d.get_axis_aligned_bounding_box()

        if related_objs:
            for related_obj in related_objs:
                curr_obs.related_objs.append(related_obj.clip_ft)
                # for visualization in rerun(用于在rerun中可视化)
                curr_obs.related_bbox.append(related_obj.bbox)
                curr_obs.related_color.append(related_obj.class_id)
        else:
            curr_obs.related_objs = []
            curr_obs.related_bbox = []
            curr_obs.related_color = []

        return curr_obs

    def split_local_object(
        self,
        obj: LocalObject,
    ) -> None:
        """split局部对象"""
        for class_id, deque in obj.split_info.items():
            new_obj = LocalObject()
            # traverse through the observation in the obj(遍历对象中的观测)
            for obs in list(obj.observations):
                if obs.class_id == class_id:
                    new_obj.add_observation(obs)
                    # delete the obs in observation
                    # TODO: ALso we actually no need to delete
                    # 从观测中删除obs
                    # TODO: 实际上我们不需要删除
                    obj.observations = [
                        ob
                        for ob in obj.observations
                        if ob.class_id != class_id or ob.idx != obs.idx
                    ]
            # new obj update(新对象更新)
            new_obj.update_info_from_observations()

            # add to local map(添加到局部地图)
            self.local_map.append(new_obj)
            # add to graph(添加到图)
            self.graph.add_node(new_obj.uid)

        split_info = obj.print_split_info()
        logger.info(
            "[LocalMap][Split] Split Local Object, splitted info: %s" % split_info
        )

        # find the obj in local map by using uid(通过uid在局部地图中找到对象)
        # remove the obj in the local map list(从局部地图列表中移除对象)
        self.to_be_eliminated.add(obj.uid)

    def save_map(self) -> None:
        # get the directory
        save_dir = self.cfg.map_save_path

        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
            logger.info(f"[LocalMap] Cleared the directory: {save_dir}")
        os.makedirs(save_dir)

        for i, obj in enumerate(self.local_map):
            if obj.save_path is not None:
                logger.info(f"[LocalMap] 正在保存第{i}个对象: {obj.save_path}")
                obj.save_to_disk()
            else:
                logger.warning("[LocalMap] 局部对象没有保存路径")
                continue

    def merge_local_map(self) -> None:
        """合并局部地图中的对象。"""

        # Use tracker for matching(使用跟踪器进行匹配)
        self.tracker.set_current_frame(self.local_map)

        # Set tracker reference(设置跟踪器参考)
        self.tracker.set_ref_map(self.local_map)
        self.tracker.matching_map(is_map_only=True)

        # After matching map, current map information will be updated
        # 匹配地图后，当前地图信息将被更新
        self.merge_info = self.tracker.get_merge_info()

        # Traverse the map, merge based on merge info
        # 遍历地图，根据合并信息进行合并
        new_local_map = []
        # Traverse merge info
        # 遍历合并信息
        for indices in self.merge_info:
            # If not merged, just add the object
            # 如果未合并，则直接添加对象
            if len(indices) == 1:
                new_local_map.append(self.local_map[indices[0]])
            else:
                new_local_map.append(
                    self.merge_local_object([self.local_map[i] for i in indices])
                )
        # New local map is generated
        # 生成新的局部地图

        logger.info(
            f"[LocalMap][Merge] Map size before merge: {len(self.local_map)}, after merge: {len(new_local_map)}"
        )

        self.local_map = new_local_map

        if self.cfg.use_rerun:
            self.visualize_local_map()

    def merge_local_object(
        self,
        obj_list: List[LocalObject],
    ) -> LocalObject:
        # This function merges a list of objects
        # Two approaches: 1) Update with all observations, 2) Use object info only
        # Using the first method for code simplicity
        """
        此函数合并对象列表。
        两种方法: 1) 使用所有观测进行更新, 2) 仅使用对象信息。
        为简化代码, 使用第一种方法。
        """
        new_obj = LocalObject()
        # Traverse objects in the given list(遍历给定列表中的对象)
        for obj in obj_list:
            for obs in obj.observations:
                new_obj.add_observation(obs)

        # Update the info of the new object(更新新对象的信息)
        new_obj.update_info_from_observations()
        new_obj.is_merged = True
        return new_obj

    def visualize_local_map(
        self,
    ) -> None:
        # Show all local objects in the local map
        """可视化局部地图。"""
        # 显示局部地图中的所有局部对象

        new_logged_entities = set()

        # Temp lists for 3d bbox overlapping drawing
        # 用于3D包围盒重叠绘制的临时列表
        obj_names = []
        obj_colors = []
        obj_bboxes = []

        for local_obj in self.local_map:
            # Newly added objects are not considered
            # 不考虑新添加的对象
            if local_obj.observed_num <= 2:
                continue

            obj_name = self.visualizer.obj_classes.get_classes_arr()[local_obj.class_id]

            # Ignore ceiling wall
            # 忽略天花板墙
            if (
                obj_name == "ceiling wall"
                or obj_name == "carpet"  # 地毯
                or obj_name == "rug"  # 毛皮地毯
                or obj_name == "ceiling_molding"# 天花板装饰条
            ):
                continue

            obj_label = f"{local_obj.observed_num}_{obj_name}"
            obj_label = obj_label.replace(" ", "_")

            base_entity_path = "world/objects"
            entity_path = f"world/objects/{obj_label}"

            positions = np.asarray(local_obj.pcd.points)
            colors = np.asarray(local_obj.pcd.colors) * 255
            colors = colors.astype(np.uint8)
            curr_obj_color = self.visualizer.obj_classes.get_class_color(obj_name)

            obj_names.append(obj_name)
            obj_colors.append(curr_obj_color)

            if self.cfg.show_local_entities:

                # Log pcd data
                rgb_pcd_entity = base_entity_path + "/rgb_pcd" + f"/{local_obj.uid}"
                self.visualizer.log(
                    rgb_pcd_entity,
                    # entity_path + "/pcd",
                    self.visualizer.Points3D(
                        positions,
                        colors=colors,
                        # radii=0.01
                        # labels=[obj_label],
                    ),
                    self.visualizer.AnyValues(
                        uuid=str(local_obj.uid),
                    ),
                )

                # Log pcd data
                sem_pcd_entity = base_entity_path + "/sem_pcd" + f"/{local_obj.uid}"
                self.visualizer.log(
                    sem_pcd_entity,
                    # entity_path + "/pcd",
                    self.visualizer.Points3D(
                        positions,
                        colors=curr_obj_color,
                        # radii=0.01
                        # labels=[obj_label],
                    ),
                    self.visualizer.AnyValues(
                        uuid=str(local_obj.uid),
                    ),
                )

                target_bbox_entity = None

                if local_obj.nav_goal:
                    bbox = local_obj.bbox
                    centers = [bbox.get_center()]
                    half_sizes = [bbox.get_extent() / 2]
                    target_bbox_entity = (
                        base_entity_path + "/bbox_target" + f"/{local_obj.uid}"
                    )
                    curr_obj_color = (255, 0, 0)

                    self.visualizer.log(
                        target_bbox_entity,
                        # entity_path + "/bbox",
                        self.visualizer.Boxes3D(
                            centers=centers,
                            half_sizes=half_sizes,
                            # rotations=bbox_quaternion,
                            colors=[curr_obj_color],
                        ),
                        self.visualizer.AnyValues(
                            uuid=str(local_obj.uid),
                        ),
                    )

                bbox = local_obj.bbox
                centers = [bbox.get_center()]
                half_sizes = [bbox.get_extent() / 2]
                # Convert rotation matrix to quaternion(将旋转矩阵转换为四元数)
                # bbox_quaternion = [self.visualizer.rotation_matrix_to_quaternion(bbox.R)]

                bbox_entity = base_entity_path + "/bbox" + f"/{local_obj.uid}"

                obj_bboxes.append(bbox)

                if local_obj.nav_goal:
                    # Set red
                    curr_obj_color = (255, 0, 0)

                self.visualizer.log(
                    bbox_entity,
                    # entity_path + "/bbox",
                    self.visualizer.Boxes3D(
                        centers=centers,
                        half_sizes=half_sizes,
                        # labels=[f"{obj_label}" + "_" + f"{local_obj.downsample_num}"],
                        labels=[f"{obj_label}"],
                        colors=[curr_obj_color],
                    ),
                    self.visualizer.AnyValues(
                        uuid=str(local_obj.uid),
                    ),
                )

            if self.cfg.show_debug_entities:

                self.visualizer.log(
                    "strips",
                    self.visualizer.LineStrips3D(
                        [
                            [
                                [0, 0, 0],
                                [1, 0, 0],
                            ],
                            [
                                [0, 0, 0],
                                [0, 1, 0],
                            ],
                            [
                                [0, 0, 0],
                                [0, 0, 1],
                            ],
                        ],
                        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                        radii=[0.025, 0.025, 0.025],
                        labels=["x_w", "y_w", "z_w"],
                    ),
                )

                # Class info (类别信息)
                class_ids = [obs.class_id for obs in local_obj.observations]
                obj_class_id_counter = Counter(class_ids)
                class_labels = ""

                for idx, data in enumerate(obj_class_id_counter.most_common()):
                    class_labels = (
                        class_labels + str(data[0]) + "_(" + str(data[1]) + ")||"
                    )

                # Split info
                split_info = ""
                for class_id, idx_deque in local_obj.split_info.items():
                    string_split = f"{class_id}" + "-" + f"{list(idx_deque)}"
                    split_info = split_info + string_split + "||"

                # 调试，should_split将为红色
                if local_obj.should_split:
                    curr_obj_color = [255, 0, 0]

                if local_obj.status == LocalObjStatus.WAITING:
                    curr_obj_color = [0, 0, 255]

                if local_obj.status == LocalObjStatus.PENDING:
                    curr_obj_color = [255, 0, 0]

                if local_obj.status == LocalObjStatus.HM_ELIMINATION:
                    curr_obj_color = [0, 0, 0]

                if local_obj.status == LocalObjStatus.LM_ELIMINATION:
                    curr_obj_color = [255, 255, 255]

                if local_obj.is_merged:
                    curr_obj_color = [0, 255, 0]

                # Debug for relations
                # Get all related objects in the
                # 关系调试
                # 获取图中所有相关对象
                related_objs = self.get_related_objects(local_obj.uid)
                related_num = len(related_objs)

                aabb_bbox = local_obj.pcd.get_axis_aligned_bounding_box()
                aabb_centers = [aabb_bbox.get_center()]
                aabb_half_sizes = [aabb_bbox.get_extent() / 2]

                # Baye debug
                # Information from the most recent observation
                # 贝叶斯调试
                # 来自最新观测的信息
                latest_obs = local_obj.get_latest_observation()
                lateset_class_id = latest_obs.class_id
                lateset_class_conf = latest_obs.conf
                lateset_class_distance = latest_obs.distance
                # Maximum common of the current object
                # 当前对象的最大公共数
                prob_class_id = np.argmax(local_obj.class_probs)
                prob_class_conf = local_obj.class_probs[prob_class_id]

                bbox_entity_debug = (
                    base_entity_path + "/bbox_debug" + f"/{local_obj.uid}"
                )
                self.visualizer.log(
                    bbox_entity_debug,
                    # entity_path + "/bbox",
                    self.visualizer.Boxes3D(
                        centers=aabb_centers,
                        half_sizes=aabb_half_sizes,
                        labels=[f"{class_labels}"],
                        colors=[curr_obj_color],
                    ),
                    self.visualizer.AnyValues(
                        uuid=str(local_obj.uid),
                        max_num=int(local_obj.max_common),
                        max_class_one=int(local_obj.split_class_id_one),
                        max_class_two=int(local_obj.split_class_id_two),
                        is_stable=bool(local_obj.is_stable),
                        is_low_mobility=bool(local_obj.is_low_mobility),
                        status=str(local_obj.status),
                        status_pending_count=int(local_obj.pending_count),
                        status_waiting_count=int(local_obj.waiting_count),
                        last_seen_idx=int(local_obj.get_latest_observation().idx),
                        related_num=int(related_num),
                        late_class_id=int(lateset_class_id),
                        late_class_conf=float(lateset_class_conf),
                        prob_class_id=int(prob_class_id),
                        prob_class_conf=float(prob_class_conf),
                        lateset_class_distance=float(lateset_class_distance),
                        entropy=float(local_obj.entropy),
                        change_rate=float(local_obj.change_rate),
                        max_prob=float(local_obj.max_prob),
                    ),
                )

                # Major plane visualization(主平面可视化)
                major_plane_entity = (
                    base_entity_path + "/major_plane" + f"/{local_obj.uid}"
                )
                if local_obj.is_low_mobility:
                    # Change z axis of half_sizes to 0
                    # 将half_sizes的z轴改为0
                    plane_half_sizes = np.copy(aabb_half_sizes)
                    plane_half_sizes[0][2] = 0
                    # Change plane center z value to the z_value
                    # 将平面中心的z值改为z_value
                    plane_center = np.copy(aabb_centers)
                    plane_center[0][2] = local_obj.major_plane_info

                    self.visualizer.log(
                        major_plane_entity,
                        self.visualizer.Boxes3D(
                            centers=plane_center,
                            half_sizes=plane_half_sizes,
                            fill_mode="solid",
                            colors=[curr_obj_color],
                        ),
                    )

                # On relation visualization
                # "On" 关系可视化
                relation_entity = base_entity_path + "/relation" + f"/{local_obj.uid}"
                # Get all related objects in the graph (获取图中所有相关对象)
                related_objs = self.get_related_objects(local_obj.uid)
                if local_obj.is_low_mobility and len(related_objs) > 0:

                    # Get current obj bbox (获取当前对象的包围盒中心)
                    local_obj_center = local_obj.bbox.get_center()

                    # All line information (所有线信息)
                    all_lines = []

                    # Traverse all related objects (遍历所有相关对象)
                    for related_obj in related_objs:

                        # Get the related object's bounding box
                        # 获取相关对象的包围盒中心
                        related_obj_center = related_obj.bbox.get_center()

                        all_lines.append(
                            np.vstack([local_obj_center, related_obj_center]).tolist()
                        )

                    self.visualizer.log(
                        relation_entity,
                        self.visualizer.LineStrips3D(
                            all_lines, colors=[[255, 255, 255]] * len(all_lines)
                        ),
                        self.visualizer.AnyValues(
                            relate=str(related_obj_center),
                        ),
                    )

            if self.cfg.show_local_entities:
                new_logged_entities.add(rgb_pcd_entity)
                new_logged_entities.add(sem_pcd_entity)
                new_logged_entities.add(bbox_entity)

                if target_bbox_entity is not None:
                    new_logged_entities.add(target_bbox_entity)

            if self.cfg.show_debug_entities:
                new_logged_entities.add(bbox_entity_debug)
                new_logged_entities.add(major_plane_entity)
                new_logged_entities.add(relation_entity)

        local_path_entity = "world/local_path"
        if self.nav_graph is not None and self.nav_graph.pos_path is not None:
            # Create a list of 3D points from the pos_path
            # 从pos_path创建3D点列表
            path_points = np.array(self.nav_graph.pos_path)

            # Log the navigation path as a line strip (connecting consecutive points)
            # 将导航路径记录为线段（连接连续的点）
            self.visualizer.log(
                local_path_entity,
                self.visualizer.LineStrips3D(
                    [
                        path_points.tolist()
                    ],  # 将点列表转换为所需格式
                    colors=[[0, 128, 255]],  # 路径为绿色
                ),
            )
            new_logged_entities.add(local_path_entity)

        if len(self.prev_entities) != 0:
            for entity_path in self.prev_entities:
                if entity_path not in new_logged_entities:
                    # logger.info(f"Clearing {entity_path}")
                    self.visualizer.log(
                        entity_path, self.visualizer.Clear(recursive=True)
                    )
        self.prev_entities = new_logged_entities

        # Visualize 3d bbox overlapping(可视化3D包围盒重叠)
        if self.cfg.show_3d_bbox_overlapped:
            self.visualizer.visualize_3d_bbox_overlapping(
                obj_names, obj_colors, obj_bboxes
            )

    def calculate_local_path(
        self, curr_pose, goal_mode=GoalMode.RANDOM, resolution=0.03
    ):
        """计算局部导航路径。"""
        import open3d as o3d

        # Get all pcd from local map(从局部地图获取所有点云)
        total_pcd = o3d.geometry.PointCloud()
        for obj in self.local_map:
            if obj.observed_num <= 3:
                continue
            obj_name = self.visualizer.obj_classes.get_classes_arr()[obj.class_id]
            # Ignore ceiling wall(忽略天花板墙)
            if (
                obj_name == "ceiling wall"
                or obj_name == "carpet"
                or obj_name == "rug"
                or obj_name == "unknown"
            ):
                continue
            total_pcd += obj.pcd

        for obj in self.global_map:
            total_pcd += obj.pcd_2d

        # curr_point_coords = curr_pose[:3, 3]
        # curr_point = o3d.geometry.PointCloud()
        # curr_point.points = o3d.utility.Vector3dVector([curr_point_coords])
        # total_pcd += curr_point

        pcd_points = np.array(total_pcd.points)

        if len(pcd_points) == 0:
            logger.error("[LocalMap][Path] No points in the point cloud!")
            return None

        # Step 1: Constructing 2D occupancy map(构建2D占据地图)
        nav_graph = NavigationGraph(self.cfg, total_pcd, resolution)
        self.nav_graph = nav_graph

        nav_graph.get_occ_map()

        # Get start and goal position(获取起点和终点位置)
        curr_position = curr_pose[:3, 3]
        start_position = nav_graph.calculate_pos_2d(curr_position)

        goal_position = self.get_goal_position(nav_graph, start_position, goal_mode)

        if goal_position is None:
            logger.warning("[LocalMap][Path] No goal position found!")
            return None

        # Step 2: Calculate path
        rrt_path = nav_graph.find_rrt_path(start_position, goal_position)

        if len(rrt_path) == 0:
            logger.warning("[LocalMap][Path] No path found!")
            return None
        else:
            return rrt_path

    def get_goal_position(self, nav_graph, start_position, goal_mode):
        """根据目标模式获取目标位置。"""
        if goal_mode == GoalMode.CLICK:
            logger.info("[LocalMap][Path] Local Goal mode: CLICK")
            return nav_graph.calculate_pos_2d(self.click_goal)

        if goal_mode == GoalMode.INQUIRY:
            logger.info("[LocalMap][Path] Local Goal mode: INQUIRY")
            # Step 1, find local objects within the global best candidate
            # 步骤1: 在全局的最佳候选位置处寻找局部对象
            candidate_objects = self.filter_objects_in_global_bbox(expand_ratio=0.1)

            if len(candidate_objects) == 0:
                logger.warning(
                    "[LocalMap][Path] No local objects found within the global best candidate!"
                )
                return None

            # Step 2. Within filtered objects, find the best score object
            # 步骤2. 在过滤后的对象中，找到得分最高的对象
            local_goal_candidate, candidate_score = (
                self.find_best_candidate_with_inquiry(candidate_objects)
            )

            # If the score is very far from the global score, return None
            # 如果得分与全局得分相差甚远，则返回None
            diff_score = abs(candidate_score - self.global_score)
            if diff_score > self.cfg.object_matching.score_difference:
                logger.warning(
                    "[LocalMap][Path] The local score is too far from the global score: ",
                    diff_score,
                )
                return None

            goal_3d = local_goal_candidate.bbox.get_center()
            goal_2d = nav_graph.calculate_pos_2d(goal_3d)

            # TODO: Check whether to use the global map or the local map
            # TODO: 检查是使用全局地图还是局部地图
            if not nav_graph.free_space_check(goal_2d) is False:

                snapped_goal = self.nav_graph.snap_to_free_space(
                    goal_2d, self.nav_graph.free_space
                )

                goal_2d = np.array(snapped_goal)

            return goal_2d

    def filter_objects_in_global_bbox(self, expand_ratio=0.1):
        """
        Find local objects that fall within the expanded global_bbox in the xy-plane.
        在xy平面上查找落在扩展的global_bbox内的局部对象。

        Parameters:
        - global_bbox: o3d.geometry.AxisAlignedBoundingBox, the global bounding box.
        - local_map: List of objects, each containing an `bbox` attribute of type o3d.geometry.AxisAlignedBoundingBox.
        - expand_ratio: float, the ratio to expand the global_bbox in the xy-plane.
        参数:
        - global_bbox: o3d.geometry.AxisAlignedBoundingBox, 全局包围盒
        - local_map: 对象列表，每个对象包含一个`bbox`属性, 类型为o3d.geometry.AxisAlignedBoundingBox
        - expand_ratio: float, 在xy平面上扩展global_bbox的比率

        Returns:
        - candidate_objects: List of local objects whose bboxes fall within the expanded global_bbox in the xy-plane.
        返回:
        - candidate_objects: 其包围盒在xy平面上落在扩展的global_bbox内的局部对象列表
        """
        # Get the min and max bounds of the global_bbox
        # 获取global_bbox的最小和最大边界
        global_min = np.array(self.global_bbox.min_bound)
        global_max = np.array(self.global_bbox.max_bound)

        # Expand the bbox in the xy-plane(在xy平面上扩展包围盒)
        expand_vector = np.array(
            [
                (global_max[0] - global_min[0]) * expand_ratio,  # Expand x
                (global_max[1] - global_min[1]) * expand_ratio,  # Expand y
                0,
            ]
        )  # No expansion in z
        expanded_min_xy = global_min[:2] - expand_vector[:2]
        expanded_max_xy = global_max[:2] + expand_vector[:2]

        # Filter local objects
        candidate_objects = []
        for obj in self.local_map:
            if obj.observed_num <= 2:
                continue
            obj_bbox = obj.bbox

            # Project the object's bbox onto the xy-plane (ignore z)
            obj_min_xy = np.array([obj_bbox.min_bound[0], obj_bbox.min_bound[1]])
            obj_max_xy = np.array([obj_bbox.max_bound[0], obj_bbox.max_bound[1]])

            # Check if the object's bbox intersects with the expanded global bbox in the xy-plane
            if (
                obj_min_xy[0] <= expanded_max_xy[0]
                and obj_max_xy[0] >= expanded_min_xy[0]
                and obj_min_xy[1] <= expanded_max_xy[1]
                and obj_max_xy[1] >= expanded_min_xy[1]
            ):
                candidate_objects.append(obj)
                obj.nav_goal = True

        return candidate_objects

    def find_best_candidate_with_inquiry(self, candidates):
        """根据查询找到最佳候选对象。"""
        import torch
        import torch.nn.functional as F

        text_query_ft = self.inquiry

        cos_sim = []

        # Loop through each object in the global map to calculate cosine similarity
        # 遍历全局地图中的每个对象以计算余弦相似度
        for obj in candidates:
            if obj.observed_num <= 2:
                continue
            obj.nav_goal = False
            obj_feat = torch.from_numpy(obj.clip_ft).to("cuda")
            max_sim = F.cosine_similarity(
                text_query_ft.unsqueeze(0), obj_feat.unsqueeze(0), dim=-1
            ).item()
            obj_name = self.visualizer.obj_classes.get_classes_arr()[obj.class_id]
            logger.info(f"[LocalMap][Inquiry] =========={obj_name}==============")
            logger.info(f"[LocalMap][Inquiry] Itself: \t{max_sim:.3f}")

            # Store the maximum similarity for this object
            # 存储此对象的最大相似度
            cos_sim.append((obj, max_sim))

        # Now we have a list of tuples [(obj, max_sim), (obj, max_sim), ...]
        # Sort the objects by similarity, from highest to lowest
        # 现在我们有一个元组列表 [(obj, max_sim), (obj, max_sim), ...]
        # 按相似度从高到低对对象进行排序
        sorted_candidates = sorted(cos_sim, key=lambda x: x[1], reverse=True)

        # Get the best candidate (highest cosine similarity)
        # 获取最佳候选对象（最高余弦相似度）
        best_candidate, best_similarity = sorted_candidates[0]

        # Output the best candidate and its similarity
        # 输出最佳候选对象及其相似度
        best_candidate_name = self.visualizer.obj_classes.get_classes_arr()[
            best_candidate.class_id
        ]
        logger.info(
            f"[LocalMap][Inquiry] Best Candidate: '{best_candidate_name}' with similarity: {best_similarity:.3f}"
        )

        # Set flag to the best candidate for visualization
        # 为可视化设置最佳候选对象的标志
        best_candidate.nav_goal = True

        logger.info(f"[LocalMap][Inquiry] global score: {self.global_score:.3f} ")

        return best_candidate, best_similarity

    def compute_pose_difference(self, curr_pose, prev_pose):
        """
        Calculate the translation and rotation difference between current and previous poses.
        计算当前位姿和前一帧位姿之间的平移和旋转差异。

        Parameters:
            curr_pose (np.ndarray): Current pose 4x4 homogeneous transformation matrix.
            prev_pose (np.ndarray): Previous frame pose 4x4 homogeneous transformation matrix.
        参数:
            curr_pose (np.ndarray): 当前位姿4x4齐次变换矩阵。
            prev_pose (np.ndarray): 前一帧位姿4x4齐次变换矩阵。

        Returns:
            tuple: (translation difference norm, rotation difference norm)
        返回:
            tuple: (平移差异范数, 旋转差异范数)
        """
        if prev_pose is not None:
            # Extract translation vector
            # 提取平移向量
            curr_pos = curr_pose[:3, 3]
            prev_pos = prev_pose[:3, 3]

            # Calculate translation difference norm
            # 计算平移差异范数
            delta_translation = np.linalg.norm(curr_pos - prev_pos)

            # Extract rotation matrix
            # 提取旋转矩阵
            curr_rot = curr_pose[:3, :3]
            prev_rot = prev_pose[:3, :3]

            # Calculate rotation matrix difference
            # 计算旋转矩阵差异
            delta_rotation_matrix = (
                curr_rot @ prev_rot.T
            )  
            # Current rotation matrix multiplied by previous frame rotation matrix transpose
            # 当前旋转矩阵乘以先前帧旋转矩阵的转置
            angle = np.arccos(
                np.clip((np.trace(delta_rotation_matrix) - 1) / 2, -1.0, 1.0)
            )  
            # Rotation angle (radians)
            # 旋转角度（弧度）
            delta_rotation = np.degrees(
                angle
            )  # Convert to degrees for easier observation

            return delta_translation, delta_rotation
        else:
            return None, None
