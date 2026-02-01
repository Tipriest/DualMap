"""
Mission 5: 高阶记忆寻物任务
- 从记忆上下文中推理 target_object, target_room, related_object
- 支持两种模式：指定房间 或 全屋优先级排序
- 根据记忆上下文进行智能推理和搜索策略
- 整合 task_2_3 和 task_3++ 的搜索能力
"""

import os
os.environ["DISPLAY"] = ""

import sys
import time
import math
import yaml
import json
import requests
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor
from action_msgs.msg import GoalStatus

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # applications/
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)  # DualMap/
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.path.append(os.path.join(PROJECT_ROOT, "applications/utils"))

from applications.query_task_subscriber import TaskSubscriber, write_log

LOG_FILE = "nav_result_mission5.txt"


def parse_memory_with_qwen(cfg_path: str, memory_context: str, user_query: str, search_mode: str):
    """
    使用 Qwen API 从记忆上下文和任务指令中推理信息

    Args:
        cfg_path: 配置文件路径
        memory_context: 记忆上下文（描述物品位置关系）
        user_query: 任务指令（如"找到妈妈早上在厨房用过的烧水壶"）
        search_mode: 搜索模式（"single_room" 或 "multi_room"）

    Returns:
        dict: {
            "target_object": str,     # 目标物体（从记忆中推理）
            "target_room": str,       # 目标房间（从记忆中推理，或"None"）
            "related_object": str,    # 相关物体（从记忆中推理，或"None"）
            "room_priority": list     # 房间优先级（全屋模式）
        }
    """
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    api_key = cfg["api_key"]
    base_url = os.getenv(
        "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    if not api_key:
        raise ValueError("请在配置文件中设置 api_key")

    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    prompt = f"""
你是一个智能机器人助手，需要根据记忆上下文和任务指令推理出物品信息。

【记忆上下文】
{memory_context}

【任务指令】
{user_query}

【搜索模式】
{search_mode}

【可用房间】
只有4个房间：livingroom(客厅), bedroom(卧室), childroom(儿童房), kitchen(厨房)

【物体词汇表】
必须从以下列表中选择对应的英文单词：
* 书包→backpack, 相框→picture frame, 篮球→basketball, 碗→bowl
* 香蕉→banana, 苹果→apple, 木马/玩具马→toy horse, 椅子→chair
* 沙发→couch, 绿植/植物→green plant, 床→bed, 桌子/书桌→table
* 电视→tv, 笔记本电脑→laptop, 微波炉→microwave, 柜子→cabinet
* 毛绒玩具/玩偶→soft toy, 地毯→carpet, 台灯→table lamp
* 床头柜→nightstand, 帐篷→tent, 积木→building blocks
* 书架→bookshelf, 燃气灶/煤气灶→gas stove, 锅→pot, 水壶/烧水壶→kettle
* 菜篮/食物篮→food basket, 水龙头→faucet

【推理任务】
1. **target_object**: 从记忆上下文中找到任务指令要找的物品（必须存在）
2. **target_room**: 从记忆上下文推理出该物品最可能在哪个房间
3. **related_object**: 从记忆上下文推理出辅助定位的相关物体（如"煤气灶旁的水壶"→related是"gas stove"）
4. **room_priority**: 如果是全屋模式，按可能性从高到低排序所有4个房间

【推理规则】
- 仔细阅读记忆上下文，找到任务指令中提到的物品及其位置描述
- target_object 必须从词汇表中选择准确的英文名称
- target_room 必须是 livingroom/bedroom/childroom/kitchen 之一，如果记忆中没有明确房间则为 "None"
- related_object 是帮助定位的参照物（如"桌上的书包"→related是"table"），如果没有则为 "None"
- 注意区分房间：书桌在卧室→bedroom，儿童玩具在儿童房→childroom，煤气灶在厨房→kitchen
- 地面/地板（floor/ground related_object

【输出格式】
只返回 JSON，不要其他文本：
{{
    "target_object": "物品英文名称",
    "target_room": "房间名称或None",
    "related_object": "相关物体英文名称或None",
    "room_priority": ["kitchen", "bedroom", "livingroom", "childroom"]
}}

【示例】
记忆：妈妈经常在厨房准备美食，她早晨用过煤气灶旁的水壶烧开水
指令：请找到妈妈早上在厨房用过的烧水壶
输出：
{{
    "target_object": "kettle",
    "target_room": "kitchen",
    "related_object": "gas stove",
    "room_priority": ["kitchen", "livingroom", "bedroom", "childroom"]
}}

现在请根据上述记忆上下文和任务指令生成JSON：
"""

    payload = {
        "model": "qwen-max",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "top_p": 0.8,
        "stream": False,
        "max_tokens": 1024,
    }

    try:
        response = requests.post(
            url, headers=headers, data=json.dumps(payload), timeout=30
        )
        response.raise_for_status()
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"].strip()
            # 尝试解析 JSON
            try:
                data = json.loads(content)
                room_priority = data.get("room_priority", ["livingroom", "bedroom", "childroom", "kitchen"])

                # 确保包含所有4个房间
                all_rooms = ["livingroom", "bedroom", "childroom", "kitchen"]
                for room in all_rooms:
                    if room not in room_priority:
                        room_priority.append(room)

                return {
                    "target_object": data.get("target_object", "bottle"),
                    "target_room": data.get("target_room", "None"),
                    "related_object": data.get("related_object", "None"),
                    "room_priority": room_priority[:4]  # 只取前4个
                }
            except json.JSONDecodeError:
                print(f"[WARN] 无法解析 LLM 响应为 JSON: {content}")
                return {
                    "target_object": "bottle",
                    "target_room": "None",
                    "related_object": "None",
                    "room_priority": ["livingroom", "bedroom", "childroom", "kitchen"]
                }
        else:
            print("[WARN] LLM 响应格式异常")
            return {
                "target_object": "bottle",
                "target_room": "None",
                "related_object": "None",
                "room_priority": ["livingroom", "bedroom", "childroom", "kitchen"]
            }

    except Exception as e:
        print(f"[ERROR] LLM 调用失败: {repr(e)}")
        return {
            "target_object": "bottle",
            "target_room": "None",
            "related_object": "None",
            "room_priority": ["livingroom", "bedroom", "childroom", "kitchen"]
        }


class Mission5Subscriber(TaskSubscriber):
    """
    Mission 5: 高阶记忆寻物任务
    整合 task_2_3 和 task_3++ 的功能，支持两种模式
    """

    def __init__(self, cfg_path: str):
        super().__init__(cfg_path)

        # 新增状态变量
        self.current_goal_handle = None  # 当前导航的 goal handle
        self.target_found = False  # 是否找到目标
        self.rooms_to_visit = []  # 待访问的房间列表
        self.searched_rooms = []  # 已搜索的房间列表
        self.room_priority = []  # LLM 推理的房间优先级
        self.user_specified_room = False  # 用户是否指定了房间
        self.search_mode = "multi_room"  # 搜索模式：single_room 或 multi_room

        # 重写 worker
        self._worker = None  # 停止父类的 worker
        self._worker = threading.Thread(target=self._task_worker_mission5, daemon=True)
        self._worker.start()

    def _task_worker_mission5(self):
        """
        Mission 5 主工作流程：根据记忆上下文和搜索模式执行任务

        模式1 (single_room): 有房间+有related → 参考task_2_3逻辑
        模式2 (multi_room): 全屋遍历 → 参考task_3++逻辑
        """
        while not self._shutdown_event.is_set():
            self._task_event.wait(timeout=0.2)
            if self._shutdown_event.is_set():
                break
            if not self._task_event.is_set():
                continue

            with self._lock:
                target_name = self.target_name
                room = self.room
                related_name = self.related_object_name

            self._task_event.clear()

            if not target_name:
                self.get_logger().warn("Worker triggered but target_name is empty.")
                continue

            try:
                # ========== 重置状态 ==========
                self.target_found = False
                self.searched_rooms = []
                with self._lock:
                    self.target_x = None
                    self.target_y = None

                # ========== 根据模式选择搜索策略 ==========
                if self.search_mode == "single_room" and room and room != "None":
                    # 模式1: 指定房间搜索（类似task_2_3）
                    self.get_logger().info(f"[模式1] 指定房间搜索: 房间={room}, 目标={target_name}, 相关={related_name}")
                    self._execute_single_room_search(room, target_name, related_name)
                else:
                    # 模式2: 全屋遍历搜索（类似task_3++）
                    self.get_logger().info(f"[模式2] 全屋遍历搜索: 目标={target_name}")
                    self._execute_multi_room_search(target_name)

            except Exception as e:
                self.get_logger().error(f"Worker exception: {repr(e)}")
                import traceback
                traceback.print_exc()

    def _execute_single_room_search(self, room_name: str, target_name: str, related_name: str):
        """
        模式1: 指定房间搜索（参考task_2_3_v8）

        流程:
        1. 如果有related：先找related → 确认后搜索target
        2. 如果无related：直接去房间锚点 → fallback搜索target
        """
        self.get_logger().info(f"===== 开始单房间搜索: {room_name} =====")
        write_log(f"单房间搜索: 房间={room_name}, 目标={target_name}, 相关={related_name}")

        # 设置房间
        self._room_cb(room_name)
        # time.sleep(0.3)

        self.get_logger().info(f"等待房间 {room_name} 信息就绪...")
        t0 = time.time()
        room_ready = False
        while True:
            with self._lock:
                room_ready = self.is_room_ready
                room_bbox = self.room_bbox

            if room_ready and room_bbox is not None:
                self.get_logger().info(f"✓ 房间 {room_name} 已就绪, bbox={room_bbox}")
                break

            if time.time() - t0 > 3.0:
                self.get_logger().error(f"❌ 等待房间 {room_name} 超时（3秒）")
                self._return_home_and_exit(f"FAIL: 房间{room_name}信息未就绪")
                return

            time.sleep(0.05)

        has_related = related_name and related_name != "None"

        if has_related:
            # 有related：先找related
            self.get_logger().info(f"→ 在{room_name}中搜索相关物体: {related_name}")
            import numpy as np

            rcorners = self.query_callback(related_name)
            if rcorners is None:
                self.get_logger().warn(f"⚠️ 在{room_name}中未找到{related_name}，使用fallback")
                self._fallback_search_at_anchor(target_name, room_name)
                return

            # 计算related位置并导航
            rpos = np.array(rcorners).mean(axis=0)
            rx, ry = float(rpos[0]), float(rpos[1])

            self.get_logger().info(f"✓ 找到{related_name} at ({rx:.2f}, {ry:.2f})")

            # 导航到related附近
            free_x, free_y = self.find_optimal_free_point_by_room_center(rx, ry, 2.0)
            dx, dy = rx - free_x, ry - free_y
            yaw = math.atan2(dy, dx)

            nav_ok = self._goto_point_with_early_stop(
                free_x, free_y, yaw,
                frame_id="map", wait_timeout=5.0, stop_distance=0.7
            )

            if not nav_ok:
                self.get_logger().warn("导航到related失败，使用fallback")
                self._fallback_search_at_anchor(target_name, room_name)
                return

            # 在related附近搜索target
            self.get_logger().info(f"→ 在{related_name}附近搜索{target_name}")
            time.sleep(0.5)

            # 发布检索请求
            self.publish_related_bbox(rx, ry, 3.0, 3.0, target_name)

            # 旋转搜索
            found = self._spin_and_wait_for_target(3)
            if found:
                self._navigate_and_check_target(target_name)
            else:
                self.get_logger().warn(f"在{related_name}附近未找到{target_name}")
                self._return_home_and_exit(f"FAIL: 在{room_name}的{related_name}附近未找到{target_name}")
        else:
            # 无related：直接fallback搜索
            self.get_logger().info(f"→ 无相关物体，直接在{room_name}锚点搜索")
            self._fallback_search_at_anchor(target_name, room_name)

    def _execute_multi_room_search(self, target_name: str):
        """
        模式2: 全屋遍历搜索（参考task_3++_v8）
        """
        # 确定房间顺序
        if self.room_priority and len(self.room_priority) > 0:
            self.rooms_to_visit = self.room_priority
            self.get_logger().info(f"使用 LLM 推理的房间顺序: {self.room_priority}")
        else:
            self.rooms_to_visit = ["livingroom", "bedroom", "childroom", "kitchen"]
            self.get_logger().info("使用默认房间顺序")

        write_log(f"开始遍历房间寻找 {target_name}，顺序: {self.rooms_to_visit}")

        # 遍历每个房间
        for room_name in self.rooms_to_visit:
            if self.target_found:
                self.get_logger().info("✓ 目标已找到，停止遍历")
                break

            self.get_logger().info(f"===== 开始访问房间: {room_name} =====")
            self.searched_rooms.append(room_name)

            self._room_cb(room_name)
            time.sleep(0.3)


            # 获取房间锚点配置，支持单个或多个anchor
            anchor_config = self.room_anchors.get(room_name, None)
            if anchor_config is None:
                self.get_logger().warn(f"⚠ 房间 {room_name} 没有锚点配置，跳过")
                continue

            # 统一转换为列表格式
            if isinstance(anchor_config, list):
                if len(anchor_config) == 2 and isinstance(anchor_config[0], (int, float)):
                    anchor_list = [anchor_config]  # 单个anchor
                else:
                    anchor_list = anchor_config  # 多个anchor
            else:
                self.get_logger().warn(f"⚠ 房间 {room_name} 锚点配置格式错误，跳过")
                continue

            self.get_logger().info(f"房间 {room_name} 有 {len(anchor_list)} 个锚点")

            # 发布房间 bbox
            self._publish_room_bbox_for_search(room_name)

            # 遍历该房间的所有anchor点
            for anchor_idx, anchor_pt in enumerate(anchor_list, 1):
                if self.target_found:
                    self.get_logger().info("✓ 目标已找到，停止遍历anchor")
                    break

                if len(anchor_pt) != 2:
                    self.get_logger().warn(f"跳过无效锚点: {anchor_pt}")
                    continue

                anchor_x, anchor_y = anchor_pt

                # 导航到当前anchor（可中断）
                self.get_logger().info(f"→ 导航到 {room_name} 第{anchor_idx}/{len(anchor_list)}个锚点: ({anchor_x:.2f}, {anchor_y:.2f})")
                ok = self._goto_point_interruptible(
                    anchor_x, anchor_y, yaw=0.0, frame_id="map", wait_timeout=5.0
                )

                if self.target_found:
                    self.get_logger().info("✓ 导航过程中收到目标位置")
                    break

                if not ok:
                    self.get_logger().warn(f"✗ 导航到锚点{anchor_idx}失败，尝试下一个")
                    continue

                # 旋转扫描
                self.get_logger().info(f"↻ 到达锚点{anchor_idx}，开始360度旋转扫描")
                for _ in range(3):
                    self._spin_360_continuous()
                    if self.target_found:
                        break

                if self.target_found:
                    self.get_logger().info("✓ 旋转过程中目标已找到")
                    break

                time.sleep(0.5)
                with self._lock:
                    if self.target_x is not None and self.target_y is not None:
                        self.target_found = True
                        self.get_logger().info(f"✓ 在锚点{anchor_idx}找到目标")
                        break

                self.get_logger().info(f"在锚点{anchor_idx}未找到目标，继续下一个锚点")

            # 检查是否在该房间找到了目标
            if self.target_found:
                break

        # 处理结果
        if self.target_found:
            self._handle_target_found()
        else:
            self._handle_all_rooms_not_found()

    def _spin_and_wait_for_target(self, rounds: int) -> bool:
        """旋转并等待目标（参考task_2_3_v8）"""
        yaw_increments = [math.pi / 4 * i for i in range(8)]
        cx, cy = self.current_x, self.current_y

        for round_num in range(rounds):
            self.get_logger().info(f"↻ 开始第 {round_num + 1}/{rounds} 轮旋转")
            for yaw in yaw_increments:
                self._goto_point(cx, cy, yaw=yaw, frame_id="map", wait_timeout=5.0)
                time.sleep(0.2)

                with self._lock:
                    if self.target_x is not None and self.target_y is not None:
                        self.get_logger().info("✓ 找到目标")
                        return True

        return False

    def _fallback_search_at_anchor(self, target_name: str, room_name: str):
        """在房间锚点fallback搜索（参考task_2_3_v8），支持多个anchor点"""
        self.get_logger().info(f"===== Fallback: 在{room_name}锚点搜索 =====")

        # 获取房间的anchor配置，支持单个或多个anchor
        anchor_config = self.room_anchors.get(room_name, None)
        if anchor_config is None:
            self.get_logger().error(f"房间{room_name}没有锚点配置")
            self._return_home_and_exit(f"FAIL: {room_name}无锚点配置")
            return

        # 统一转换为列表格式
        if isinstance(anchor_config, list):
            # 检查是否是坐标对 [x, y] 还是坐标列表 [[x1, y1], [x2, y2], ...]
            if len(anchor_config) == 2 and isinstance(anchor_config[0], (int, float)):
                # 单个anchor: [x, y]
                anchor_list = [anchor_config]
            else:
                # 多个anchor: [[x1, y1], [x2, y2], ...]
                anchor_list = anchor_config
        else:
            self.get_logger().error(f"房间{room_name}的锚点配置格式错误")
            self._return_home_and_exit(f"FAIL: {room_name}锚点配置错误")
            return

        self.get_logger().info(f"房间{room_name}共有{len(anchor_list)}个锚点")

        # 遍历所有anchor点进行搜索
        for anchor_idx, anchor_pt in enumerate(anchor_list, 1):
            if len(anchor_pt) != 2:
                self.get_logger().warn(f"跳过无效锚点配置: {anchor_pt}")
                continue

            anchor_x, anchor_y = anchor_pt
            self.get_logger().info(f"→ 尝试第{anchor_idx}/{len(anchor_list)}个锚点: ({anchor_x:.2f}, {anchor_y:.2f})")

            # 尝试在当前anchor搜索
            found = self._search_at_single_anchor(target_name, room_name, anchor_x, anchor_y, anchor_idx)

            if found:
                # 找到目标，导航过去
                self._navigate_and_check_target(target_name)
                return
            else:
                self.get_logger().info(f"✗ 在第{anchor_idx}个锚点未找到{target_name}，尝试下一个")

        # 所有anchor都尝试完毕，仍未找到
        self.get_logger().warn(f"在{room_name}的所有{len(anchor_list)}个锚点均未找到{target_name}")
        self._return_home_and_exit(f"FAIL: 在{room_name}的所有锚点均未找到{target_name}")

    def _search_at_single_anchor(self, target_name: str, room_name: str, anchor_x: float, anchor_y: float, anchor_idx: int) -> bool:
        """在单个anchor点进行搜索

        Args:
            target_name: 目标物体名称
            room_name: 房间名称
            anchor_x, anchor_y: anchor坐标
            anchor_idx: anchor索引（用于日志）

        Returns:
            bool: True=找到目标, False=未找到
        """
        # 导航到anchor点
        ok = self._goto_point(anchor_x, anchor_y, yaw=0.0, frame_id="map", wait_timeout=5.0)

        if not ok:
            self.get_logger().warn(f"无法到达锚点{anchor_idx} ({anchor_x:.2f}, {anchor_y:.2f})")
            return False

        self.get_logger().info(f"✓ 到达锚点{anchor_idx}，开始搜索")

        # 发布检索请求
        room_bbox = self.room_edges.get(room_name, None)
        if room_bbox:
            min_x, max_x, min_y, max_y = room_bbox
            room_cx = (min_x + max_x) / 2.0
            room_cy = (min_y + max_y) / 2.0
            room_w = max_x - min_x
            room_h = max_y - min_y
            self.publish_related_bbox(room_cx, room_cy, room_w, room_h, target_name)

        # 旋转搜索
        found = self._spin_and_wait_for_target(3)
        return found

    def _navigate_and_check_target(self, target_name: str):
        """导航到目标并确认（参考task_2_3_v8）"""
        with self._lock:
            final_target_x = self.target_x
            final_target_y = self.target_y

        self.get_logger().info(f"✓ dualmap返回目标位置: ({final_target_x:.2f}, {final_target_y:.2f})")
        self.get_logger().info(f"→ 导航到'{target_name}'... ⏹0.5m提前停止")

        final_free_x, final_free_y = self.find_optimal_free_point_by_room_center(
            final_target_x, final_target_y, 2.5
        )

        dx = final_target_x - final_free_x
        dy = final_target_y - final_free_y
        target_yaw = math.atan2(dy, dx)

        nav_ok = self._goto_point_with_early_stop(
            final_free_x, final_free_y, target_yaw,
            frame_id="map", wait_timeout=5.0, stop_distance=0.7
        )

        if not nav_ok:
            # 多次尝试接近目标
            dx = self.target_x - self.current_x
            dy = self.target_y - self.current_y
            dyaw = math.atan2(dy, dx)
            for i in range(1, 6):
                self.get_logger().warn(f"⚠️  导航失败，尝试第{5-i}/5个点")
                new_point_x = self.target_x - i /5.0 * dx
                new_point_y = self.target_y - i /5.0 * dy
                nav_ok = self._goto_point_with_early_stop(
                    new_point_x, new_point_y, dyaw,
                    frame_id="map", wait_timeout=5.0, stop_distance=0.5
                )
                if not nav_ok and i==5:
                    nav_ok = self._goto_point_with_early_stop(
                        self.current_x, self.current_y, dyaw,
                        frame_id="map", wait_timeout=5.0, stop_distance=0.5
                    )
                elif nav_ok:
                    break

        # 拍照确认
        self.get_logger().info(f"✓ 到达目标位置，拍照确认...")
        time.sleep(0.5)

        if self.latest_image is not None:
            import cv2
            cv_image = self.latest_image.copy()
            save_path = self._save_rgb_snapshot(cv_image, prefix="success")
            if save_path:
                self.get_logger().info(f"📸 已保存成功图片: {save_path}")

        self.get_logger().info(f"✓ 任务成功：找到了'{target_name}'")
        write_log(f"任务成功: {target_name}")

        time.sleep(5.0)
        self._return_home_and_exit("Task completed successfully")

        write_log(f"开始遍历房间寻找 {target_name}，顺序: {self.rooms_to_visit}")

        # ========== 遍历每个房间进行搜索 ==========
        for room_name in self.rooms_to_visit:
            # 检查1: 如果已找到目标，立即停止遍历
            if self.target_found:
                self.get_logger().info("✓ 目标已找到，停止遍历后续房间")
                break

            self.get_logger().info(f"===== 开始访问房间: {room_name} =====")
            self.searched_rooms.append(room_name)  # 维护已搜索房间列表

            # 步骤1: 设置当前房间（更新父类的 self.room 和相关状态）
            self._room_cb(room_name)
            time.sleep(0.3)  # 等待房间边界信息就绪

            # 步骤2: 获取房间锚点（预设的房间观察位置）
            anchor_pt = self.room_anchors.get(room_name, None)
            if anchor_pt is None:
                self.get_logger().warn(f"⚠ 房间 {room_name} 没有锚点配置，跳过")
                continue

            anchor_x, anchor_y = anchor_pt

            # 步骤3: 发布房间 bbox 给 dualmap 进行目标检索
            # dualmap 会在该房间范围内搜索目标物体
            self._publish_room_bbox_for_search(room_name)

            # 步骤4: 导航到房间锚点（可中断导航）
            # 如果在导航过程中收到 dualmap 响应，会设置 target_found=True
            self.get_logger().info(f"→ 导航到 {room_name} 锚点: ({anchor_x:.2f}, {anchor_y:.2f})")
            ok = self._goto_point_interruptible(
                anchor_x, anchor_y, yaw=0.0, frame_id="map", wait_timeout=5.0
            )

            # 检查2: 导航过程中是否收到目标位置
            if self.target_found:
                self.get_logger().info("✓ 导航过程中收到目标位置，中断后续流程")
                break

            if not ok:
                self.get_logger().warn(f"✗ 导航到 {room_name} 失败，继续下一个房间")
                continue

            # 步骤5: 到达后连续旋转360度（不停留，快速扫描）
            # 旋转过程中 dualmap 持续检索，增加发现目标的概率
            self.get_logger().info(f"↻ 到达 {room_name}，开始360度旋转扫描")
            self._spin_360_continuous()
            self.get_logger().info(f"↻ 到达 {room_name}，开始360度旋转扫描")
            self._spin_360_continuous()
            self.get_logger().info(f"↻ 到达 {room_name}，开始360度旋转扫描")
            self._spin_360_continuous()
            # 检查3: 旋转过程中是否找到目标
            if self.target_found:
                self.get_logger().info("✓ 旋转过程中目标已找到")
                break

            # 步骤6: 旋转完成后等待 dualmap 最终响应
            # 给 dualmap 一点时间完成检索并返回结果
            time.sleep(0.5)
            with self._lock:
                if self.target_x is not None and self.target_y is not None:
                    self.target_found = True
                    self.get_logger().info("✓ 收到 dualmap 最终响应，目标已找到")
                    break

                # ========== 任务结果处理 ==========
                if self.target_found:
                    # 情况A: 找到目标 → 导航到目标 → 返回起点
                    self._handle_target_found()
                else:
                    # 情况B: 未找到目标 → 根据模式决定处理方式
                    if self.user_specified_room:
                        # 模式1: 用户指定房间但未找到 → 直接返回起点报FAIL
                        self._handle_specified_room_not_found()
                    else:
                        # 模式2/3: 遍历所有房间都未找到 → 返回起点报FAIL
                        self._handle_all_rooms_not_found()

    def _publish_room_bbox_for_search(self, room_name: str):
        """
        发布房间 bbox 给 dualmap 进行检索

        功能说明:
        1. 获取房间的边界信息（从配置文件 room_edges 读取）
        2. 计算房间的中心点和尺寸
        3. 通过 ROS topic 发布给 dualmap，告诉它在这个范围内搜索目标
        4. dualmap 收到后会在该区域的点云/语义地图中检索目标物体

        Args:
            room_name: 房间名称（livingroom/bedroom/childroom/kitchen）
        """
        room_bbox = self.room_edges.get(room_name, None)
        if room_bbox is None:
            self.get_logger().warn(f"⚠ 房间 {room_name} 没有边界信息配置")
            return

        # 解包房间边界：[min_x, max_x, min_y, max_y]
        min_x, max_x, min_y, max_y = room_bbox
        room_cx = (min_x + max_x) / 2.0
        room_cy = (min_y + max_y) / 2.0
        room_w = max_x - min_x
        room_h = max_y - min_y

        self.get_logger().info(
            f"📤 发布检索请求到 dualmap: 房间={room_name}, "
            f"中心=({room_cx:.2f}, {room_cy:.2f}), 尺寸=({room_w:.2f}x{room_h:.2f})"
        )

        # 调用父类方法发布检索请求（通过 /dualmap/search_request topic）
        self.publish_related_bbox(room_cx, room_cy, room_w, room_h, self.target_name)

    def _goto_point_interruptible(
        self, x: float, y: float, yaw: float, frame_id: str, wait_timeout: float
    ) -> bool:
        """
        可中断的导航：在导航过程中检查是否收到目标位置，如果收到则取消当前导航

        与普通 _goto_point 的区别:
        - 普通: 阻塞等待导航完成，不可中断
        - 可中断: 在等待过程中持续检查 target_found 标志，一旦为 True 立即取消导航

        实现机制:
        1. 发送导航目标给 Nav2
        2. 在等待结果期间，每 0.1 秒检查一次 target_found
        3. 如果 dualmap 返回了目标位置（remap_target_callback 设置 target_found=True）
        4. 立即调用 cancel_goal_async() 取消当前导航任务
        5. 返回 False 表示被中断（但这是期望的行为）

        Args:
            x, y: 目标位置世界坐标
            yaw: 目标朝向角度
            frame_id: 坐标系（通常是 "map"）
            wait_timeout: 等待 action server 的超时时间

        Returns:
            bool: True=正常到达, False=被中断或失败（需要调用方检查 target_found 区分）
        """
        if not self._client.wait_for_server(timeout_sec=wait_timeout):
            self.get_logger().error("NavigateToPose server not available")
            return False

        from geometry_msgs.msg import PoseStamped
        from nav2_msgs.action import NavigateToPose

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = frame_id
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)

        from applications.query_task_subscriber import yaw_to_quaternion
        qx, qy, qz, qw = yaw_to_quaternion(float(yaw))
        goal_msg.pose.pose.orientation.x = qx
        goal_msg.pose.pose.orientation.y = qy
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw

        self.get_logger().info(f"发送导航目标: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")

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
                self.current_goal_handle = gh  # 保存 handle 用于取消

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

        # 等待导航完成，同时检查是否收到目标
        nav_timeout = 300.0
        start_time = time.time()

        while True:
            if done_evt.wait(timeout=0.1):
                break

            # 检查是否超时
            if time.time() - start_time > nav_timeout:
                self.get_logger().error("导航超时")
                self._cancel_current_goal()
                return False

            # 检查是否收到目标位置
            with self._lock:
                if self.target_x is not None and self.target_y is not None:
                    self.target_found = True
                    self.get_logger().info("收到目标位置，取消当前导航")
                    self._cancel_current_goal()
                    return False

        if result_holder["accepted"] is not True:
            return False

        status = result_holder["status"]
        if status == GoalStatus.STATUS_SUCCEEDED:
            return True

        return False

    def _cancel_current_goal(self):
        """取消当前的导航目标"""
        if self.current_goal_handle is not None:
            try:
                self.get_logger().info("取消当前导航目标")
                cancel_future = self.current_goal_handle.cancel_goal_async()
                # 不等待取消完成，直接返回
            except Exception as e:
                self.get_logger().error(f"取消导航失败: {repr(e)}")
            finally:
                self.current_goal_handle = None

    def _goto_point_with_early_stop(
        self, target_x: float, target_y: float, yaw: float,
        frame_id: str, wait_timeout: float, stop_distance: float = 0.5
    ) -> bool:
        """
        导航到目标点，但在距离目标0.5m时提前停止

        Args:
            target_x, target_y: 目标位置
            yaw: 目标朝向
            frame_id: 坐标系
            wait_timeout: 等待超时
            stop_distance: 提前停止距离（默认0.5m）

        Returns:
            bool: True=成功到达或提前停止, False=失败
        """
        if not self._client.wait_for_server(timeout_sec=wait_timeout):
            self.get_logger().error("NavigateToPose server not available")
            return False

        from geometry_msgs.msg import PoseStamped
        from nav2_msgs.action import NavigateToPose
        from applications.query_task_subscriber import yaw_to_quaternion

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = frame_id
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(target_x)
        goal_msg.pose.pose.position.y = float(target_y)

        qx, qy, qz, qw = yaw_to_quaternion(float(yaw))
        goal_msg.pose.pose.orientation.x = qx
        goal_msg.pose.pose.orientation.y = qy
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw

        self.get_logger().info(f"发送导航目标: ({target_x:.2f}, {target_y:.2f}), 将在{stop_distance}m处提前停止")

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
                self.current_goal_handle = gh

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
        start_time = time.time()

        while True:
            if done_evt.wait(timeout=0.1):
                break

            if time.time() - start_time > nav_timeout:
                self.get_logger().error("导航超时")
                self._cancel_current_goal()
                return False

            # 检查距离，提前0.5m停止
            current_x = self.current_x
            current_y = self.current_y
            distance = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)

            if distance <= stop_distance:
                self.get_logger().info(f"✓ 距离目标{distance:.2f}m，提前停止导航")
                self._cancel_current_goal()
                return True

        if result_holder["accepted"] is not True:
            return False

        status = result_holder["status"]
        if status == GoalStatus.STATUS_SUCCEEDED:
            return True
        elif status == GoalStatus.STATUS_CANCELED:
            return True

        return False

    def _spin_360_continuous(self):
        """
        连续旋转360度扫描房间，不等待每个角度完成

        实现策略:
        - 每45度发送一次旋转指令（8个方向）
        - 不等待每个旋转完成，快速连续发送
        - 在旋转过程中持续检查 target_found 和 target_x/target_y
        - 一旦发现目标立即停止旋转

        为什么这样设计:
        1. 快速扫描：不等待完成可以更快覆盖360度
        2. 实时响应：dualmap 在任意角度都可能返回结果
        3. 提高效率：找到目标后立即停止，不浪费时间
        """
        yaw_increments = [math.pi / 4 * i for i in range(8)]  # 0, 45, 90, ..., 315度
        cx, cy = self.current_x, self.current_y

        for yaw in yaw_increments:
            if self.target_found:
                self.get_logger().info("旋转过程中目标已找到，停止旋转")
                break

            # 不等待导航完成，直接发送下一个角度
            self._goto_point(cx, cy, yaw=yaw, frame_id="map", wait_timeout=5.0)

            # 检查是否收到目标
            with self._lock:
                if self.target_x is not None and self.target_y is not None:
                    self.target_found = True
                    self.get_logger().info("旋转中收到目标位置")
                    break

    def _handle_target_found(self):
        """
        处理找到目标的情况：完整的目标处理流程
        """
        with self._lock:
            target_x = self.target_x
            target_y = self.target_y

        if target_x is None or target_y is None:
            self.get_logger().error("目标位置为空")
            return

        self.get_logger().info(f"导航到目标: ({target_x:.2f}, {target_y:.2f}) ⏹提前0.5m停止")
        write_log(f"目标找到: {self.target_name} at ({target_x:.2f}, {target_y:.2f})")

        # 计算最近可达点
        free_x, free_y = self.find_optimal_free_point_by_room_center(
            target_x, target_y, 1.5
        )

        # 计算朝向目标的yaw角
        dx = target_x - free_x
        dy = target_y - free_y
        target_yaw = math.atan2(dy, dx)

        # 使用提前截断的导航方法
        nav_ok = self._goto_point_with_early_stop(
            free_x, free_y, target_yaw,
            frame_id="map", wait_timeout=5.0, stop_distance=0.5
        )

        # FIXME: 这个地方不能再使用VLM进行check
        # if nav_ok:
        #     # RGB check 验证
        #     is_complete = self.check_task()
        #     if is_complete:
        #         write_log(f"任务完成: {self.target_name}")
        #         self.get_logger().info("任务验证成功")
        #     else:
        #         write_log(f"任务验证失败: {self.target_name}")
        #         self.get_logger().warn("RGB check 失败，进入 recovery")
        #         self.run_recovery()
        # else:
        if not nav_ok:
            dx = self.target_x - self.current_x
            dy = self.target_y - self.current_y
            dyaw = math.atan2(dy, dx)
            for i in range(1, 6):
                self.get_logger().warn(f"⚠️  导航到计算点失败，尝试直接走向目标， 第{5-i}/5个点")
                new_point_x = self.target_x - i /5.0 * dx
                new_point_y = self.target_y - i /5.0 * dy
                nav_ok = self._goto_point_with_early_stop(
                new_point_x, new_point_y, dyaw,
                frame_id="map", wait_timeout=5.0, stop_distance=0.5
            )
                if not nav_ok:
                    self.get_logger().info(f"→ 面向目标 (x: {new_point_x}, y: {new_point_y}) not ok, 走一个更近的点")
                    # continue
                    if i==5:
                        nav_ok = self._goto_point_with_early_stop(self.current_x, self.current_y, dyaw,frame_id="map", wait_timeout=5.0, stop_distance=0.5)
                else:
                    self.get_logger().info(f"→ 面向目标 (x: {new_point_x}, y: {new_point_y}) ok, 就去这个点了")
                    break

        # dualmap已通过在线检测找到目标，现在到达位置后拍照确认
        self.get_logger().info(f"✓ 到达目标位置，拍照确认...")
        # 等待图像更新
        time.sleep(0.5)
        # 拍照保存
        if self.latest_image is not None:
            import cv2
            cv_image = self.latest_image.copy()
            save_path = self._save_rgb_snapshot(cv_image, prefix="success")
            if save_path:
                self.get_logger().info(f"📸 已保存成功图片: {save_path}")

        self.get_logger().info(f"✓ 任务成功：找到了'{self.target_name}'")
        write_log(f"任务成功: {self.target_name}")

        # 停留5秒
        self.get_logger().info("⏸️  停留5秒...")
        time.sleep(5.0)
        self._return_home_and_exit("Task completed successfully")

        # # 返回原点
        # self.get_logger().info("返回原点")
        # self._goto_point(0.0, 0.0, yaw=0.0, frame_id="map", wait_timeout=5.0)

        # self.request_exit("任务完成")

    def _return_home_and_exit(self, reason: str):
        """
        返回起点并退出任务
        """
        self.get_logger().info("===== 返回起点 =====")
        write_log("返回起点 (-0.8, -0.8)")

        return_ok = self._goto_point(-0.8, -0.8, yaw=0.0, frame_id="map", wait_timeout=5.0)

        if return_ok:
            self.get_logger().info("✓ 成功返回起点")
            write_log("成功返回起点")
        else:
            self.get_logger().warn("✗ 返回起点失败")
            write_log("返回起点失败")

        time.sleep(0.5)
        self.request_exit(reason)


    def _handle_specified_room_not_found(self):
        """
        处理用户指定房间但未找到目标的情况

        触发条件:
        - 用户明确指定了房间（如"去卧室找杯子"）
        - 在该房间搜索完成后未找到目标

        处理流程:
        1. 记录失败日志（指定房间未找到）
        2. 直接返回起点（不再搜索其他房间）
        3. 打印醒目的失败信息
        4. 调用 request_exit 结束任务（状态：FAIL）

        设计理由:
        - 用户明确指定了房间，说明有特定意图
        - 不应该擅自搜索其他房间（可能不符合用户期望）
        - 快速失败，让用户知道结果并重新决策
        """
        room_name = self.searched_rooms[0] if self.searched_rooms else "指定房间"
        self.get_logger().warn(f"在 {room_name} 中未找到目标 {self.target_name}")
        write_log(f"任务失败: 在 {room_name} 中未找到 {self.target_name}")

        # 返回原点
        self.get_logger().info("返回原点")
        self._goto_point(0.0, 0.0, yaw=0.0, frame_id="map", wait_timeout=5.0)

        print(f"\n{'='*50}")
        print(f"任务失败: 在 {room_name} 中未找到 {self.target_name}")
        print(f"{'='*50}\n")

        self.request_exit(f"FAIL: 在 {room_name} 中未找到目标")

    def _handle_all_rooms_not_found(self):
        """
        处理所有房间都搜索完但未找到目标的情况

        触发条件:
        - 用户未指定房间（如"找杯子"）
        - LLM推理或默认顺序遍历所有房间
        - 所有房间都搜索完成，均未找到目标

        处理流程:
        1. 记录失败日志（包含已搜索的房间列表）
        2. 返回起点
        3. 打印详细的失败信息（包括已搜索房间）
        4. 调用 request_exit 结束任务（状态：FAIL）

        设计理由:
        - 已经尽力搜索所有可能的位置
        - 提供详细的搜索记录帮助分析
        - 明确告知用户目标不在已知房间中
        """
        self.get_logger().warn(f"所有房间遍历完成，未找到目标 {self.target_name}")
        searched_list = ", ".join(self.searched_rooms)
        write_log(f"任务失败: 遍历房间 [{searched_list}] 未找到 {self.target_name}")

        # 返回原点
        self.get_logger().info("返回原点")
        self._goto_point(-1.0, 1.0, yaw=0.0, frame_id="map", wait_timeout=5.0)

        print(f"\n{'='*50}")
        print(f"任务失败: 遍历所有房间未找到 {self.target_name}")
        print(f"已搜索房间: {searched_list}")
        print(f"{'='*50}\n")

        self.request_exit(f"FAIL: 遍历所有房间未找到目标")

    def remap_target_callback(self, msg):
        """
        重写父类的 remap_target_callback，添加 target_found 标志

        父类行为:
        - 解析 dualmap 返回的目标位置
        - 更新 self.target_x 和 self.target_y

        子类扩展:
        - 在收到目标位置后，立即设置 target_found=True
        - 这个标志用于中断当前的导航和房间遍历
        - 使系统能够立即响应 dualmap 的检索结果

        触发时机:
        - dualmap 在点云/语义地图中找到目标物体
        - 通过 /dualmap/search_result topic 发布结果
        - 本方法作为订阅回调被调用
        """
        super().remap_target_callback(msg)

        with self._lock:
            if self.target_x is not None and self.target_y is not None:
                self.target_found = True
                self.get_logger().info("收到 dualmap 响应，设置 target_found=True")


def main():
    cfg_path = os.path.join(PROJECT_ROOT, "config/query/query_task_3pp.yaml")

    print("=" * 60)
    print("Mission 5: 高阶记忆寻物任务")
    print("=" * 60)
    # ========== 初始化 ROS ==========
    print("\n正在初始化 ROS 和加载模型...")
    rclpy.init()
    node = Mission5Subscriber(cfg_path)

    print("✓ ROS 和模型加载完成")

    # ========== 输入0: 搜索模式 ==========
    print("\n【输入0】请选择搜索模式：")
    print("  1 - single_room（指定房间搜索，需要房间+相关物体信息）")
    print("  2 - multi_room（全屋优先级遍历搜索）")
    mode_choice = input("请输入模式编号 (1 or 2): ").strip()

    if mode_choice == "1":
        search_mode = "single_room"
    elif mode_choice == "2":
        search_mode = "multi_room"
    else:
        print("[WARN] 无效输入，使用默认模式: multi_room")
        search_mode = "multi_room"

    print(f"✓ 已选择模式: {search_mode}")

    # ========== 输入1: 记忆上下文 ==========
    print("\n【输入1】请输入记忆上下文：")
    print("示例：爸爸日常会在书桌上工作，他的黑色书包放在书桌中央；")
    print("      卧室床头柜的台灯是妈妈睡前阅读用的；")
    print("      妈妈经常在厨房准备美食，她早晨用过煤气灶旁的水壶烧开水")
    memory_context = input("记忆上下文: ").strip()

    if not memory_context:
        print("[ERROR] 记忆上下文不能为空！")
        return

    print(f"✓ 已接收记忆上下文（{len(memory_context)}字）")

    # ========== 输入2: 任务指令 ==========
    print("\n【输入2】请输入任务指令：")
    print("示例：请找到妈妈早上在厨房用过的烧水壶")
    query_text = input("任务指令: ").strip()

    if not query_text:
        print("[ERROR] 任务指令不能为空！")
        return

    print(f"✓ 已接收任务指令")

    # ========== LLM 推理 ==========
    print("\n" + "=" * 60)
    print("正在调用 LLM 推理...")
    print("=" * 60)

    write_log(f"开始任务 [模式:{search_mode}] 指令='{query_text}'")

    qwen_result = parse_memory_with_qwen(cfg_path, memory_context, query_text, search_mode)
    target_object = qwen_result["target_object"]
    target_room = qwen_result["target_room"]
    related_object = qwen_result["related_object"]
    room_priority = qwen_result["room_priority"]

    # ========== 显示推理结果 ==========
    print("\n" + "=" * 60)
    print("LLM 推理结果：")
    print("=" * 60)
    print(f"  目标物体:   {target_object}")
    print(f"  目标房间:   {target_room if target_room != 'None' else '未指定'}")
    print(f"  相关物体:   {related_object if related_object != 'None' else '无'}")
    print(f"  房间优先级: {' -> '.join(room_priority)}")
    print("=" * 60)


    # ========== 设置任务参数 ==========
    node.search_mode = search_mode
    node.target_name = target_object
    node.room = target_room if target_room != "None" else None
    node.related_object_name = related_object if related_object != "None" else None
    node.room_priority = room_priority

    # ✅ 如果 LLM 给了房间：先把房间信息准备好（room_bbox / is_room_ready）
    if node.room is not None:
        node._room_cb(node.room)

    # ✅ 再触发任务

    # 触发任务
    node._task_event.set()

    # ========== 启动执行器 ==========
    print("\n" + "=" * 60)
    print("开始执行任务...")
    print("=" * 60)

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
