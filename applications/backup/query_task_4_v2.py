"""
Task 4 V2: 带语义避障的目标搜索
- 解析用户指令提取 target_room, target_object, related_object, avoid_object
- 必须有 room 或 related_object 其中之一
- 避开指定的语义障碍物（如地毯）
- 导航到目标并验证后返回起点
- 统一使用 query_task_3pp.yaml 配置
"""

import os
os.environ["DISPLAY"] = ""

import sys
import time
import yaml
import json
import requests
import threading
import numpy as np

import rclpy
from rclpy.executors import MultiThreadedExecutor

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.path.append(os.path.join(PROJECT_ROOT, "applications/utils"))

from applications.query_task_subscriber import TaskSubscriber, write_log

LOG_FILE = "nav_result_task4.txt"


class Task4Subscriber(TaskSubscriber):
    """
    Task 4 专用订阅器：带语义避障，重写 _task_worker 添加旋转等待逻辑并确保返回起点
    """
    
    def __init__(self, cfg_path: str):
        super().__init__(cfg_path)
        self.max_spin_rounds = 3  # 最多旋转 3 圈
        # 停止父类的 worker
        self._shutdown_event.set()
        if hasattr(self, '_worker') and self._worker:
            self._worker.join(timeout=1.0)
        # 启动新的 worker
        self._shutdown_event.clear()
        self._worker = threading.Thread(target=self._task_worker_v2, daemon=True)
        self._worker.start()
    
    def _task_worker_v2(self):
        """
        Task 4 的工作流程：找到目标物体（带语义避障）
        
        与 Task 2-3 的区别：
        - 需要先调用 _hazard_cb() 设置避障物体（如地毯）
        - Nav2 会在导航时避开这些语义障碍物
        - 其他流程与 Task 2-3 完全相同
        
        执行流程：
        1. 等待房间信息就绪（如果指定了房间）
        2. 在 CLIP map 中查询 target 的初步位置
        3. 导航到 related 物体位置或房间锚点
        4. 旋转 3 圈，每圈 8 个方向，等待 dualmap 返回精确位置
        5. 收到位置后导航到目标并执行 RGB check
        6. 返回起点 (0, 0)
        
        关键点：
        - 语义避障通过 Nav2 的 keepout filter 实现
        - 避障物体需要在导航前通过 _hazard_cb 设置
        """
        while not self._shutdown_event.is_set():
            self._task_event.wait(timeout=0.2)
            if self._shutdown_event.is_set():
                break
            if not self._task_event.is_set():
                continue

            with self._lock:
                target_name = self.target_name
                related_name = self.related_object_name

            self._task_event.clear()

            if not target_name:
                self.get_logger().warn("Worker triggered but target_name is empty.")
                continue

            try:
                # ===== 等待 room_ready =====
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
                            self.get_logger().error("[room] require_room_filter=True but room not ready")
                            raise RuntimeError("room not ready")

                        time.sleep(0.05)

                # ===== 步骤2: 在 CLIP map 中查询 target 的粗略位置 =====
                # 注意：这只是初步定位，用于判断目标是否在地图中
                # 精确位置将在旋转过程中由 dualmap 实时检索提供
                corners = self.query_callback(target_name)
                if corners is None:
                    self.get_logger().error(f"❌ Target '{target_name}' not found in CLIP map.")
                    self._return_home_and_exit("Target not found in map")
                    continue

                target_pos = np.array(corners).mean(axis=0)
                target_x, target_y = float(target_pos[0]), float(target_pos[1])
                free_x, free_y = self.find_optimal_free_point_by_room_center(target_x, target_y, 1.0)

                # 这里先不设置 target_x/target_y，避免误判
                # 将在旋转过程中由 dualmap 更新
                with self._lock:
                    self.target_x = None  # 清空，等待 dualmap 更新
                    self.target_y = None

                self.get_logger().info(f"[query] target '{target_name}' 初步定位 -> ({target_x:.3f}, {target_y:.3f})")
                self.get_logger().info("⚠️  这只是粗略位置，将等待 dualmap 返回精确位置")

                # ===== 步骤3: 导航到观察位置（带语义避障）=====
                # 与 Task 2-3 相同，但导航过程中 Nav2 会避开之前设置的障碍物
                nav_ok = False
                rx, ry, delta_rx, delta_ry = None, None, None, None
                
                if related_name != "None":
                    rcorners = self.query_callback(related_name)
                    if rcorners is None:
                        self.get_logger().warn(f"⚠️  Related object '{related_name}' not found, fallback to room anchor")
                        if self.room_anchor_pt is not None:
                            anchor_x, anchor_y = self.room_anchor_pt
                            self.get_logger().info(f"→ 前往房间锚点（避开障碍物）: ({anchor_x:.3f}, {anchor_y:.3f})")
                            nav_ok = self._goto_point(anchor_x, anchor_y, yaw=0.0, frame_id="map", wait_timeout=5.0)
                    else:
                        rpos = np.array(rcorners).mean(axis=0)
                        rx, ry = float(rpos[0]), float(rpos[1])
                        delta_rx = rcorners[1][0] - rcorners[0][0]
                        delta_ry = rcorners[2][1] - rcorners[1][1]
                        free_rx, free_ry = self.find_optimal_free_point_by_room_center(rx, ry, 1.2)
                        self.get_logger().info(f"[query] related '{related_name}' -> ({rx:.3f}, {ry:.3f})")
                        self.get_logger().info("🚫 导航过程将避开语义障碍物")
                        nav_ok = self._goto_and_face_target(free_rx, free_ry, rx, ry)
                else:
                    if self.room_anchor_pt is not None:
                        anchor_x, anchor_y = self.room_anchor_pt
                        self.get_logger().info(f"→ 无相关物体，前往房间锚点（避开障碍物）: ({anchor_x:.3f}, {anchor_y:.3f})")
                        nav_ok = self._goto_point(anchor_x, anchor_y, yaw=0.0, frame_id="map", wait_timeout=5.0)

                if not nav_ok:
                    self.get_logger().error("❌ 导航到观察位置失败")
                    self._return_home_and_exit("Navigation failed")
                    continue

                # ===== 步骤4: 旋转并等待 dualmap 返回精确位置 =====
                self.get_logger().info(f"===== 开始旋转 {self.max_spin_rounds} 圈等待 dualmap 返回结果 =====")
                
                # 发布检索请求给 dualmap
                if related_name != "None" and rx is not None:
                    self.get_logger().info(f"📤 发布检索请求: 在 '{related_name}' 附近搜索 '{target_name}'")
                    self.publish_related_bbox(rx, ry, delta_rx, delta_ry, self.target_name)
                else:
                    self.get_logger().info(f"📤 发布检索请求: 在整个房间内搜索 '{target_name}'")
                    room_cx = (self.room_bbox[0] + self.room_bbox[1]) / 2.0
                    room_cy = (self.room_bbox[2] + self.room_bbox[3]) / 2.0
                    room_w = self.room_bbox[1] - self.room_bbox[0]
                    room_h = self.room_bbox[3] - self.room_bbox[2]
                    self.publish_related_bbox(room_cx, room_cy, room_w, room_h, self.target_name)
                
                target_found = self._spin_and_wait_for_target(self.max_spin_rounds)

                if target_found:
                    # ===== 步骤5: 找到目标，导航到目标位置 =====
                    self.get_logger().info("✓ dualmap 返回了目标位置")
                    
                    with self._lock:
                        final_target_x = self.target_x
                        final_target_y = self.target_y
                    
                    self.get_logger().info(f"→ 导航到目标精确位置: ({final_target_x:.2f}, {final_target_y:.2f})")
                    
                    final_free_x, final_free_y = self.find_optimal_free_point_by_room_center(
                        final_target_x, final_target_y, 1.0
                    )
                    
                    # 导航到目标（仍然会避开障碍物）
                    self.get_logger().info("🚫 导航到目标过程将避开语义障碍物")
                    nav_to_target_ok = self._goto_and_face_target(
                        final_free_x, final_free_y, final_target_x, final_target_y
                    )
                    
                    if not nav_to_target_ok:
                        self.get_logger().error("❌ 导航到目标位置失败")
                        write_log(f"任务失败: 导航到目标失败")
                        self._return_home_and_exit("Task FAIL - navigation to target failed")
                    else:
                        # ===== 步骤6: 执行 RGB 视觉验证 =====
                        self.get_logger().info("📷 执行 RGB check (VLM 验证)...")
                        is_complete = self.check_task()
                        if is_complete:
                            self.get_logger().info("✅ RGB check 通过，任务成功完成")
                            write_log(f"任务完成: {target_name}")
                            self._return_home_and_exit("Task complete - SUCCESS")
                        else:
                            self.get_logger().warn("⚠️  RGB check 未通过，VLM 未识别到目标")
                            write_log(f"任务失败: RGB check 未通过")
                            self._return_home_and_exit("Task complete - RGB check failed")
                else:
                    # 旋转 3 圈后仍未找到
                    self.get_logger().error(f"✗ 旋转 {self.max_spin_rounds} 圈后仍未找到目标")
                    write_log(f"任务失败: 旋转 {self.max_spin_rounds} 圈未找到 {target_name}")
                    self._return_home_and_exit("Task FAIL - target not found after spinning")

            except Exception as e:
                self.get_logger().error(f"Worker exception: {repr(e)}")
                import traceback
                traceback.print_exc()
                self._return_home_and_exit(f"Exception: {e}")

    def _spin_and_wait_for_target(self, rounds: int) -> bool:
        """
        原地旋转指定圈数，每圈 8 个方向（45度间隔），等待 dualmap 返回目标位置
        
        工作原理：
        1. 在当前位置 (cx, cy) 原地旋转，不移动
        2. 每个方向停留 0.3 秒，给 dualmap 时间处理
        3. dualmap 在后台持续检索，一旦找到会通过 remap_target_callback 更新 target_x/target_y
        4. 每次旋转前检查是否已收到目标位置
        5. 如果 3 圈都没找到，返回 False
        
        Args:
            rounds: 旋转圈数（通常是 3）
            
        Returns:
            bool: True=找到目标（target_x/target_y 被更新），False=未找到
        """
        import math
        
        cx, cy = self.current_x, self.current_y
        angles_per_round = 8  # 每圈 8 个方向 (360° / 8 = 45° 间隔)
        
        for round_num in range(rounds):
            self.get_logger().info(f"↻ 第 {round_num + 1}/{rounds} 圈旋转 (8个方向)")
            
            for i in range(angles_per_round):
                # ===== 每次旋转前先检查是否已收到目标位置 =====
                with self._lock:
                    if self.target_x is not None and self.target_y is not None:
                        self.get_logger().info(f"✓ 在第 {round_num + 1} 圈第 {i + 1} 个方向收到目标位置")
                        return True
                
                # ===== 旋转到下一个角度 =====
                yaw = (math.pi / 4) * i  # 0, 45, 90, ..., 315 度
                self._goto_point(cx, cy, yaw=yaw, frame_id="map", wait_timeout=5.0)
                time.sleep(0.3)  # 给 dualmap 检索的时间
        
        # ===== 所有圈旋转完成，最后检查一次 =====
        with self._lock:
            if self.target_x is not None and self.target_y is not None:
                self.get_logger().info("✓ 旋转完成时收到目标位置")
                return True
        
        self.get_logger().error(f"❌ 旋转 {rounds} 圈后仍未收到 dualmap 的目标位置")
        return False

    def _return_home_and_exit(self, reason: str):
        """
        返回起点并退出任务
        """
        self.get_logger().info("===== 返回起点 =====")
        write_log("返回起点 (0, 0)")
        
        return_ok = self._goto_point(0.0, 0.0, yaw=0.0, frame_id="map", wait_timeout=5.0)
        
        if return_ok:
            self.get_logger().info("✓ 成功返回起点")
            write_log("成功返回起点")
        else:
            self.get_logger().warn("✗ 返回起点失败")
            write_log("返回起点失败")
        
        time.sleep(0.5)
        self.request_exit(reason)


def parse_command_with_qwen(cfg_path: str, user_query: str):
    """
    使用 Qwen API 解析用户指令
    
    Returns:
        dict: {
            "target_room": str,      # 目标房间或"None"
            "target_object": str,     # 目标物体（必须）
            "related_object": str,    # 相关物体或"None"
            "avoid_object": str       # 避障物体或"None"
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
请从以下用户指令中提取关键信息：
用户指令："{user_query}"

提取内容：
1. **目标房间** (target_room): 要去的房间，只能是 bedroom, childroom, livingroom, kitchen 之一，或 "None"
2. **目标物体** (target_object): 需要找到的物品（必须存在）
3. **相关物体** (related_object): 辅助定位的物体，如"床上的被子"中的"床"，或 "None"
4. **避障物体** (avoid_object): 路途中需要避开的物体，如"地毯"，或 "None"

规则：
- target_object 必须存在，不能为空
- 物体名称返回英文类型，如"杯子"→"cup", "地毯"→"rug"
- 房间名称必须是 bedroom/childroom/livingroom/kitchen 之一
- 相关物体是命令中明确提到的辅助物体
- 避障物体是命令中明确要求"避开"、"不要踩"等的物体
- 只返回 JSON 格式

示例：
"去卧室拿床上的被子，不要踩地毯" → {{"target_room": "bedroom", "target_object": "quilt", "related_object": "bed", "avoid_object": "rug"}}
"避开地毯去厨房找杯子" → {{"target_room": "kitchen", "target_object": "cup", "related_object": "None", "avoid_object": "rug"}}

输出格式：
{{
    "target_room": "房间名或None",
    "target_object": "物品英文名",
    "related_object": "物品英文名或None",
    "avoid_object": "物品英文名或None"
}}

现在请生成JSON：
"""

    payload = {
        "model": "qwen-max",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "top_p": 0.8,
        "stream": False,
        "max_tokens": 512,
    }

    try:
        response = requests.post(
            url, headers=headers, data=json.dumps(payload), timeout=30
        )
        response.raise_for_status()
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"].strip()
            
            # 提取 JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                parsed_data = json.loads(json_match.group())
                
                # 确保必需字段存在
                if "target_object" not in parsed_data or not parsed_data["target_object"]:
                    parsed_data["target_object"] = "bottle"
                
                if "target_room" not in parsed_data:
                    parsed_data["target_room"] = "None"
                    
                if "related_object" not in parsed_data:
                    parsed_data["related_object"] = "None"
                
                if "avoid_object" not in parsed_data:
                    parsed_data["avoid_object"] = "None"
                
                return parsed_data
            else:
                raise ValueError("未找到有效JSON")
        else:
            raise ValueError("API响应异常")

    except Exception as e:
        print(f"[ERROR] LLM调用失败: {e}")
        return {
            "target_room": "None",
            "target_object": "bottle",
            "related_object": "None",
            "avoid_object": "None"
        }


def main():
    cfg_path = os.path.join(PROJECT_ROOT, "config/query/query_task_3pp.yaml")

    print("=" * 50)
    print("Task 4: 带语义避障的目标搜索")
    print("=" * 50)

    # 读取指令
    query_text = input("请输入指令（如'避开地毯去卧室拿床上的被子'）：")
    write_log(f"开始任务: 指令='{query_text}'")
    
    # LLM 解析
    qwen_result = parse_command_with_qwen(cfg_path, query_text)
    target_room = qwen_result["target_room"]
    target_object = qwen_result["target_object"]
    related_object = qwen_result["related_object"]
    avoid_object = qwen_result["avoid_object"]

    print("=" * 50)
    print(f"目标房间: {target_room if target_room != 'None' else '未指定'}")
    print(f"目标物体: {target_object}")
    print(f"相关物体: {related_object if related_object != 'None' else '无'}")
    print(f"避障物体: {avoid_object if avoid_object != 'None' else '无'}")
    print("=" * 50)

    # 验证：必须有 room 或 related_object 其中之一
    if target_room == "None" and related_object == "None":
        print("[ERROR] Task 4 要求必须指定房间或相关物体！")
        write_log("FAIL: 缺少房间或相关物体信息")
        return

    # 初始化 ROS
    rclpy.init()
    node = Task4Subscriber(cfg_path)

    # 等待初始化
    time.sleep(1.0)

    # 设置避障物体（如果指定）
    if avoid_object != "None":
        node._hazard_cb(avoid_object)
        print(f"设置避障物体: {avoid_object}")
        write_log(f"避障物体: {avoid_object}")
    
    print("=" * 50)

    # 设置房间（如果指定）
    if target_room != "None":
        node.room = target_room
        node._room_cb(target_room)
        print(f"设置目标房间: {target_room}")
        
        # 等待房间准备
        wait_start = time.time()
        while not node.is_room_ready:
            if time.time() - wait_start > 5.0:
                print("[WARN] 等待房间准备超时")
                break
            time.sleep(0.1)
        print("房间准备完成")

    print("=" * 50)

    # 设置目标信息并触发任务
    node.target_name = target_object
    if related_object == "None":
        node.related_object_name = "None"
        node._target_cb(target_object)
        print(f"开始搜索目标: {target_object}")
    else:
        node.related_object_name = related_object
        node._related_obj_cb(related_object)
        print(f"通过相关物体 {related_object} 搜索目标: {target_object}")

    # 启动执行器
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

    print("=" * 50)
    print("Task 4 执行完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
