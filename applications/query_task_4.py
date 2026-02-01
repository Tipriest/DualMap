"""
Task 4 V5: 独立的地毯避障版本

完全独立实现，不依赖其他task版本
核心逻辑：
1. 解析指令 → 确定房间访问顺序（用户指定/LLM推理/默认）
2. **配置地毯避障** - 任务开始时检测地毯并写入keepout配置
3. 遍历房间：导航到锚点 → 360度旋转扫描 → 等待dualmap响应
4. 如果收到目标位置 → 导航过去 → 保存图片 → 返回原点(0,-0.3)
5. 如果遍历完所有房间仍未找到 → 返回原点

注意：
- 所有导航自动避开地毯（通过Nav2 costmap filters）
- 地毯bbox不膨胀，使用原始检测结果
- keepout配置路径：/root/nav2_ws/src/nav2_bringup/params/keepout_bboxes.yaml
- 完全独立，不依赖query_task_3ppv0
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
from std_msgs.msg import Bool, String

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # applications/
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)  # DualMap/
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.path.append(os.path.join(PROJECT_ROOT, "applications/utils"))

from applications.query_task_subscriber import TaskSubscriber, write_log

LOG_FILE = "nav_result_task4.txt"


def parse_command_with_qwen(cfg_path: str, query_text: str) -> dict:
    """
    使用Qwen LLM解析用户指令，提取目标物体、房间和房间优先级
    
    返回示例:
    {
        "target_object": "toy horse",
        "target_room": "bedroom",  # 或 "None"
        "room_priority": ["bedroom", "childroom", "livingroom"]
    }
    """
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # LLM prompt - 包含28个物体映射
    prompt = f"""你是一个智能家居助手，负责解析用户的找物指令。

**任务**: 从用户指令中提取以下信息：
1. target_object: 要找的目标物体（英文）
2. target_room: 指定的房间（英文，如果没有明确指定则为"None"）
3. room_priority: 根据常识推理最可能找到该物体的房间顺序（列表）

**物体名称映射**（中文→英文）：
- 背包→backpack, 画框/相框→picture frame, 篮球→basketball, 碗→bowl
- 香蕉→banana, 苹果→apple, 木马→toy horse, 椅子→chair, 沙发→couch
- 绿植→green plant, 床→bed, 桌子→table, 电视→tv, 笔记本电脑→laptop
- 微波炉→microwave, 柜子→cabinet, 毛绒玩具→soft toy, 地毯→carpet
- 台灯→table lamp, 床头柜→nightstand, 帐篷→tent, 积木→building blocks
- 书架→bookshelf, 燃气灶→gas stove, 锅→pot, 水壶→kettle
- 菜篮→food basket, 水龙头→faucet

**房间名称**（只有这4个房间）：
- 客厅→livingroom, 卧室→bedroom, 儿童房→childroom, 厨房→kitchen

**重要规则**：
- '床上'、'桌上'、'柜子里'等位置描述不是房间，必须返回"None"
- 只有明确提到"客厅"、"卧室"、"儿童房"、"厨房"才算指定房间
- 例如："床上的木马" → target_room="None"（床上不是房间）
- 例如："卧室床上的木马" → target_room="bedroom"（有明确房间）

**输出格式**（JSON）：
{{
    "target_object": "物体英文名",
    "target_room": "房间英文名或None",
    "room_priority": ["房间1", "房间2", "房间3", "房间4"]
}}

**用户指令**: "{query_text}"

请严格按照JSON格式输出，不要有任何额外文字。"""

    # 调用Qwen API
    api_url = cfg.get("llm_api_url", "http://localhost:8000/v1/chat/completions")
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "model": cfg.get("llm_model", "Qwen2.5-7B-Instruct"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 256
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        result_text = response.json()["choices"][0]["message"]["content"].strip()
        
        # 提取JSON（去除可能的markdown标记）
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(result_text)
        
        # 验证结果
        if "target_object" not in result:
            result["target_object"] = "unknown"
        if "target_room" not in result:
            result["target_room"] = "None"
        if "room_priority" not in result or not result["room_priority"]:
            result["room_priority"] = ["livingroom", "bedroom", "childroom", "kitchen"]
        
        return result
        
    except Exception as e:
        print(f"LLM解析失败: {e}")
        # 返回默认值
        return {
            "target_object": query_text,
            "target_room": "None",
            "room_priority": ["livingroom", "bedroom", "childroom", "kitchen"]
        }


class Task4Subscriber(TaskSubscriber):
    """
    Task 4 独立订阅器：遍历房间搜索 + 地毯避障
    不依赖其他task实现，完全独立
    """

    def __init__(self, cfg_path: str):
        super().__init__(cfg_path)
        
        # 地毯避障相关
        self.bbox_ready_pub = self.create_publisher(Bool, '/bbox_config_ready', 10)
        self.carpet_configured = False
        
        # 任务状态
        self.target_found = False
        self.searched_rooms = []
        self.rooms_to_visit = []
        self.user_specified_room = False
        self.room_priority = []
        
        # 停止父类的worker，启动自己的
        self._shutdown_event.set()
        if hasattr(self, '_worker') and self._worker:
            self._worker.join(timeout=1.0)
        
        self._shutdown_event.clear()
        self._worker = threading.Thread(target=self._task_worker_task4, daemon=True)
        self._worker.start()
        
        self.get_logger().info("Task4Subscriber initialized (独立版本，带地毯避障)")

    def _task_worker_task4(self):
        """
        Task 4 独立工作流程
        """
        while not self._shutdown_event.is_set():
            self._task_event.wait(timeout=0.2)
            if self._shutdown_event.is_set():
                break
            if not self._task_event.is_set():
                continue

            with self._lock:
                target_name = self.target_name
                room = getattr(self, 'room', None)

            self._task_event.clear()

            if not target_name:
                self.get_logger().warn("Worker triggered but target_name is empty.")
                continue

            try:
                # ========== 步骤0: 配置地毯避障 ==========
                if not self.carpet_configured:
                    self.get_logger().info("=" * 50)
                    self.get_logger().info("🚫 开始配置地毯语义避障...")
                    self.get_logger().info("=" * 50)
                    self._setup_carpet_avoidance()
                    self.carpet_configured = True
                
                # ========== 重置状态 ==========
                self.target_found = False
                self.searched_rooms = []
                with self._lock:
                    self.target_x = None
                    self.target_y = None
                
                # ========== 确定要访问的房间列表 ==========
                if room and room != "None":
                    self.rooms_to_visit = [room]
                    self.user_specified_room = True
                    self.get_logger().info(f"[模式1] 用户指定房间: {room}")
                elif self.room_priority and len(self.room_priority) > 0:
                    self.rooms_to_visit = self.room_priority
                    self.user_specified_room = False
                    self.get_logger().info(f"[模式2] LLM推理房间顺序: {self.room_priority}")
                else:
                    self.rooms_to_visit = ["livingroom", "bedroom", "childroom", "kitchen"]
                    self.user_specified_room = False
                    self.get_logger().info("[模式3] 使用默认房间顺序")

                write_log(f"开始遍历房间寻找 {target_name}（带地毯避障），顺序: {self.rooms_to_visit}", filename=LOG_FILE)

                # ========== 遍历每个房间进行搜索 ==========
                for room_name in self.rooms_to_visit:
                    if self.target_found:
                        self.get_logger().info("✓ 目标已找到，停止遍历")
                        break

                    self.get_logger().info(f"===== 开始访问房间: {room_name} =====")
                    self.searched_rooms.append(room_name)
                    
                    # 发布房间名称
                    self._room_cb(room_name)
                    time.sleep(0.5)

                    # 等待房间信息
                    if not self._wait_for_room_info(room_name, timeout=5.0):
                        self.get_logger().warn(f"⚠ 房间 {room_name} 信息超时，跳过")
                        continue

                    with self._lock:
                        room_anchor = self.room_anchor_pt
                        room_bbox = self.room_bbox

                    if room_anchor is None:
                        self.get_logger().warn(f"⚠ 房间 {room_name} 没有锚点，跳过")
                        continue

                    anchor_x, anchor_y = room_anchor
                    
                    # 发布检索请求
                    self.publish_related_bbox(anchor_x, anchor_y, 2.0, 2.0, target_name)

                    # 导航到房间锚点（自动避开地毯）
                    self.get_logger().info(f"→ 导航到 {room_name} 锚点: ({anchor_x:.2f}, {anchor_y:.2f}) 🚫避开地毯")
                    ok = self._goto_point(anchor_x, anchor_y, yaw=0.0, frame_id="map", wait_timeout=5.0)

                    if not ok:
                        self.get_logger().warn(f"✗ 导航到 {room_name} 失败，继续下一个房间")
                        continue

                    # 到达锚点，开始360度旋转扫描
                    self.get_logger().info(f"↻ 到达 {room_name}，开始360度旋转扫描")
                    target_found = self._spin_and_wait_for_target(3)

                    if target_found:
                        self.target_found = True
                        self.get_logger().info("✓ 旋转过程中找到目标")
                        break

                    # 额外等待0.5秒看是否收到响应
                    time.sleep(0.5)
                    with self._lock:
                        if self.target_x is not None and self.target_y is not None:
                            self.target_found = True
                            self.get_logger().info("✓ 收到 dualmap 最终响应")
                            break

                # ========== 任务结果处理 ==========
                if self.target_found:
                    self._handle_target_found()
                else:
                    if self.user_specified_room:
                        self._handle_specified_room_not_found()
                    else:
                        self._handle_all_rooms_not_found()

            except Exception as e:
                self.get_logger().error(f"Worker exception: {repr(e)}")
                import traceback
                traceback.print_exc()
                self._return_home_and_exit(f"Exception: {e}")

    def _wait_for_room_info(self, room_name: str, timeout: float = 5.0) -> bool:
        """等待房间信息就绪"""
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                if self.is_room_ready and self.room_bbox is not None and self.room_anchor_pt is not None:
                    return True
            time.sleep(0.05)
        return False

    def _spin_and_wait_for_target(self, rounds: int) -> bool:
        """
        原地旋转指定圈数，每圈8个方向，等待dualmap返回目标位置
        
        Args:
            rounds: 旋转圈数
            
        Returns:
            bool: True=找到目标, False=未找到
        """
        cx, cy = self.current_x, self.current_y
        
        for round_idx in range(rounds):
            self.get_logger().info(f"↻ 第 {round_idx+1}/{rounds} 圈")
            
            for direction_idx in range(8):
                # 检查是否已找到
                with self._lock:
                    if self.target_x is not None and self.target_y is not None:
                        self.get_logger().info("✓ 找到目标，停止旋转")
                        return True
                
                # 旋转到下一个方向
                yaw = math.pi / 4 * direction_idx
                self._goto_point(cx, cy, yaw=yaw, frame_id="map", wait_timeout=5.0)
                time.sleep(0.3)
        
        return False

    def _setup_carpet_avoidance(self):
        """
        配置地毯避障：
        1. 使用YOLO检测地毯位置
        2. 将地毯bbox写入keepout_bboxes.yaml
        3. 发布ready信号触发bbox_mask_server加载配置
        """
        self.get_logger().info("🔍 使用YOLO检测地毯位置...")
        
        # 使用继承的query_callback方法检测地毯
        carpet_corners = self.query_callback("carpet")
        
        if carpet_corners is None:
            self.get_logger().warn("⚠️  未找到地毯(carpet)，跳过避障设置")
            self.get_logger().warn("⚠️  导航过程将不会避开地毯")
            return
        
        self.get_logger().info(f"✓ 找到地毯，corners: {carpet_corners}")
        
        # 直接使用地毯的原始边界（不膨胀）
        corners_list = [
            [float(carpet_corners[0][0]), float(carpet_corners[0][1])],
            [float(carpet_corners[1][0]), float(carpet_corners[1][1])],
            [float(carpet_corners[2][0]), float(carpet_corners[2][1])],
            [float(carpet_corners[3][0]), float(carpet_corners[3][1])]
        ]
        
        # 构建bbox配置（Nav2 costmap filter格式）
        bbox_config = {
            'bboxes': [
                {
                    'frame': 'map',
                    'corners': corners_list
                }
            ],
            'resolution': 0.01,
            'topic': '/keepout_filter_mask',
            'target_frame': 'map'
        }
        
        # 写入配置文件到Nav2工作空间
        keepout_yaml_path = "/root/nav2_ws/src/nav2_bringup/params/keepout_bboxes.yaml"
        
        try:
            with open(keepout_yaml_path, 'w') as f:
                yaml.dump(bbox_config, f, default_flow_style=False)
            
            self.get_logger().info(f"✓ 已写入地毯避障配置: {keepout_yaml_path}")
            time.sleep(0.5)
            
            # 发布ready信号
            ready_msg = Bool()
            ready_msg.data = True
            self.bbox_ready_pub.publish(ready_msg)
            self.get_logger().info("✓ 已发布配置就绪信号 (/bbox_config_ready)")
            
            time.sleep(1.0)
            self.get_logger().info("=" * 50)
            self.get_logger().info("✓ 地毯避障配置完成！")
            self.get_logger().info("✓ 所有后续导航将通过Nav2 costmap filters自动避开地毯区域")
            self.get_logger().info("=" * 50)
            
        except Exception as e:
            self.get_logger().error(f"❌ 写入地毯避障配置失败: {e}")
            import traceback
            traceback.print_exc()

    def _handle_target_found(self):
        """处理找到目标的情况"""
        with self._lock:
            target_x = self.target_x
            target_y = self.target_y

        self.get_logger().info("=" * 50)
        self.get_logger().info(f"✓✓✓ 找到目标！位置: ({target_x:.2f}, {target_y:.2f})")
        self.get_logger().info("=" * 50)
        
        # 导航到目标位置（自动避开地毯）
        self.get_logger().info(f"→ 导航到目标位置... 🚫避开地毯")
        ok = self._goto_point(target_x, target_y, yaw=0.0, frame_id="map", wait_timeout=5.0)
        
        if not ok:
            self.get_logger().error("✗ 导航到目标失败")
            write_log(f"导航到目标失败: ({target_x:.2f}, {target_y:.2f})", filename=LOG_FILE)
            self._return_home_and_exit("Navigation to target failed")
            return
        
        # 到达目标位置
        self.get_logger().info("✓ 到达目标位置")
        time.sleep(0.5)
        
        # 保存成功图片
        if self.latest_image is not None:
            import cv2
            cv_image = self.latest_image.copy()
            save_path = self._save_rgb_snapshot(cv_image, prefix="success")
            if save_path:
                self.get_logger().info(f"📸 已保存成功图片: {save_path}")
        else:
            self.get_logger().warn("⚠️  无法保存照片: latest_image 为空")
        
        # 暂停5秒
        self.get_logger().info("⏸  暂停5秒...")
        time.sleep(5.0)
        
        # 返回起点
        room_info = f"于房间 {self.searched_rooms[-1]}" if self.searched_rooms else ""
        write_log(f"任务成功：找到 {self.target_name} {room_info}", filename=LOG_FILE)
        self._return_home_and_exit("Task completed successfully")

    def _handle_specified_room_not_found(self):
        """处理在指定房间未找到的情况"""
        room_name = self.rooms_to_visit[0]
        self.get_logger().error(f"✗ 在指定房间 {room_name} 未找到 {self.target_name}")
        write_log(f"失败：在房间 {room_name} 未找到 {self.target_name}", filename=LOG_FILE)
        self._return_home_and_exit(f"Not found in {room_name}")

    def _handle_all_rooms_not_found(self):
        """处理遍历所有房间仍未找到的情况"""
        self.get_logger().error(f"✗ 遍历所有房间后未找到 {self.target_name}")
        self.get_logger().error(f"   已访问房间: {', '.join(self.searched_rooms)}")
        write_log(f"失败：遍历所有房间 {self.searched_rooms} 后未找到 {self.target_name}", filename=LOG_FILE)
        self._return_home_and_exit("Not found in all rooms")

    def _return_home_and_exit(self, reason: str):
        """返回起点(0,-0.3)并退出任务"""
        self.get_logger().info("===== 返回起点 (0, -0.3) =====")
        write_log("返回起点 (0, -0.3)", filename=LOG_FILE)
        
        return_ok = self._goto_point(0.0, -0.8, yaw=0.0, frame_id="map", wait_timeout=5.0)
        
        if return_ok:
            self.get_logger().info("✓ 成功返回起点")
            write_log("成功返回起点", filename=LOG_FILE)
        else:
            self.get_logger().warn("✗ 返回起点失败")
            write_log("返回起点失败", filename=LOG_FILE)
        
        time.sleep(0.5)
        self.request_exit(reason)


def main():
    cfg_path = os.path.join(PROJECT_ROOT, "config/query/query_task_3pp.yaml")

    print("=" * 50)
    print("正在初始化 Task 4 (独立版本 + 地毯避障)...")
    print("=" * 50)
    
    rclpy.init()
    node = Task4Subscriber(cfg_path)

    print("=" * 50)
    print("ROS 和模型加载完成，等待用户输入...")
    print("=" * 50)

    query_text = input("请输入指令：")
    write_log(f"开始Task4任务: 指令='{query_text}'", filename=LOG_FILE)
    
    # 使用Qwen解析指令
    qwen_result = parse_command_with_qwen(cfg_path, query_text)
    target_object = qwen_result["target_object"]
    target_room = qwen_result["target_room"]
    room_priority = qwen_result["room_priority"]

    print("=" * 50)
    print(f"目标物品: {target_object}")
    print(f"目标房间: {target_room if target_room != 'None' else '未指定'}")
    print(f"房间优先级: {' -> '.join(room_priority) if room_priority else '默认顺序'}")
    print("=" * 50)

    # 设置任务参数
    node.target_name = target_object
    node.room = target_room if target_room != "None" else None
    node.room_priority = room_priority

    # 触发任务开始
    node._task_event.set()

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

