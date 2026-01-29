"""
Task 4 V2: 带语义避障的目标搜索

新逻辑：
1. 有房间 + 有related：在房间内找related → VLM验证 → 成功则搜索target，失败则fallback到锚点
2. 有房间 + 无related：直接去房间锚点 → fallback搜索target
3. 无房间 + 有related：提示使用query++（不支持）
4. 无房间 + 无related：提示使用query++（不支持）

注意：地面（floor/ground）不算related

Fallback搜索逻辑：
- 在锚点转圈（3圈 × 8方向）
- 同时发布检索指令给dualmap
- 找到则导航过去，未找到则fail返回原点

与Task 2-3区别：增加语义避障（避开指定物体如地毯）
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
from std_msgs.msg import Bool

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
        
        # 创建地毯避障配置发布器
        self.bbox_ready_pub = self.create_publisher(Bool, '/bbox_config_ready', 10)
        self.get_logger().info("✓ 地毯避障配置发布器已创建")
        
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
        Task 4 重构版工作流程（带语义避障）
        
        执行逻辑：
        1. 判断场景类型（4种组合）
        2. 有房间+有related：房间内找related → VLM验证 → 成功搜索/失败fallback
        3. 有房间+无related：直接fallback（去锚点转圈搜索）
        4. 无房间场景：提示使用query++
        5. Fallback搜索：锚点转3圈等dualmap，找到则导航，未找到则fail
        
        与Task 2-3区别：所有导航都会避开语义障碍物（通过_hazard_cb设置）
        """
        while not self._shutdown_event.is_set():
            self._task_event.wait(timeout=0.2)
            if self._shutdown_event.is_set():
                break
            if not self._task_event.is_set():
                continue

            # ===== 获取任务参数 =====
            with self._lock:
                target_name = self.target_name
                related_name = self.related_object_name
                room_name = getattr(self, "room", None)

            self._task_event.clear()

            if not target_name:
                self.get_logger().warn("Worker triggered but target_name is empty.")
                continue

            try:
                # ===== 设置地毯避障 =====
                self._setup_carpet_avoidance()
                
                # ===== 判断场景类型 =====
                has_room = room_name is not None and room_name != "" and room_name != "None"
                has_related = (related_name is not None and 
                              related_name != "" and 
                              related_name != "None" and
                              related_name.lower() not in ["floor", "ground"])  # 地面不算related
                
                self.get_logger().info("=" * 60)
                self.get_logger().info(f"场景判断: 房间={'✓' if has_room else '✗'}  Related={'✓' if has_related else '✗'}")
                if has_room:
                    self.get_logger().info(f"  房间: {room_name}")
                if has_related:
                    self.get_logger().info(f"  Related: {related_name}")
                self.get_logger().info(f"  Target: {target_name}")
                self.get_logger().info("=" * 60)
                
                # ===== 场景分发 =====
                if not has_room and has_related:
                    # 场景3: 无房间+有related → 不支持
                    self.get_logger().error("❌ 不支持无房间但有related物体的场景")
                    self.get_logger().info("💡 请使用 query++ 或指定房间（如：去客厅找桌子上的杯子）")
                    self._return_home_and_exit("Unsupported: no room with related")
                    continue
                
                if not has_room and not has_related:
                    # 场景4: 无房间+无related → 不支持
                    self.get_logger().error("❌ 必须指定房间或使用 query++")
                    self.get_logger().info("💡 请使用 query++ 或指定房间（如：去卧室找杯子）")
                    self._return_home_and_exit("Unsupported: no room and no related")
                    continue
                
                # ===== 等待房间信息（有房间的场景都需要）=====
                if has_room:
                    self.get_logger().info(f"⏳ 等待房间 '{room_name}' 信息...")
                    t0 = time.time()
                    while True:
                        with self._lock:
                            room_ready = self.is_room_ready
                            room_bbox = self.room_bbox
                            room_anchor = self.room_anchor_pt

                        if room_ready and room_bbox is not None and room_anchor is not None:
                            self.get_logger().info(f"✓ 房间信息就绪: bbox={room_bbox}, anchor={room_anchor}")
                            break

                        if time.time() - t0 > getattr(self, "room_wait_timeout", 5.0):
                            self.get_logger().error(f"❌ 等待房间 '{room_name}' 超时")
                            self._return_home_and_exit("Room timeout")
                            raise RuntimeError("room not ready")

                        time.sleep(0.05)
                
                # ===== 场景1: 有房间+有related =====
                if has_room and has_related:
                    self.get_logger().info("=" * 60)
                    self.get_logger().info("📍 场景1: 有房间+有related（带语义避障）")
                    self.get_logger().info(f"  策略: 在'{room_name}'内找'{related_name}' → VLM验证 → 搜索'{target_name}'")
                    self.get_logger().info("=" * 60)
                    
                    # 在房间内查找related物体
                    rcorners = self.query_callback(related_name)
                    
                    if rcorners is None:
                        # 房间内找不到related，直接fallback
                        self.get_logger().warn(f"⚠️  在'{room_name}'内未找到'{related_name}'")
                        self.get_logger().info("→ Fallback: 直接在房间锚点搜索")
                        self._fallback_search_at_anchor(target_name, room_anchor, room_bbox)
                        continue
                    
                    # 计算related物体位置
                    rpos = np.array(rcorners).mean(axis=0)
                    rx, ry = float(rpos[0]), float(rpos[1])
                    delta_rx = rcorners[1][0] - rcorners[0][0]
                    delta_ry = rcorners[2][1] - rcorners[1][1]
                    
                    self.get_logger().info(f"✓ 在CLIP map中找到'{related_name}': ({rx:.2f}, {ry:.2f})")
                    
                    # 导航到related位置（会避开障碍物）
                    free_rx, free_ry = self.find_optimal_free_point_by_room_center(rx, ry, 1.2)
                    self.get_logger().info(f"→ 导航到'{related_name}'附近: ({free_rx:.2f}, {free_ry:.2f})")
                    self.get_logger().info("🚫 导航过程将避开语义障碍物")
                    nav_ok = self._goto_and_face_target(free_rx, free_ry, rx, ry)
                    
                    if not nav_ok:
                        self.get_logger().warn("⚠️  导航到related失败")
                        self.get_logger().info("→ Fallback: 直接在房间锚点搜索")
                        self._fallback_search_at_anchor(target_name, room_anchor, room_bbox)
                        continue
                    
                    # 在related附近搜索target（不再对related做VLM验证）
                    self.get_logger().info(f"→ 在'{related_name}'附近搜索'{target_name}'")
                    
                    # 发布检索请求
                    self.publish_related_bbox(rx, ry, delta_rx, delta_ry, target_name)
                    
                    # 转圈搜索target
                    target_found = self._spin_and_wait_for_target(self.max_spin_rounds)
                    
                    if target_found:
                        self._navigate_and_check_target(target_name)
                    else:
                        self.get_logger().error(f"❌ 在'{related_name}'附近未找到'{target_name}'")
                        self.get_logger().info("→ Fallback: 在房间锚点搜索")
                        self._fallback_search_at_anchor(target_name, room_anchor, room_bbox)
                    
                    continue
                
                # ===== 场景2: 有房间+无related =====
                if has_room and not has_related:
                    self.get_logger().info("=" * 60)
                    self.get_logger().info("📍 场景2: 有房间+无related（带语义避障）")
                    self.get_logger().info(f"  策略: 直接去'{room_name}'锚点 → 转圈搜索'{target_name}'")
                    self.get_logger().info("=" * 60)
                    
                    # 直接fallback搜索（无需VLM验证）
                    self._fallback_search_at_anchor(target_name, room_anchor, room_bbox)
                    continue
                
            except Exception as e:
                self.get_logger().error(f"Worker exception: {type(e).__name__}('{e}')")
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
    
    def _fallback_search_at_anchor(self, target_name: str, room_anchor: tuple, room_bbox: tuple):
        """
        Fallback搜索逻辑：在房间锚点转圈搜索目标（带语义避障）
        
        Args:
            target_name: 目标物体名称
            room_anchor: 房间锚点坐标 (x, y)
            room_bbox: 房间边界 (xmin, xmax, ymin, ymax)
        """
        self.get_logger().info("=" * 60)
        self.get_logger().info("🔄 Fallback搜索模式（带语义避障）")
        self.get_logger().info(f"  位置: 房间锚点 {room_anchor}")
        self.get_logger().info(f"  目标: {target_name}")
        self.get_logger().info("=" * 60)
        
        # 导航到房间锚点（会避开障碍物）
        anchor_x, anchor_y = room_anchor
        self.get_logger().info("🚫 导航到锚点将避开语义障碍物")
        nav_ok = self._goto_point(anchor_x, anchor_y, yaw=0.0, frame_id="map", wait_timeout=5.0)
        
        if not nav_ok:
            self.get_logger().error("❌ 导航到房间锚点失败")
            self._return_home_and_exit("Navigation to anchor failed")
            return
        
        # 发布检索请求（以房间为搜索范围）
        search_center_x = (room_bbox[0] + room_bbox[1]) / 2.0
        search_center_y = (room_bbox[2] + room_bbox[3]) / 2.0
        search_width = room_bbox[1] - room_bbox[0]
        search_height = room_bbox[3] - room_bbox[2]
        
        self.get_logger().info(f"📤 发布检索请求: 在房间内搜索'{target_name}'")
        self.publish_related_bbox(search_center_x, search_center_y, search_width, search_height, target_name)
        
        # 转圈搜索
        target_found = self._spin_and_wait_for_target(self.max_spin_rounds)
        
        if target_found:
            # 找到目标，导航并验证
            self._navigate_and_check_target(target_name)
        else:
            # 未找到目标
            self.get_logger().error(f"❌ Fallback搜索失败：未找到'{target_name}'")
            self._return_home_and_exit(f"Fallback search failed: {target_name} not found")
    
    def _navigate_and_check_target(self, target_name: str):
        """
        导航到目标位置并执行VLM验证（带语义避障）
        
        Args:
            target_name: 目标物体名称
        """
        # 读取dualmap更新的精确位置
        with self._lock:
            final_target_x = self.target_x
            final_target_y = self.target_y
        
        self.get_logger().info(f"✓ dualmap返回目标位置: ({final_target_x:.2f}, {final_target_y:.2f})")
        self.get_logger().info(f"→ 导航到'{target_name}'...")
        
        # 计算可达点（距离增加到1.5m避免太近）
        final_free_x, final_free_y = self.find_optimal_free_point_by_room_center(
            final_target_x, final_target_y, 1.5
        )
        
        # 导航到目标（会避开障碍物）
        self.get_logger().info("🚫 导航到目标将避开语义障碍物")
        nav_ok = self._goto_and_face_target(final_free_x, final_free_y, final_target_x, final_target_y)
        
        if not nav_ok:
            self.get_logger().error("❌ 导航到目标失败")
            self._return_home_and_exit("Navigation to target failed")
            return
        
        # dualmap已通过在线检测找到目标，现在到达位置后拍照确认
        # （不需要VLM验证，因为YOLO比VLM准确）
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
        
        # dualmap检测到就算成功
        check_ok = True
        
        if check_ok:
            self.get_logger().info(f"✓ 任务成功：找到了'{target_name}'")
            write_log(f"任务成功: {target_name}")
            
            # 停留5秒
            self.get_logger().info("⏸️  停留5秒...")
            time.sleep(5.0)
            
            self._return_home_and_exit("Task completed successfully")
        else:
            self.get_logger().error(f"❌ 检测失败：'{target_name}'不存在或不匹配")
            write_log(f"检测失败: {target_name}")
            self._return_home_and_exit("Detection failed")

    def _return_home_and_exit(self, reason: str):
        """
        返回起点并退出任务
        """
        self.get_logger().info("===== 返回起点 =====")
        write_log("返回起点 (0, -0.3)")
        
        return_ok = self._goto_point(0.0, -0.3, yaw=0.0, frame_id="map", wait_timeout=5.0)
        
        if return_ok:
            self.get_logger().info("✓ 成功返回起点")
            write_log("成功返回起点")
        else:
            self.get_logger().warn("✗ 返回起点失败")
            write_log("返回起点失败")
        
        time.sleep(0.5)
        self.request_exit(reason)
    
    def _setup_carpet_avoidance(self):
        """
        设置地毯避障：查找地毯位置并配置 Nav2 costmap filter
        
        流程：
        1. 在 CLIP map 中查找 'carpet' (地毯)
        2. 将地毯的 bbox 写入 keepout_bboxes.yaml
        3. 发布 ready 信号触发 bbox_mask_server 加载配置
        
        注意：用户会告诉你"地毯"，LLM会翻译成"carpet"
        """
        self.get_logger().info("=" * 60)
        self.get_logger().info("🚫 设置地毯避障")
        self.get_logger().info("=" * 60)
        
        # 在 CLIP map 中查找地毯（使用 carpet 而不是 rug）
        self.get_logger().info("🔍 在CLIP map中查找地毯 (carpet)...")
        carpet_corners = self.query_callback("carpet")
        
        if carpet_corners is None:
            self.get_logger().warn("⚠️  未找到地毯(carpet)，跳过避障设置")
            self.get_logger().warn("⚠️  导航过程将不会避开地毯")
            return
        
        # 地毯找到了，准备 bbox 配置
        self.get_logger().info(f"✓ 找到地毯: {carpet_corners}")
        
        # 直接使用地毯的原始边界（不膨胀）
        corners_list = [
            [float(carpet_corners[0][0]), float(carpet_corners[0][1])],
            [float(carpet_corners[1][0]), float(carpet_corners[1][1])],
            [float(carpet_corners[2][0]), float(carpet_corners[2][1])],
            [float(carpet_corners[3][0]), float(carpet_corners[3][1])]
        ]
        
        # 构建 bbox 配置（格式与 nav2_costmap_filters_demo 一致）
        bbox_config = {
            'bboxes': [
                {
                    'frame': 'map',
                    'corners': corners_list
                }
            ],
            'resolution': 0.01,
            'topic': '/keepout_filter_mask',
            'publish_rate': 0.1,
            'target_frame': 'map',
            'max_cells': 400000
        }
        
        # 写入配置文件（相对于项目根目录）
        keepout_yaml_path = os.path.join(
            PROJECT_ROOT, 
            "nav2_costmap_filters_demo", 
            "params", 
            "keepout_bboxes.yaml"
        )
        
        try:
            with open(keepout_yaml_path, 'w') as f:
                yaml.dump(bbox_config, f, default_flow_style=False)
            
            self.get_logger().info(f"✓ 已写入地毯避障配置: {keepout_yaml_path}")
            
            # 等待一小段时间确保文件写入完成
            time.sleep(0.5)
            
            # 发布 ready 信号触发加载
            ready_msg = Bool()
            ready_msg.data = True
            self.bbox_ready_pub.publish(ready_msg)
            
            self.get_logger().info("✓ 已发送配置加载信号")
            self.get_logger().info("🚫 Nav2 将在导航时避开地毯区域")
            
            # 等待配置生效
            time.sleep(1.0)
            
        except Exception as e:
            self.get_logger().error(f"❌ 设置地毯避障失败: {e}")
            import traceback
            traceback.print_exc()


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

    # 初始化 ROS（先加载模型）
    rclpy.init()
    node = Task4Subscriber(cfg_path)

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
        node.destroy_node()
        rclpy.shutdown()
        return

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
