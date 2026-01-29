"""
Explore++: 按顺序遍历所有房间并返回起点
- 遍历顺序：客厅 (livingroom) → 厨房 (kitchen) → 卧室 (bedroom) → 儿童房 (childroom)
- 不涉及任何物体检索逻辑
- 纯粹的底盘控制，用于探索环境
- 统一使用 query_task_3pp.yaml 配置
- 策略：遍历完成后保存地图，停掉 SLAM，发静态 TF，然后返回起点
"""

import os
os.environ["DISPLAY"] = ""

import sys
import time
import yaml
import threading
import subprocess
import numpy as np

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry
from action_msgs.msg import GoalStatus
from tf2_ros import StaticTransformBroadcaster

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

LOG_FILE = "nav_result_explore.txt"


def write_log(msg: str):
    """写入日志到文件"""
    with open(LOG_FILE, "a") as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {msg}\n")


class StaticTFPublisher(Node):
    """
    静态 TF 发布器：持续发布 map -> odom 的变换
    用于在停掉 SLAM 后维持地图坐标系
    
    关键：使用定时器持续发布，而不是只发一次
    """
    
    def __init__(self):
        super().__init__('static_tf_publisher')
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.is_publishing = False
        self.timer = None
        self.get_logger().info("静态 TF 发布器已初始化")
    
    def start_publishing(self):
        """
        开始持续发布 map -> odom 的静态变换
        使用定时器每 0.1 秒发布一次
        """
        if self.is_publishing:
            self.get_logger().warn("TF 已在发布中")
            return
        
        # 立即发布一次
        self._publish_transform()
        
        # 启动定时器，持续发布（10Hz）
        self.timer = self.create_timer(0.1, self._publish_transform)
        self.is_publishing = True
        self.get_logger().info("✅ 开始持续发布静态 TF: map -> odom (10Hz)")
    
    def stop_publishing(self):
        """停止发布 TF"""
        if self.timer:
            self.timer.cancel()
            self.is_publishing = False
            self.get_logger().info("⏹️  停止发布静态 TF")
    
    def _publish_transform(self):
        """
        发布 map -> odom 的静态变换（单位变换）
        """
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'
        
        # 单位变换（无平移、无旋转）
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        
        self.tf_broadcaster.sendTransform(t)


class ExploreSubscriber(Node):
    """
    Explore++ 专用订阅器：轻量级导航节点
    
    功能：
    - 只负责底盘导航，不做物体检索
    - 不加载 CLIP 模型，减少启动时间
    - 按固定顺序访问房间锚点
    - 全部完成后返回起点
    """
    
    def __init__(self, cfg_path: str):
        super().__init__("explore_navigator")
        
        # 加载配置
        with open(cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        
        # 房间锚点
        self.room_anchors = self.cfg.get("room_anchors", {})
        self.get_logger().info(f"已加载房间锚点: {list(self.room_anchors.keys())}")
        
        # Callback group（允许并发回调）
        self._cbg = ReentrantCallbackGroup()
        
        # 订阅 odometry 获取当前位置
        self.position_sub = self.create_subscription(
            Odometry, "/odom", self._position_callback, 10, callback_group=self._cbg
        )
        
        # 订阅地图（用于路径规划）
        self.map_sub = self.create_subscription(
            OccupancyGrid, "/map", self._map_callback, 10, callback_group=self._cbg
        )
        
        # Nav2 Action Client
        self._action_name = "/navigate_to_pose"
        self._nav_client = ActionClient(
            self, NavigateToPose, self._action_name, callback_group=self._cbg
        )
        
        # 当前位姿
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        
        # 地图数据
        self.map_data = None
        self.map_info = None
        self.map_received = False
        
        # 任务状态
        self.exploration_complete = False
        self.map_server_process = None
        
        self.get_logger().info("✅ Explore 导航节点初始化完成（轻量级，无 CLIP）")
    
    def _position_callback(self, msg: Odometry):
        """更新当前位置"""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        # 计算 yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    def _map_callback(self, msg: OccupancyGrid):
        """处理地图数据"""
        try:
            width = msg.info.width
            height = msg.info.height
            resolution = msg.info.resolution
            origin_x = msg.info.origin.position.x
            origin_y = msg.info.origin.position.y
            
            data = np.array(msg.data, dtype=np.int8).reshape((height, width))
            
            self.map_data = data
            self.map_info = {
                "width": width,
                "height": height,
                "resolution": resolution,
                "origin_x": origin_x,
                "origin_y": origin_y,
            }
            self.map_received = True
        except Exception as e:
            self.get_logger().error(f"地图回调错误: {e}")
    
    def _goto_point(self, x: float, y: float, yaw: float = 0.0, 
                    frame_id: str = "map", wait_timeout: float = 5.0) -> bool:
        """
        导航到指定位置
        
        Args:
            x, y: 目标位置（世界坐标）
            yaw: 目标朝向（弧度）
            frame_id: 参考坐标系
            wait_timeout: 等待 action server 的超时时间
            
        Returns:
            bool: 导航是否成功
        """
        # 等待 action server
        if not self._nav_client.wait_for_server(timeout_sec=wait_timeout):
            self.get_logger().error("❌ Nav2 action server 未响应")
            return False
        
        # 创建目标
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = frame_id
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0
        
        # 设置朝向（四元数）
        goal_msg.pose.pose.orientation.x = 0.0
        goal_msg.pose.pose.orientation.y = 0.0
        goal_msg.pose.pose.orientation.z = np.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = np.cos(yaw / 2.0)
        
        # 发送目标
        self.get_logger().info(f"→ 导航到 ({x:.2f}, {y:.2f}), yaw={yaw:.2f}")
        send_goal_future = self._nav_client.send_goal_async(goal_msg)
        
        # 等待目标被接受
        rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=5.0)
        
        if not send_goal_future.done():
            self.get_logger().error("❌ 发送导航目标超时")
            return False
        
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("❌ 导航目标被拒绝")
            return False
        
        # 等待导航完成
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        
        result = result_future.result()
        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("✅ 导航成功")
            return True
        else:
            self.get_logger().warn(f"⚠️  导航失败，状态码: {result.status}")
            return False
    
    def _spin_360(self, x: float, y: float, num_directions: int = 8) -> bool:
        """
        在指定位置原地旋转 360 度
        
        用于探索任务，让 SLAM 从多个角度观察环境，提高建图质量
        
        Args:
            x, y: 旋转中心位置（保持不变）
            num_directions: 旋转方向数（默认 8 个方向，即 45° 间隔）
            
        Returns:
            bool: 旋转是否完全成功
        """
        import math
        
        success_count = 0
        
        for i in range(num_directions):
            # 计算当前方向的 yaw 角度
            # 0, π/4, π/2, 3π/4, π, 5π/4, 3π/2, 7π/4
            # (即 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
            yaw = (2 * math.pi / num_directions) * i
            
            # 在当前位置旋转到指定角度
            ok = self._goto_point(x, y, yaw=yaw, frame_id="map", wait_timeout=5.0)
            
            if ok:
                success_count += 1
            else:
                self.get_logger().warn(f"⚠️  旋转到第 {i+1}/{num_directions} 个方向失败")
            
            # 短暂停留，让 SLAM 处理数据
            time.sleep(0.3)
        
        # 如果至少完成了一半的旋转，就认为成功
        return success_count >= (num_directions / 2)
        
    def explore_rooms(self, room_sequence: list, return_point: tuple = (-0.3, 0.0), 
                      tf_publisher=None):
        """
        按顺序遍历所有房间并返回起点
        
        执行流程：
        1. 依次导航到每个房间的锚点
        2. 无论成功与否，继续下一个房间
        3. 遍历完成后原地停下，保存地图
        4. 启动静态 TF 发布（提前启动，确保无缝切换）
        5. 停掉 SLAM Toolbox
        6. 启动 map_server 发布静态地图
        7. 返回起点（略微往后，避免 odom 漂移）
        8. 设置完成标志
        
        Args:
            room_sequence: 房间遍历顺序列表，如 ["livingroom", "kitchen", "bedroom", "childroom"]
            return_point: 返回目标点，默认 (-0.3, 0.0)，即起点往后 30cm
            tf_publisher: 静态 TF 发布器节点（用于提前启动 TF）
        
        注意：
        - 不涉及任何检索逻辑，只是简单的导航
        - 用于底盘控制和环境探索
        - 每个房间导航失败不会中断整个流程
        - 关键：儿童房回起点时 odom 会飘，所以要先保存地图再返回
        - 关键：静态 TF 必须在停止 SLAM 前启动，避免 TF 断层
        """
        total_rooms = len(room_sequence)
        success_count = 0
        failed_rooms = []
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("🚀 开始 Explore++ 任务：遍历所有房间")
        self.get_logger().info(f"📋 遍历顺序: {' → '.join(room_sequence)}")
        self.get_logger().info("=" * 60)
        write_log(f"开始 Explore++ 任务，遍历顺序: {room_sequence}")
        
        # ===== 步骤1: 依次遍历每个房间 =====
        for idx, room_name in enumerate(room_sequence, 1):
            self.get_logger().info("")
            self.get_logger().info(f"{'=' * 60}")
            self.get_logger().info(f"📍 [{idx}/{total_rooms}] 前往房间: {room_name}")
            self.get_logger().info(f"{'=' * 60}")
            
            # 获取房间锚点
            anchor_pt = self.room_anchors.get(room_name, None)
            if anchor_pt is None:
                self.get_logger().error(f"❌ 房间 {room_name} 没有配置锚点，跳过")
                write_log(f"跳过房间 {room_name}: 缺少锚点配置")
                failed_rooms.append(room_name)
                continue
            
            anchor_x, anchor_y = anchor_pt
            self.get_logger().info(f"🎯 目标位置: ({anchor_x:.2f}, {anchor_y:.2f})")
            
            # 导航到房间锚点
            # wait_timeout=5.0 表示等待 Nav2 action server 响应的超时时间
            # 实际导航时间由 Nav2 的 nav_timeout (默认300s) 控制
            ok = self._goto_point(
                anchor_x, anchor_y, yaw=0.0, frame_id="map", wait_timeout=5.0
            )
            
            if ok:
                self.get_logger().info(f"✅ 成功到达 {room_name}")
                write_log(f"成功到达房间: {room_name} at ({anchor_x:.2f}, {anchor_y:.2f})")
                
                # 在房间原地旋转一圈，帮助 SLAM 建图
                self.get_logger().info(f"↻ 在 {room_name} 原地旋转 360° (8个方向)")
                spin_ok = self._spin_360(anchor_x, anchor_y)
                
                if spin_ok:
                    self.get_logger().info(f"✅ {room_name} 旋转完成")
                    write_log(f"{room_name} 旋转完成")
                else:
                    self.get_logger().warn(f"⚠️  {room_name} 旋转未完全完成")
                    write_log(f"{room_name} 旋转部分失败")
                
                success_count += 1
            else:
                self.get_logger().warn(f"⚠️  导航到 {room_name} 失败，继续下一个房间")
                write_log(f"导航失败: {room_name}")
                failed_rooms.append(room_name)
            
            # 短暂停留，让传感器稳定
            time.sleep(0.5)
        
        # ===== 步骤2: 原地停下，准备保存地图 =====
        self.get_logger().info("")
        self.get_logger().info("=" * 60)
        self.get_logger().info("💾 所有房间遍历完成，准备保存地图")
        self.get_logger().info("=" * 60)
        write_log("所有房间遍历完成，原地停下准备保存地图")
        
        # 确保机器人完全停止
        time.sleep(1.0)
        
        # ===== 步骤3: 保存地图为 yaml 和 pgm =====
        self.get_logger().info("📤 调用 map_saver 保存地图...")
        self.get_logger().info("⏳ 保存地图可能需要 10-30 秒，请耐心等待...")
        write_log("保存地图为 yaml 和 pgm 格式")
        
        save_map_success = self._save_slam_map()
        
        if save_map_success:
            self.get_logger().info("✅ 地图保存成功")
            write_log("地图保存成功")
        else:
            self.get_logger().warn("⚠️  地图保存失败或超时")
            write_log("地图保存失败")
        
        time.sleep(0.5)  # 减少等待时间
        
        # ===== 步骤4: 启动静态 TF 发布（关键：在停止 SLAM 前启动）=====
        self.get_logger().info("=" * 60)
        self.get_logger().info("📡 启动静态 TF 发布器 (map->odom)")
        self.get_logger().info("⚠️  关键步骤：必须在停止 SLAM 前启动，确保 TF 无缝切换")
        self.get_logger().info("⚠️  使用定时器持续发布 (10Hz)，确保 Nav2 始终能查到 TF")
        self.get_logger().info("=" * 60)
        write_log("启动静态 TF 发布器（持续发布模式）")
        
        if tf_publisher is not None:
            tf_publisher.start_publishing()  # 启动持续发布
            self.get_logger().info("✅ 静态 TF 已开始持续发布 (10Hz)")
            write_log("静态 TF 已启动并持续发布")
            # 等待 TF 发布稳定
            time.sleep(1.0)
        else:
            self.get_logger().warn("⚠️  未提供 tf_publisher，无法启动静态 TF")
            write_log("静态 TF 启动失败：未提供 tf_publisher")
        
        # ===== 步骤5: 停掉 SLAM Toolbox =====
        self.get_logger().info("🛑 停止 SLAM Toolbox...")
        write_log("停止 SLAM Toolbox")
        
        stop_slam_success = self._stop_slam_toolbox()
        
        if stop_slam_success:
            self.get_logger().info("✅ SLAM Toolbox 已停止")
            write_log("SLAM Toolbox 已停止")
        else:
            self.get_logger().warn("⚠️  停止 SLAM Toolbox 失败")
            write_log("停止 SLAM Toolbox 失败")
        
        time.sleep(0.5)  # 减少等待时间
        
        # ===== 步骤6: 启动 map_server 发布静态地图 =====
        self.get_logger().info("🗺️  启动 map_server 加载静态地图...")
        write_log("启动 map_server")
        
        map_server_success = self._start_map_server()
        
        if map_server_success:
            self.get_logger().info("✅ map_server 已启动并加载地图")
            write_log("map_server 已启动")
        else:
            self.get_logger().warn("⚠️  启动 map_server 失败")
            write_log("启动 map_server 失败")
        
        time.sleep(1.0)  # 等待 map_server 完全启动（减少等待时间）
        
        # ===== 步骤7: 返回起点（往后偏移，避免 odom 漂移）=====
        self.get_logger().info("")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"🏠 返回起点（目标: {return_point}）")
        self.get_logger().info("⚠️  使用静态 TF + 静态地图导航（SLAM 已停止）")
        self.get_logger().info("=" * 60)
        write_log(f"返回起点: {return_point}")
        
        return_x, return_y = return_point
        # 增加超时时间，避免长距离导航超时
        return_ok = self._goto_point(return_x, return_y, yaw=0.0, frame_id="map", wait_timeout=10.0)
        
        if return_ok:
            self.get_logger().info(f"✅ 成功返回起点 {return_point}")
            write_log(f"成功返回起点 {return_point}")
        else:
            self.get_logger().warn("⚠️  返回起点失败")
            write_log("返回起点失败")
        
        # ===== 步骤8: 总结并设置完成标志 =====
        self.get_logger().info("")
        self.get_logger().info("=" * 60)
        self.get_logger().info("📊 Explore++ 任务完成统计")
        self.get_logger().info(f"   总房间数: {total_rooms}")
        self.get_logger().info(f"   成功到达: {success_count}")
        self.get_logger().info(f"   失败/跳过: {len(failed_rooms)}")
        if failed_rooms:
            self.get_logger().info(f"   失败房间: {', '.join(failed_rooms)}")
        self.get_logger().info(f"   地图保存: {'✅ 成功' if save_map_success else '⚠️  失败'}")
        self.get_logger().info(f"   静态TF启动: {'✅ 成功' if (tf_publisher is not None) else '⚠️  失败'}")
        self.get_logger().info(f"   停止SLAM: {'✅ 成功' if stop_slam_success else '⚠️  失败'}")
        self.get_logger().info(f"   启动map_server: {'✅ 成功' if map_server_success else '⚠️  失败'}")
        self.get_logger().info(f"   返回起点: {'✅ 成功' if return_ok else '⚠️  失败'}")
        self.get_logger().info("=" * 60)
        
        write_log(f"Explore++ 完成: 成功 {success_count}/{total_rooms} 个房间")
        if failed_rooms:
            write_log(f"失败房间: {failed_rooms}")
        
        # 标记完成
        self.exploration_complete = True
        time.sleep(0.5)  # 给导航一点时间完全停止
        
        # 记录最终状态
        final_status = "SUCCESS" if (success_count == total_rooms and return_ok) else "PARTIAL"
        self.get_logger().info(f"🏁 Explore++ 任务结束 - {final_status}")
        write_log(f"Explore++ 任务结束 - {final_status}")
    
    def _save_slam_map(self) -> bool:
        """
        使用 map_saver 保存地图为 yaml 和 pgm 格式
        供后续 SLAM Toolbox 读取
        
        Returns:
            bool: 保存是否成功
        """
        try:
            # 使用 ros2 run nav2_map_server map_saver_cli 保存地图
            # 保存为 explore_map.yaml 和 explore_map.pgm
            # -f 参数指定文件名（不含扩展名）
            # --ros-args -p save_map_timeout:=10000 设置超时时间
            cmd = [
                "ros2", "run", "nav2_map_server", "map_saver_cli",
                "-f", "explore_map",
                "--ros-args", "-p", "save_map_timeout:=10000.0"
            ]
            
            self.get_logger().info(f"执行命令: {' '.join(cmd)}")
            self.get_logger().info("💾 保存地图为: explore_map.yaml 和 explore_map.pgm")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30.0  # 增加超时时间，大地图可能需要更久
            )
            
            if result.returncode == 0:
                self.get_logger().info(f"保存地图输出: {result.stdout}")
                self.get_logger().info("✅ 地图文件已保存:")
                self.get_logger().info("   - explore_map.yaml (地图配置)")
                self.get_logger().info("   - explore_map.pgm (占用栅格图)")
                return True
            else:
                self.get_logger().error(f"保存地图失败: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.get_logger().error("保存地图超时")
            return False
        except Exception as e:
            self.get_logger().error(f"保存地图异常: {e}")
            return False
    
    def _stop_slam_toolbox(self) -> bool:
        """
        停止 SLAM Toolbox 节点
        
        使用 ros2 lifecycle 命令将 SLAM Toolbox 设置为 inactive 状态
        或者直接 kill 掉 slam_toolbox 进程
        
        Returns:
            bool: 停止是否成功
        """
        try:
            # 方法1: 使用 lifecycle 管理（如果 SLAM Toolbox 支持）
            # cmd = [
            #     "ros2", "lifecycle", "set",
            #     "/slam_toolbox",
            #     "inactive"
            # ]
            
            # 方法2: 直接 kill 进程（更直接）
            cmd = ["pkill", "-f", "slam_toolbox"]
            
            self.get_logger().info(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5.0
            )
            
            # pkill 返回 0 表示找到并 kill 了进程
            if result.returncode == 0:
                self.get_logger().info("SLAM Toolbox 进程已终止")
                return True
            else:
                self.get_logger().warn(f"未找到 SLAM Toolbox 进程或已停止: {result.stderr}")
                return True  # 也算成功，因为目标是让它停止
                
        except subprocess.TimeoutExpired:
            self.get_logger().error("停止 SLAM Toolbox 超时")
            return False
        except Exception as e:
            self.get_logger().error(f"停止 SLAM Toolbox 异常: {e}")
            return False
    
    def _start_map_server(self) -> bool:
        """
        启动 map_server 加载并发布静态地图
        在后台运行，持续发布 /map topic
        
        Returns:
            bool: 启动是否成功
        """
        try:
            # 使用 ros2 run nav2_map_server map_server 启动地图服务器
            # --ros-args -p yaml_filename:=explore_map.yaml 指定地图文件
            # 需要在后台运行，所以使用 Popen
            cmd = [
                "ros2", "run", "nav2_map_server", "map_server",
                "--ros-args",
                "-p", "yaml_filename:=explore_map.yaml",
                "-p", "use_sim_time:=false"
            ]
            
            self.get_logger().info(f"执行命令: {' '.join(cmd)}")
            self.get_logger().info("🗺️  启动 map_server (后台进程)")
            
            # 在后台启动 map_server
            self.map_server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 等待一小段时间，检查进程是否正常启动
            time.sleep(1.0)
            
            if self.map_server_process.poll() is None:
                # 进程仍在运行，说明启动成功
                self.get_logger().info(f"✅ map_server 进程已启动 (PID: {self.map_server_process.pid})")
                self.get_logger().info("   - 持续发布 /map topic")
                self.get_logger().info("   - 加载地图: explore_map.yaml")
                return True
            else:
                # 进程已退出，说明启动失败
                stdout, stderr = self.map_server_process.communicate()
                self.get_logger().error(f"map_server 启动失败:")
                self.get_logger().error(f"stdout: {stdout}")
                self.get_logger().error(f"stderr: {stderr}")
                return False
                
        except Exception as e:
            self.get_logger().error(f"启动 map_server 异常: {e}")
            return False
    
    def cleanup_map_server(self):
        """
        清理 map_server 进程（如果需要停止）
        """
        if self.map_server_process and self.map_server_process.poll() is None:
            self.get_logger().info("停止 map_server 进程...")
            self.map_server_process.terminate()
            try:
                self.map_server_process.wait(timeout=5.0)
                self.get_logger().info("✅ map_server 已停止")
            except subprocess.TimeoutExpired:
                self.get_logger().warn("map_server 未响应，强制终止")
                self.map_server_process.kill()
                self.map_server_process.wait()


def main():
    """
    主函数：启动 Explore++ 任务
    
    流程：
    1. 加载配置文件
    2. 初始化 ROS 节点
    3. 启动后台执行器
    4. 执行房间遍历（客厅→厨房→卧室→儿童房）
    5. 保存地图
    6. 启动静态 TF 发布器（关键：在停止 SLAM 前）
    7. 停止 SLAM
    8. 启动 map_server 发布静态地图
    9. 返回起点
    10. 清理资源
    
    关键改进：
    - 静态 TF 在停止 SLAM **之前**启动，确保 TF 无缝切换
    - 避免 Nav2 出现 "Extrapolation Error" 错误
    """
    cfg_path = os.path.join(PROJECT_ROOT, "config/query/query_task_3pp.yaml")
    
    # 定义房间遍历顺序（修改后）
    # 顺序：客厅 (livingroom) → 厨房 (kitchen) → 卧室 (bedroom) → 儿童房 (childroom)
    room_sequence = ["livingroom", "kitchen", "bedroom", "childroom"]
    
    # 返回点：起点往后 30cm，避免 odom 漂移
    return_point = (-0.6, 0.0)
     
    # 确认开始
    input("按 Enter 键开始探索任务...")
    print("")
    
    write_log(f"启动 Explore++ 任务: {room_sequence}")
    
    # ===== 初始化 ROS =====
    rclpy.init()
    
    # 创建主探索节点
    explore_node = ExploreSubscriber(cfg_path)
    
    # 创建静态 TF 发布器节点（先创建，后面会用到）
    tf_publisher = StaticTFPublisher()

    # ===== 启动 executor 在后台线程 =====
    # 使用多线程执行器允许并发处理回调
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(explore_node)
    executor.add_node(tf_publisher)  # 添加 TF 发布器节点
    
    # 在单独的线程中运行 executor，避免阻塞主线程
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    # 等待初始化完成
    time.sleep(1.0)

    # ===== 执行探索任务 =====
    # 注意：tf_publisher 会在 explore_rooms 内部使用，在停止 SLAM 前启动
    explore_node.explore_rooms(room_sequence, return_point, tf_publisher)

    # ===== 等待任务完成（房间遍历完成标志）=====
    # 注意：explore_rooms 会在内部完成所有操作：
    #   1. 遍历房间
    #   2. 保存地图
    #   3. 启动静态 TF（关键：在停止 SLAM 前）
    #   4. 停止 SLAM
    #   5. 启动 map_server
    #   6. 返回起点
    while not explore_node.exploration_complete:
        time.sleep(0.5)

    # 再等一会儿确保所有导航完全停止
    time.sleep(1.0)
    
    # ===== 清理资源 =====
    print("")
    print("=" * 60)
    print("🧹 清理 ROS 资源...")
    
    # 注意：map_server 和静态 TF 发布器会持续运行
    # 这是必要的，因为 Nav2 需要持续接收地图和 TF
    print("⚠️  以下服务将继续运行:")
    print("   - map_server (发布 /map topic)")
    print("   - 静态 TF 发布器 (map->odom 变换)")
    print("   如需停止请手动 Ctrl+C 或 pkill")
    
    executor.shutdown()
    explore_node.destroy_node()
    # tf_publisher.destroy_node()  # 保持运行
    # explore_node.cleanup_map_server()  # 保持运行
    rclpy.shutdown()

    print("✅ Explore++ 任务执行完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
