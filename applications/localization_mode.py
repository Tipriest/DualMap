"""
定位模式 (Localization Mode)：在已有地图上进行导航

功能：
- 读取保存的地图文件 (yaml + pgm)
- 持续发布静态 TF (map -> odom)
- 启动 map_server 发布地图
- 替代 SLAM Toolbox，用于纯导航场景

使用场景：
- 探索任务完成后，切换到定位模式进行导航
- 不需要建图，只需要在已知地图上导航
- 减少计算资源占用

启动方式：
    python localization_mode.py [map_file_path]
    
示例：
    python localization_mode.py explore_map
    python localization_mode.py /path/to/my_map
"""

import os
os.environ["DISPLAY"] = ""

import sys
import time
import signal
import subprocess
import numpy as np

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster

# 默认地图文件路径（不含扩展名）
DEFAULT_MAP_PATH = "explore_map"


class LocalizationNode(Node):
    """
    定位模式节点：发布静态 TF 和地图
    """
    
    def __init__(self, map_path: str):
        super().__init__('localization_node')
        
        self.map_path = map_path
        self.map_yaml = f"{map_path}.yaml"
        self.map_pgm = f"{map_path}.pgm"
        
        # 检查地图文件是否存在
        if not os.path.exists(self.map_yaml):
            self.get_logger().error(f"❌ 地图文件不存在: {self.map_yaml}")
            raise FileNotFoundError(f"Map file not found: {self.map_yaml}")
        
        if not os.path.exists(self.map_pgm):
            self.get_logger().error(f"❌ 地图文件不存在: {self.map_pgm}")
            raise FileNotFoundError(f"Map file not found: {self.map_pgm}")
        
        self.get_logger().info(f"✅ 找到地图文件:")
        self.get_logger().info(f"   - {self.map_yaml}")
        self.get_logger().info(f"   - {self.map_pgm}")
        
        # 静态 TF 发布器
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.tf_timer = None
        
        # map_server 进程
        self.map_server_process = None
        
        self.get_logger().info("✅ 定位模式节点初始化完成")
    
    def start_tf_publishing(self):
        """
        开始持续发布 map -> odom 的静态变换
        """
        self.get_logger().info("📡 启动静态 TF 发布器 (map -> odom)")
        
        # 立即发布一次
        self._publish_tf()
        
        # 启动定时器，持续发布（10Hz）
        self.tf_timer = self.create_timer(0.1, self._publish_tf)
        
        self.get_logger().info("✅ 静态 TF 开始持续发布 (10Hz)")
    
    def stop_tf_publishing(self):
        """停止发布 TF"""
        if self.tf_timer:
            self.tf_timer.cancel()
            self.get_logger().info("⏹️  停止发布静态 TF")
    
    def _publish_tf(self):
        """发布 map -> odom 的静态变换（单位变换）"""
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
    
    def start_map_server(self) -> bool:
        """
        启动 map_server 加载并发布地图
        
        Returns:
            bool: 启动是否成功
        """
        try:
            self.get_logger().info("🗺️  启动 map_server...")
            
            # 使用绝对路径
            abs_map_yaml = os.path.abspath(self.map_yaml)
            
            cmd = [
                "ros2", "run", "nav2_map_server", "map_server",
                "--ros-args",
                "-p", f"yaml_filename:={abs_map_yaml}",
                "-p", "use_sim_time:=false"
            ]
            
            self.get_logger().info(f"执行命令: {' '.join(cmd)}")
            
            # 在后台启动 map_server
            self.map_server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 等待一小段时间，检查进程是否正常启动
            time.sleep(1.5)
            
            if self.map_server_process.poll() is None:
                # 进程仍在运行，说明启动成功
                self.get_logger().info(f"✅ map_server 已启动 (PID: {self.map_server_process.pid})")
                self.get_logger().info(f"   - 发布地图: {abs_map_yaml}")
                self.get_logger().info(f"   - 话题: /map")
                return True
            else:
                # 进程已退出，说明启动失败
                stdout, stderr = self.map_server_process.communicate()
                self.get_logger().error(f"❌ map_server 启动失败:")
                self.get_logger().error(f"stdout: {stdout}")
                self.get_logger().error(f"stderr: {stderr}")
                return False
                
        except Exception as e:
            self.get_logger().error(f"❌ 启动 map_server 异常: {e}")
            return False
    
    def stop_map_server(self):
        """停止 map_server 进程"""
        if self.map_server_process and self.map_server_process.poll() is None:
            self.get_logger().info("🛑 停止 map_server...")
            self.map_server_process.terminate()
            try:
                self.map_server_process.wait(timeout=5.0)
                self.get_logger().info("✅ map_server 已停止")
            except subprocess.TimeoutExpired:
                self.get_logger().warn("⚠️  map_server 未响应，强制终止")
                self.map_server_process.kill()
                self.map_server_process.wait()
    
    def cleanup(self):
        """清理资源"""
        self.get_logger().info("🧹 清理资源...")
        self.stop_tf_publishing()
        self.stop_map_server()
        self.get_logger().info("✅ 清理完成")


def signal_handler(signum, frame):
    """处理 Ctrl+C 信号"""
    print("\n收到中断信号，正在退出...")
    sys.exit(0)


def main():
    """
    主函数：启动定位模式
    """
    print("=" * 60)
    print("🧭 定位模式 (Localization Mode)")
    print("=" * 60)
    print("")
    print("功能：")
    print("  - 读取保存的地图文件")
    print("  - 发布静态 TF (map -> odom)")
    print("  - 启动 map_server 发布地图")
    print("  - 供 Nav2 进行导航")
    print("")
    print("=" * 60)
    print("")
    
    # 获取地图路径参数
    if len(sys.argv) > 1:
        map_path = sys.argv[1]
        print(f"📁 使用指定地图: {map_path}")
    else:
        map_path = DEFAULT_MAP_PATH
        print(f"📁 使用默认地图: {map_path}")
    
    print("")
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    
    # 初始化 ROS
    rclpy.init()
    
    try:
        # 创建定位节点
        node = LocalizationNode(map_path)
        
        # 启动静态 TF 发布
        print("=" * 60)
        print("📡 启动静态 TF 发布器...")
        node.start_tf_publishing()
        print("✅ 静态 TF 已启动 (10Hz)")
        print("")
        
        # 启动 map_server
        print("=" * 60)
        print("🗺️  启动 map_server...")
        map_server_ok = node.start_map_server()
        
        if not map_server_ok:
            print("❌ map_server 启动失败，退出")
            node.cleanup()
            rclpy.shutdown()
            return
        
        print("✅ map_server 已启动")
        print("")
        
        # 启动 executor
        print("=" * 60)
        print("🚀 定位模式已就绪")
        print("")
        print("状态：")
        print(f"  ✅ 地图文件: {map_path}.yaml")
        print(f"  ✅ 静态 TF: map -> odom (持续发布)")
        print(f"  ✅ 地图发布: /map topic")
        print("")
        print("可以使用 Nav2 进行导航了！")
        print("")
        print("按 Ctrl+C 停止...")
        print("=" * 60)
        print("")
        
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        
        try:
            # 持续运行
            executor.spin()
        except KeyboardInterrupt:
            print("\n收到中断信号")
        finally:
            # 清理
            print("")
            print("=" * 60)
            print("🧹 正在清理资源...")
            executor.shutdown()
            node.cleanup()
            node.destroy_node()
            rclpy.shutdown()
            print("✅ 定位模式已退出")
            print("=" * 60)
    
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        rclpy.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()
