"""
完成task 01：进入指定区域，不调用建图，只从config文件中读取房间区域内预设的目标点
#TODO: 要不要返回原点
"""

import os

os.environ["DISPLAY"] = ""

import sys
import time
import math
import yaml
import threading
import json
import requests

import numpy as np


import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose

from action_msgs.msg import GoalStatus

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # applications/
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)  # DualMap/
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import datetime

LOG_FILE = "nav_result.txt"


def write_log(message):
    """
    记录带有时间戳的日志到文件
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    log_entry = f"[{timestamp}] {message}\n"
    # 输出到控制台方便调试，也可注释掉
    print(log_entry.strip())
    with open(LOG_FILE, "a") as f:
        f.write(log_entry)


STATUS_NAME = {
    GoalStatus.STATUS_UNKNOWN: "UNKNOWN",
    GoalStatus.STATUS_ACCEPTED: "ACCEPTED",
    GoalStatus.STATUS_EXECUTING: "EXECUTING",
    GoalStatus.STATUS_CANCELING: "CANCELING",
    GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
    GoalStatus.STATUS_CANCELED: "CANCELED",
    GoalStatus.STATUS_ABORTED: "ABORTED",
}


def yaw_to_quaternion(yaw: float):
    qz = math.sin(yaw * 0.5)
    qw = math.cos(yaw * 0.5)
    return 0.0, 0.0, qz, qw


def quat_to_yaw(qx, qy, qz, qw) -> float:
    # yaw (Z) from quaternion
    return math.atan2(
        2.0 * (qw * qz + qx * qy), qw * qw + qx * qx - qy * qy - qz * qz
    )


class TaskSubscriber(Node):
    def __init__(self, cfg_path: str):
        super().__init__("nav2_goal_sender")

        with open(cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # ====== callback group：允许并发回调（配合 MultiThreadedExecutor）======
        self._cbg = ReentrantCallbackGroup()

        # ====== Nav2 Action Client ======
        self._action_name = "/navigate_to_pose"
        self._client = ActionClient(
            self, NavigateToPose, self._action_name, callback_group=self._cbg
        )

        # ====== 从配置文件读取房间边界 ======
        self.room_anchors = {}
        if "room_anchors" in self.cfg:
            self.room_anchors = self.cfg["room_anchors"]
            self.get_logger().info(
                f"Loaded room anchors from config: {list(self.room_anchors.keys())}"
            )

        self.get_logger().info(
            "TaskSubscriber initialized. Waiting for topics..."
        )

    def _region_cb(self, region: str):
        """回调函数：接收目标区域，匹配与指令最相似的区域"""
        region_anchor = self.room_anchors.get(region, None)

        print("Sending goal to ", region_anchor)

        ok = self._goto_point(
            region_anchor[0],
            region_anchor[1],
            yaw=0.0,
            frame_id="map",
            wait_timeout=5.0,
        )

        print("Goal result: ", ok)
        write_log(f"Goal result:  {ok}")

        if ok:
            write_log(f"Enter region! {region}")

        ok = self._goto_point(0, 0, yaw = 0.0, frame_id="map", wait_timeout=5.0)
        if ok:
            write_log(f"Retuen to origin after entering region {region}")

    # ====================== Nav2 Action：异步 + Event 等待 ======================

    def _goto_point(
        self, x: float, y: float, yaw: float, frame_id: str, wait_timeout: float
    ) -> bool:
        """
        发送 NavigateToPose 并等待 result（在 worker 线程里 wait，不阻塞 ROS 回调线程）。
        FLAG: 不再做任何提前截断/cancel逻辑，完全交给 Nav2 自己的容忍度。
        """
        if not self._client.wait_for_server(timeout_sec=wait_timeout):
            self.get_logger().error(
                f"NavigateToPose server not available: '{self._action_name}' (waited {wait_timeout}s)"
            )
            return False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = frame_id
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        qx, qy, qz, qw = yaw_to_quaternion(float(yaw))
        goal_msg.pose.pose.orientation.x = qx
        goal_msg.pose.pose.orientation.y = qy
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw

        self.get_logger().info(
            f"Send goal: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f} ({frame_id})"
        )

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
                self.get_logger().info("Goal accepted. Waiting result...")

                rfut = gh.get_result_async()

                def _on_result(rf):
                    try:
                        res = rf.result()
                        result_holder["status"] = (
                            None if res is None else int(res.status)
                        )
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
        ok = done_evt.wait(timeout=nav_timeout)
        if not ok:
            self.get_logger().error(f"Navigation timeout after {nav_timeout}s.")
            return False

        if result_holder["accepted"] is not True:
            self.get_logger().error("Goal rejected / no goal_handle.")
            return False

        status = result_holder["status"]
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Navigation SUCCEEDED.")
            return True

        self.get_logger().warn(
            f"Navigation finished with status={status} ({STATUS_NAME.get(status, '???')})"
        )
        return False


def parse_command_with_qwen(cfg_path: str, user_query: str):
    """
    使用 Qwen 官方 API 解析用户指令，提取导航参数。

    Args:
        user_query: 用户输入的指令文本

    Returns:
        包含 target_room, target_object, related_object, avoid_object 的字典
    """
    # 从cfg读取API密钥和基础URL
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    api_key = cfg["api_key"]
    base_url = os.getenv(
        "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    if not api_key:
        raise ValueError("请设置 QWEN_API_KEY 环境变量")

    # 构建与OpenAI兼容的请求格式
    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    #  prompt 确保解析格式一致
    prompt = f"""请从用户指令：{user_query}，中提取出目标区域，只可能是以下三种区域之一：
            1. 主卧床头桌区域 -> 返回 "bedroom"
            2. 儿童看护区域 -> 返回 "childroom"
            3. 煤气看护区域 -> 返回 "kitchen"

            要求：
            1. 只返回上述三个字符串之一，不要添加任何其他文本
            2. 即使指令中有其他对象，也只关注区域
            3. 如果无法确定，返回最可能的区域
            """

    # 构建请求体
    payload = {
        "model": "qwen-max",  # MODEL qwen-turbo, qwen-plus
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,  # 低温度以保证输出稳定性
        "top_p": 0.8,
        "stream": False,
        "max_tokens": 50,
    }

    response = requests.post(
        url, headers=headers, data=json.dumps(payload), timeout=30
    )
    response.raise_for_status()  # 检查HTTP错误

    result = response.json()

    print("Qwen API Response:", result)
    if "choices" in result and len(result["choices"]) > 0:
        content = result["choices"][0]["message"]["content"].strip().lower()

        # 验证返回值
        valid_responses = {"bedroom", "childroom", "kitchen"}
        if content in valid_responses:
            return content
        else:
            # 尝试从内容中提取关键词
            if "主卧" in user_query or "床头" in user_query:
                return "bedroom"
            elif "儿童" in user_query:
                return "childroom"
            elif "煤气" in user_query or "厨房" in user_query:
                return "kitchen"
            else:
                raise ValueError(f"无法解析指令: {user_query}")
    else:
        raise ValueError("API响应格式异常")


def main():
    # 从配置读取
    cfg_path = os.path.join(PROJECT_ROOT, "config/query/query_task_1.yaml")
    # cfg_path = "/home/tang123/Documents/DualMap/config/query/query_task_1.yaml"

    # 读取指令
    query_text = input("请输入指令：")
    write_log(f"Start: Command received - '{query_text}'")
    region_result = parse_command_with_qwen(cfg_path, query_text)

    print("***********************************************")
    print(f"Goal Region: {region_result}")
    print("***********************************************")

    # 初始化ROS和Node
    rclpy.init()
    node = TaskSubscriber(cfg_path)

    # 等待初始化
    time.sleep(1)

    node._region_cb(region_result)

    print("************************************************")

    # 启动执行器
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
