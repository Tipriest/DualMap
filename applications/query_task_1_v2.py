"""
Task 1 V2: 进入指定区域并返回起点
- 直接导航到配置文件中预设的房间锚点
- 到达后立即返回起点
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

import rclpy
from rclpy.executors import MultiThreadedExecutor
from action_msgs.msg import GoalStatus

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.path.append(os.path.join(PROJECT_ROOT, "applications/utils"))

from applications.query_task_subscriber import TaskSubscriber, write_log

LOG_FILE = "nav_result_task1.txt"


def parse_command_with_qwen(cfg_path: str, user_query: str):
    """
    使用 Qwen API 解析用户指令，提取目标区域

    Returns:
        str: 房间名称 (bedroom/childroom/kitchen/livingroom)
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
请从用户指令中提取目标区域：
用户指令："{user_query}"

可能的区域只有4个：
- bedroom (卧室/主卧/床头桌区域)
- childroom (儿童房/儿童看护区域)
- kitchen (厨房/煤气看护区域)
- livingroom (客厅/起居室)

要求：
- 只返回上述4个英文单词之一
- 不要添加任何其他文本
- 如果无法确定，根据常识推理最可能的区域

现在请返回区域名称：
"""

    payload = {
        "model": "qwen-max",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "top_p": 0.8,
        "stream": False,
        "max_tokens": 50,
    }

    try:
        response = requests.post(
            url, headers=headers, data=json.dumps(payload), timeout=30
        )
        response.raise_for_status()
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"].strip().lower()

            # 验证返回值
            valid_rooms = {"bedroom", "childroom", "kitchen", "livingroom"}
            if content in valid_rooms:
                return content

            # 尝试从内容中提取关键词
            for room in valid_rooms:
                if room in content:
                    return room

            # 根据用户指令关键词推理
            query_lower = user_query.lower()
            if "主卧" in user_query or "床头" in user_query or "卧室" in user_query:
                return "bedroom"
            elif "儿童" in user_query or "孩子" in user_query:
                return "childroom"
            elif "煤气" in user_query or "厨房" in user_query:
                return "kitchen"
            elif "客厅" in user_query or "起居" in user_query:
                return "livingroom"
            else:
                print(f"[WARN] 无法解析区域，默认返回 livingroom")
                return "livingroom"
        else:
            print("[WARN] API响应异常，默认返回 livingroom")
            return "livingroom"

    except Exception as e:
        print(f"[ERROR] LLM调用失败: {e}")
        return "livingroom"


class Task1Subscriber(TaskSubscriber):
    """
    Task 1 专用订阅器：导航到指定区域并返回
    """

    def __init__(self, cfg_path: str):
        super().__init__(cfg_path)
        self.navigation_complete = False

    def navigate_to_region_and_return(self, region: str):
        """
        导航到指定区域并返回起点

        执行流程：
        1. 从配置文件获取房间锚点坐标
        2. 导航到房间锚点
        3. 无论成功与否，都返回起点 (0, 0)
        4. 设置完成标志，让 main 函数退出

        Args:
            region: 目标区域名称 (bedroom/childroom/kitchen/livingroom)

        注意：
        - Task 1 不涉及物体检索，只是简单的导航任务
        - 不调用 request_exit()，避免 ROS 过早关闭
        - 通过 navigation_complete 标志通知 main 函数
        """
        # ===== 步骤1: 获取区域锚点 =====
        anchor_pt = self.room_anchors.get(region, None)
        if anchor_pt is None:
            self.get_logger().error(f"❌ 区域 {region} 没有配置锚点")
            write_log(f"FAIL: 区域 {region} 配置缺失")
            # 注意：不调用 request_exit，直接标记完成
            self.navigation_complete = True
            return

        anchor_x, anchor_y = anchor_pt

        self.get_logger().info(f"===== Task 1: 前往 {region} =====")
        write_log(f"导航到区域: {region} at ({anchor_x:.2f}, {anchor_y:.2f})")

        # ===== 步骤2: 导航到目标区域 =====
        # wait_timeout=5.0 表示等待 Nav2 action server 响应的超时时间
        # 实际导航时间由 Nav2 的 nav_timeout (默认300s) 控制
        ok = self._goto_point(
            anchor_x, anchor_y, yaw=0.0, frame_id="map", wait_timeout=5.0
        )

        if ok:
            self.get_logger().info(f"✓ 成功到达 {region}")
            write_log(f"成功到达区域: {region}")
        else:
            self.get_logger().warn(f"✗ 导航到 {region} 失败")
            write_log(f"导航失败: {region}")

        # ===== 步骤3: 返回起点 (关键！) =====
        # 无论前面导航成功与否，都必须返回起点
        self.get_logger().info("===== 返回起点 =====")
        write_log("返回起点 (0, 0)")

        return_ok = self._goto_point(0.0, 0.0, yaw=0.0, frame_id="map", wait_timeout=5.0)

        if return_ok:
            self.get_logger().info("✓ 成功返回起点")
            write_log("任务完成: 成功返回起点")
        else:
            self.get_logger().warn("✗ 返回起点失败")
            write_log("任务完成: 返回起点失败")

        # ===== 步骤4: 标记完成 =====
        # 设置标志让 main 函数知道任务已完成
        self.navigation_complete = True
        time.sleep(0.5)  # 给导航一点时间完全停止

        # 记录最终状态
        status = "SUCCESS" if (ok and return_ok) else "PARTIAL"
        self.get_logger().info(f"Task 1 完成 - {status}")
        write_log(f"Task 1 完成 - {status}")


def main():
    cfg_path = os.path.join(PROJECT_ROOT, "config/query/query_task_3pp.yaml")

    print("=" * 50)
    print("Task 1: 进入指定区域并返回")
    print("=" * 50)



    # 初始化 ROS
    rclpy.init()
    node = Task1Subscriber(cfg_path)

    # 读取指令
    query_text = input("请输入指令（如'去卧室'）：")
    write_log(f"开始任务: 指令='{query_text}'")

    # LLM 解析区域
    region = parse_command_with_qwen(cfg_path, query_text)

    print("=" * 50)
    print(f"目标区域: {region}")
    print("=" * 50)


    # 启动 executor 在后台线程
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    # 等待初始化
    time.sleep(1.0)

    # 执行导航任务
    node.navigate_to_region_and_return(region)

    # 等待任务完成
    while not node.navigation_complete:
        time.sleep(0.5)

    # 再等一会儿确保所有导航完成
    time.sleep(1.5)

    # 清理
    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()

    print("=" * 50)
    print("Task 1 执行完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
