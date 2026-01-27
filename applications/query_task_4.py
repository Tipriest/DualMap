"""
完成task04：找到相关物体 & 动态避障 & pop 语义障碍
NOTE: 不包含 semantic hazard的处理部分，默认task 23没有需要避障的语义物体（地毯）
dualmap 主机端执行：订阅目标/相关物体/房间等，基于离线 local map 查询位置；
并通过 Nav2 NavigateToPose 导航到目标点，并支持面向目标的旋转与recovery流程。
"""

import os

os.environ["DISPLAY"] = ""

import sys
import time
import yaml
import json
import requests

import rclpy
from rclpy.executors import MultiThreadedExecutor


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # applications/
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)  # DualMap/
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# sys.path.append("/home/tang123/Documents/DualMap/applications/utils")
sys.path.append(os.path.join(PROJECT_ROOT, "applications/utils"))

from query_task_subscriber import (
    TaskSubscriber,
    write_log
)

LOG_FILE = "nav_result_task2-3.txt"



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
    prompt = f"""
请从以下用户指令中提取三个关键要素：
用户指令：“{user_query}”
请提取：
1. **目标房间** (target_room)：要去的房间类型（如卧室、厨房、客厅等）
2. **相关物体** (related_object)：与目标物体相关的物体，可能是家具的类型（如床、桌子等）
3. **寻找物品** (target_object)：需要在目标房间找到的物品
4. **避开物品** (avoid_object)：路途中需要避开的东西
规则：
- 如果某项信息不明确或不存在，请返回 "None"
- 物品名称应该是具体的（如"被子"而不是"那个被子"），一定会有需要找到的物体！！！
- 相关物体的意思是，例如"去卧室拿床上的被子"，相关物体就是“床”，如果没有相关物体，请返回 "None"，相关物体如果存在一定是在命令中提到的
- 有可能不存在相关物体！！比如去书房找瓶子，就没有相关物体，你应当对 related_object 返回"None"!!!
- 只返回JSON格式，不要有其他文本
- 房间只可能是bedroom，childroom，livingroom，kitchen 中的一个，名称必须原样返回 4者中的一个，如 bedroom！！！
- 返回的物体名称需要是英文的类型，比如输出的指令是“床”，你应当返回“bed”
输出格式：
{{
    "target_room": "房间名称",
    "related_object": "物品名称",
    "target_object": "物品名称",
    "avoid_object": "物品名称"
}}
现在请生成JSON：
"""

    # 构建请求体
    payload = {
        "model": "qwen-max",  # MODEL qwen-turbo, qwen-plus
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,  # 低温度以保证输出稳定性
        "top_p": 0.8,
        "stream": False,
        "max_tokens": 1024,
    }

    try:
        # 发送请求
        response = requests.post(
            url, headers=headers, data=json.dumps(payload), timeout=30
        )
        response.raise_for_status()  # 检查HTTP错误

        result = response.json()

        # 解析响应
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]

            # 清理响应内容，提取JSON部分
            content = content.strip()

            # 查找JSON对象
            import re

            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                json_str = json_match.group()
                parsed_data = json.loads(json_str)

                # 确保所有键都存在，缺失的键设为"None"
                required_keys = [
                    "target_room",
                    "related_object",
                    "target_object",
                    "avoid_object",
                ]
                for key in required_keys:
                    if key not in parsed_data:
                        parsed_data[key] = "None"

                return parsed_data
            else:
                raise ValueError("API响应中未找到有效的JSON格式")
        else:
            raise ValueError("API响应格式异常")

    except requests.exceptions.RequestException as e:
        print(f"API请求失败: {e}")
        # 返回默认值或抛出异常，根据你的错误处理策略决定
        return {
            "target_room": "None",
            "related_object": "None",
            "target_object": "None",
            "avoid_object": "None",
        }
    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {e}")
        return {
            "target_room": "None",
            "related_object": "None",
            "target_object": "None",
            "avoid_object": "None",
        }


def main():
    # 从配置读取
    # cfg_path = "/home/tang123/Documents/DualMap/config/query/query_task_4.yaml"
    cfg_path = os.path.join(PROJECT_ROOT, "config/query/query_task_4.yaml")

    # 读取指令
    query_text = input("请输入指令：")
    # LOG1
    write_log(f"Start: Command received - '{query_text}'")
    qwen_result = parse_command_with_qwen(cfg_path, query_text)

    # DEBUG: 免解析指令
    # target_room = "bed room"
    # target_name = "bed"
    # related_object = "bed"
    # avoid_hazard = "None"

    target_room = qwen_result["target_room"]
    target_name = qwen_result["target_object"]
    related_object = qwen_result["related_object"]
    avoid_hazard = qwen_result["avoid_object"]

    # 初始化ROS和Node
    rclpy.init()
    node = TaskSubscriber(cfg_path)

    # 等待初始化
    time.sleep(1)

    if avoid_hazard != "None":
        node._hazard_cb(avoid_hazard)
        print(f"poped HAZARD: {avoid_hazard}")
    print("************************************************")

    # 设置目标信息
    if target_room != "None":
        node.room = target_room
        node._room_cb(target_room)
        print(f"目标房间: {target_room}")

        # 等待房间准备完成
        wait_start = time.time()
        while not node.is_room_ready:
            if time.time() - wait_start > 5.0:
                print("等待房间准备超时！")
                break
            time.sleep(0.1)
        print("ROOM READY!")

    print("************************************************")

    if related_object == "None":
        node.related_object_name = "None"
        node.target_name = target_name
        node._target_cb(target_name)
    else:
        node.related_object_name = related_object
        node.target_name = target_name
        node._related_obj_cb(related_object)

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
