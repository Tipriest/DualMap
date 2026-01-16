'''
Docstring for applications.oe_llm
- 板卡端执行，调用本地部署的LLM模型进行任务指令解析
'''

import json
import requests
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import re

class TaskPublisher(Node):
    def __init__(self):
        super().__init__('task_target_pub')
        self.publisher_room_ = self.create_publisher(String, 'target_room', 10)
        self.publisher_target_ = self.create_publisher(String, 'target_name', 10)
        self.related_obj_publisher_ = self.create_publisher(String, 'related_object', 10)
        self.hazard_publisher_ = self.create_publisher(String, 'semantic_hazard', 10)

    def publish_room(self, target_room):
        msg = String()
        msg.data = target_room
        self.publisher_room_.publish(msg)
        self.get_logger().info(f'Published room name: {msg.data}')

    def publish_task(self, target_name):
        msg = String()
        msg.data = target_name
        self.publisher_target_.publish(msg)
        self.get_logger().info(f'Published target name: {msg.data}')

    def publish_related_obj(self, related_object):
        msg = String()
        msg.data = related_object
        self.related_obj_publisher_.publish(msg)
        self.get_logger().info(f'Published related object: {msg.data}')


    def publish_hazard(self, avoid_hazard):
        msg = String()
        msg.data = avoid_hazard
        self.hazard_publisher_.publish(msg)
        self.get_logger().info(f'Published semantic hazard: {msg.data}')


def task_extract():

    rclpy.init()
    task_pub = TaskPublisher()

    # MindIE网络服务的位置
    url = "http://127.0.0.1:1025/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    query_text = input("请输入问题：")
    prompt = f"""
    请从以下用户指令中提取三个关键要素：

    用户指令："{query_text}"

    请提取：
    1. **目标房间** (target_room)：要去的房间类型（如卧室、厨房、客厅等），如果有定语也保留（如孩子的卧室等）
    2. **相关物体** (related_object)：与目标房间相关的物体（如床、桌子等）
    3. **寻找物品** (target_object)：需要在目标房间找到的物品
    4. **避开物品** (avoid_object)：路途中需要避开的东西

    规则：
    - 如果某项信息不明确或不存在，请返回 "None"
    - 物品名称应该是具体的（如"被子"而不是"那个被子"），一定会有需要找到的物体！！！
    - 相关物体的意思是，例如"去卧室拿床上的被子"，相关物体就是“床”，如果没有相关物体，请返回 "None"，相关物体如果存在一定是在命令中提到的
    - 只返回JSON格式，不要有其他文本

    输出格式：
    {{
        "target_room": "房间名称",
        "related_object": "物品名称",
        "target_object": "物品名称",
        "avoid_object": "物品名称"
    }}

    现在请生成JSON：
    不要思考。
    """

    # 正确的消息格式
    data = {
        "model": "qwen3",  # config.json中设置的模型名称，按需修改
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": 1280,
        "temperature": 0.7,
    }

    start_time = time.time()

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        result = response.json()

        # 提取回答内容
        if "choices" in result and result["choices"]:
            content = result["choices"][0]["message"]["content"]
            # print("模型回答:")
            # print(content)
        else:
            print("完整响应:")
            print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"请求失败：{e}")
        if hasattr(e, "response") and e.response:
            print("错误详情:", e.response.text)

    end_time = time.time()
    print("cost time", end_time-start_time)

    content_dict = {}
    # 解析JSON
    try:
        content_dict = json.loads(content)
    except json.JSONDecodeError:
        # 如果失败，尝试从文本中提取 JSON 部分
        # 查找 {...} 格式的 JSON

        json_match = re.search(r'\{[\s\S]*\}', content)  # 注意：贪婪，可能拿到多段时仍不稳
        if json_match:
            json_str = json_match.group()
            try:
                content_dict = json.loads(json_str)
            except Exception:
                content_dict = {}
        else:
            # 如果没有找到 JSON，尝试提取键值对
            """
                提取的内容包含： 1、房间 2、相关物体 3、寻找物品 4、避开物品
            """
            room_match = re.search(r'"target_room"\s*:\s*"([^"]+)"', content)
            if room_match:
                content_dict["target_room"] = room_match.group(1)

            related_match = re.search(r'"related_object"\s*:\s*"([^"]+)"', content)
            if related_match:
                content_dict["related_object"] = related_match.group(1)

            target_match = re.search(r'"target_object"\s*:\s*"([^"]+)"', content)
            if target_match:
                content_dict["target_object"] = target_match.group(1)

            avoid_match = re.search(r'"avoid_object"\s*:\s*"([^"]+)"', content)
            if avoid_match:
                content_dict["avoid_object"] = avoid_match.group(1)
            else:
                # 正则判断 None/null
                if re.search(r'"avoid_object"\s*:\s*(null|"None")', content):
                    content_dict["avoid_object"] = "None"

    # 现在可以安全访问字典了
    target_room = content_dict.get("target_room", "None")
    target_name = content_dict.get("target_object", "None")
    related_object = content_dict.get("related_object", "None")
    avoid_hazard = content_dict.get("avoid_object", "None")

    # TODO: 这里有room没有确定对面已经接到了
    if target_room != "None":
        task_pub.publish_room(target_room)
        print(f"Published target name: {target_room}")

    if related_object != "None":
        task_pub.publish_related_obj(related_object)
        print(f"Published related object: {related_object}")

        # 确保offline部分先拿到物体的变量
        time.sleep(2)

    if target_name != "None":
        task_pub.publish_task(target_name)
        print(f"Published target name: {target_name}")

    if avoid_hazard != "None":
        task_pub.publish_hazard(avoid_hazard)
        print(f"Published semantic hazard: {avoid_hazard}")

    rclpy.spin(task_pub)

def fake_task_extract():

    rclpy.init()
    task_pub = TaskPublisher()
    # 现在可以安全访问字典了
    target_room = "bed room"
    target_name = "carpet"
    related_object = "bed"
    avoid_hazard = "carpet"

    # TODO: 这里有room没有确定对面已经接到了
    if target_room != "None":
        task_pub.publish_room(target_room)
        print(f"Published target name: {target_room}")

    if related_object != "None":
        task_pub.publish_related_obj(related_object)
        print(f"Published related object: {related_object}")

        # 确保offline部分先拿到物体的变量
        time.sleep(2)

    if target_name != "None":
        task_pub.publish_task(target_name)
        print(f"Published target name: {target_name}")

    if avoid_hazard != "None":
        task_pub.publish_hazard(avoid_hazard)
        print(f"Published semantic hazard: {avoid_hazard}")

    rclpy.spin(task_pub)

if __name__ == '__main__':

    # task_extract()
    fake_task_extract()