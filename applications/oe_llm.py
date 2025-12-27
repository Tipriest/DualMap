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
        self.publisher_ = self.create_publisher(String, 'target_name', 10)
        self.hazard_publisher_ = self.create_publisher(String, 'semantic_hazard', 10)

    def publish_task(self, target_name):
        msg = String()
        msg.data = target_name
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published target name: {msg.data}')
        
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
    2. **寻找物品** (target_object)：需要在目标房间找到的物品
    3. **避开物品** (avoid_object)：路途中需要避开的东西

    规则：
    - 如果某项信息不明确或不存在，请返回 "None"
    - 物品名称应该是具体的（如"被子"而不是"那个被子"），一定会有需要找到的物体！！！
    - 只返回JSON格式，不要有其他文本

    输出格式：
    {{
        "target_room": "房间名称",
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
    

    # 解析JSON
    try:
        content_dict = json.loads(content)
    except json.JSONDecodeError:
        # 如果失败，尝试从文本中提取 JSON 部分
        # 查找 {...} 格式的 JSON
        json_match = re.search(r'\{[^{}]*\}', content)
        if json_match:
            json_str = json_match.group()
            content_dict = json.loads(json_str)
        else:
            # 如果没有找到 JSON，尝试提取键值对
            content_dict = {}
            # 查找 target_object
            target_match = re.search(r'"target_object":\s*"([^"]+)"', content)
            if target_match:
                content_dict["target_object"] = target_match.group(1)
            
            # 查找 avoid_object
            avoid_match = re.search(r'"avoid_object":\s*"([^"]+)"', content)
            if avoid_match:
                content_dict["avoid_object"] = avoid_match.group(1)
            elif '"avoid_object":\s*"None"' in content or '"avoid_object":\s*null' in content:
                content_dict["avoid_object"] = "None"
        
        # 现在可以安全访问字典了
        target_name = content_dict.get("target_object", "None")
        avoid_hazard = content_dict.get("avoid_object", "None")

    
    task_pub.publish_task(target_name)
    task_pub.publish_hazard(avoid_hazard)
    
    print(f"Published target name: {target_name}")
    print(f"Published semantic hazard: {avoid_hazard}")
    

    rclpy.spin(task_pub)
    

if __name__ == '__main__':
    
    task_extract()