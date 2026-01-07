import json
import requests
from colorama import Fore, Style

class TaskExtractor:
    def __init__(
        self,
        query_text,
        llm_model: str = "deepseek-r1:14b"
    ):
        self.llm_model = llm_model
        self.base_url = "http://localhost:11434"
        self.query_text = query_text
    
        
        self._check_ollama_service()
        
        print("==> LLM loaded.")
        
    def _check_ollama_service(self):
        """检查Ollama服务是否正常运行"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama service is running")
                
                # 检查模型是否已下载
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                
                if self.llm_model not in model_names:
                    print(f"⚠️  Model '{self.llm_model}' not found in Ollama.")
                    print(f"    Available models: {', '.join(model_names)}")
                    print(f"    Please run: ollama pull {self.llm_model}")
            else:
                print("❌ Ollama service returned error")
        except requests.ConnectionError:
            raise


    
    def generate_streaming(self, prompt: str, system_prompt: str = None) -> str:
        """使用流式API获取完整响应"""
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": True,  # 启用流式
            "options": {
                "temperature": 0.0,
                "num_predict": 2048,  # 更大的生成上限
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "stop": ["</s>"],  # 只保留主要停止词
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=300
            )
            
            if response.status_code == 200:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            json_data = json.loads(line.decode('utf-8'))
                            if 'response' in json_data:
                                full_response += json_data['response']
                            if json_data.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue
                
                print(Fore.LIGHTGREEN_EX + f"==> 流式生成完成 ({len(full_response)} 字符)" + Style.RESET_ALL)
                return full_response.strip()
            else:
                print(f"流式API错误: {response.status_code}")
                return ""
                
        except Exception as e:
            print(f"流式生成错误: {e}")
            return ""


    def extract_navigation_components(self) -> dict:
        """提取导航指令的三个要素"""
        
        prompt = f"""
        请从以下用户指令中提取三个关键要素：
        
        用户指令："{self.query_text}"
        
        请提取：
        1. **目标房间** (target_room)：要去的房间类型（如卧室、厨房、客厅等），如果有定语也保留（如孩子的卧室等）
        2. **寻找物品** (target_object)：需要在目标房间找到的物品
        3. **避开物品** (avoid_object)：路途中需要避开的东西
        
        规则：
        - 如果某项信息不明确或不存在，请返回 "None"
        - 物品名称应该是具体的（如"被子"而不是"那个被子"）
        - 只返回JSON格式，不要有其他文本
        
        输出格式：
        {{
            "target_room": "房间名称",
            "target_object": "物品名称", 
            "avoid_object": "物品名称"
        }}
        
        现在请生成JSON：
        """
        
        # 使用流式API
        response = self.generate_streaming(prompt)
        print(f"LLM响应: {response}")
        
        # 解析JSON
        try:
            result = json.loads(response)
            for key in ["target_room", "target_object", "avoid_object"]:
                if key not in result:
                    result[key] = "None"
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                except:
                    result = {
                        "target_room": "None",
                        "target_object": "None",
                        "avoid_object": "None"
                    }
            else:
                result = {
                    "target_room": "None",
                    "target_object": "None",
                    "avoid_object": "None"
                }
        
        print(f"解析结果: {result}")
        return result


    
# if __name__ == "__main__":    

#     user_query = input("Input your order: ")
#     ai_scientist = TaskExtractor(user_query)

#     targets = ai_scientist.extract_navigation_components()
        