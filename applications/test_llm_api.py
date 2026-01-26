import yaml
import requests
import os
import json
import sys

# 配置文件路径，保持与主程序一致
CFG_PATH = "/home/tang123/Documents/DualMap/config/query/query_task_2_3.yaml"


def test_qwen_api():
    print(f"正在读取配置文件: {CFG_PATH}")

    # 1. 读取配置文件
    try:
        with open(CFG_PATH, "r") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[错误] 找不到配置文件: {CFG_PATH}")
        return False
    except Exception as e:
        print(f"[错误] 读取配置文件失败: {e}")
        return False

    # 2. 获取 API Key
    api_key = cfg.get("api_key")
    if not api_key:
        print("[错误] 配置文件中未找到 'api_key' 字段。")
        return False
    else:
        print(f"[成功] 找到 API Key (前4位: {api_key[:4]}...)")

    # 3. 准备请求参数
    # 使用环境变量中的 base_url，如果没有则使用默认值
    base_url = os.getenv(
        "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    url = f"{base_url}/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # 构建一个简单的测试 Prompt
    payload = {
        "model": "qwen-max",  # 使用与主程序相同的模型
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Please say 'API Connection Successful' and nothing else.",
            },
        ],
        "temperature": 0.1,
        "max_tokens": 50,
    }

    print(f"\n正在向 {url} 发送测试请求...")

    # 4. 发送请求
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)

        # 5. 检查结果
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                print("-" * 50)
                print("[测试成功] API 响应正常！")
                print(f"AI 回复: {content}")
                print("-" * 50)
                return True
            else:
                print(f"[错误] API 返回格式异常: {result}")
                return False
        else:
            print(f"[错误] HTTP 请求失败，状态码: {response.status_code}")
            print(f"错误详情: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print("[错误] 请求超时，请检查网络连接或代理设置。")
        return False
    except Exception as e:
        print(f"[错误] 发生未知异常: {e}")
        return False


if __name__ == "__main__":
    success = test_qwen_api()
    sys.exit(0 if success else 1)
