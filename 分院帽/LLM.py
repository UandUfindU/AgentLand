from langchain.llms.base import LLM
from typing import Any, Dict, List, Optional
from openai import OpenAI
from langchain.callbacks.manager import CallbackManagerForLLMRun
from datetime import datetime  # 引入datetime模块

class DeepSeek_LLM(LLM):
    client: OpenAI = None

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        super().__init__()
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        print("DeepSeek API 客户端已初始化")

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        try:
            # 记录开始时间
            start_time = datetime.now()

            messages = [
                {"role": "system", "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"},
                {"role": "user", "content": prompt}
            ]
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                #stream=False
                stream=True
            )
            
            # 记录结束时间
            end_time = datetime.now()
            
            # 计算总响应时间
            total_time = end_time - start_time
            
            # 打印总响应时间
            print(f"Model response time: {total_time}")

            return response.choices[0].message.content
        
        except Exception as e:
            print(f"Error: {e}")
            return "Error in DeepSeek API call."

    @property
    def _llm_type(self) -> str:
        return "DeepSeek_LLM"

from langchain.llms.base import LLM
from typing import Any, List, Optional
from zhipuai import ZhipuAI
from langchain.callbacks.manager import CallbackManagerForLLMRun
from datetime import datetime

class GLMFlash_LLM(LLM):
    client: ZhipuAI = None

    def __init__(self, api_key: str, base_url: str = "https://open.bigmodel.cn"):
        super().__init__()
        self.client = ZhipuAI(api_key=api_key)
        print("ZhipuAI 客户端已初始化")

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        try:
            # 记录开始时间
            start_time = datetime.now()

            messages = [
                {"role": "system", "content": "你是哈利波特故事中魔法学院的分院帽，你负责将学生分到格兰芬多，赫奇帕奇，拉文克劳以及斯莱特林四个学院。你要不断提问，直到得到足够的信息才返回：决定了！XX学院"},
                {"role": "user", "content": prompt}
            ]
            response = self.client.chat.completions.create(
                model="glm-4-flash",  # 使用GLM-4-flash模型
                messages=messages,
                stream=False,
                max_tokens=1024,  # 你可以根据需要调整
                temperature=0.95,  # 可选参数
                top_p=0.7,  # 可选参数
                stop=stop  # 停止词
            )
            
            # 记录结束时间
            end_time = datetime.now()
            
            # 计算总响应时间
            total_time = end_time - start_time
            
            # 打印总响应时间
            print(f"Model response time: {total_time}")

            return response.choices[0].message.content
        
        except Exception as e:
            print(f"Error: {e}")
            return "Error in GLM-4-flash API call."

    @property
    def _llm_type(self) -> str:
        return "GLMFlash_LLM"
    
class GLMFlash_LLM_Memory(LLM):
    client: ZhipuAI = None
    conversation_history: List[Dict[str, str]] = []

    def __init__(self, api_key: str, base_url: str = "https://open.bigmodel.cn"):
        super().__init__()
        self.client = ZhipuAI(api_key=api_key)
        print("ZhipuAI 客户端已初始化")

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        try:
            # 记录开始时间
            start_time = datetime.now()

            # 将用户输入添加到对话历史
            self.conversation_history.append({"role": "user", "content": prompt})

            # 包含系统提示与对话历史
            messages = [
                {"role": "system", "content": "你是哈利波特故事中魔法学院的分院帽，你负责将学生分到格兰芬多，赫奇帕奇，拉文克劳以及斯莱特林四个学院。你要不断提问，直到得到足够的信息才返回：决定了！XX学院"},
                {"role": "assistant", "content": "哦，小家伙，向我打个招呼吧！"}
            ] + [{"role": item["role"], "content": item["content"]} for item in self.conversation_history]

            response = self.client.chat.completions.create(
                model="glm-4-flash",
                messages=messages,
                stream=False,
                max_tokens=1024,
                temperature=0.95,
                top_p=0.7,
                stop=stop
            )
            
            # 记录结束时间
            end_time = datetime.now()
            total_time = end_time - start_time
            print(f"Model response time: {total_time}")

            # 将模型响应添加到对话历史
            bot_response = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": bot_response})

            return bot_response
        
        except Exception as e:
            print(f"Error: {e}")
            return "Error in GLM-4-flash API call."
        
    def clear_memory(self):
        """清空对话历史"""
        self.conversation_history = []
        print("对话历史已清空。")

        
    
    @property
    def _llm_type(self) -> str:
        return "GLMFlash_LLM"





