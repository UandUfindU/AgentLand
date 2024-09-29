from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from LLM import *

llm = GLMFlash_LLM_Memory(api_key="50d79333873f56e2ac8029560c32d2aa.pxMNM2qMDk5KdvdQ")

# 定义聊天提示模板
prompt_message = """
作为分院帽，你有以下几点要注意
语气：古老而自信，充满魔法世界的魅力
意图：通过不断提出假设的魔法世界中的问题，判断带上魔法帽的人的性格（例如 如果你身处地牢，你愿如何脱困？）
目的：挑选出佩戴者该前往的学院，格兰芬多——勇敢、活力、骑士精神。 拉文克劳——心思敏捷、机智、博学。 斯莱特林——为达目的不择手段。 赫奇帕奇——正直、忠诚、诚实、不怕艰辛。

你将不断提问，直到信息足够后返回：决定了！你的归宿是

以下是佩戴者的最新回话:
{question}
"""
prompt = ChatPromptTemplate.from_messages([("human", prompt_message)])

# 定义 RAG 链
chain = (
    {"question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

#构建main函数
if __name__ == "__main__":
    print("欢迎来到 Harry Potter 世界！")
    print("你好小家伙，我是分院帽，和我打个招呼吧！")
    # 测试循环
    while True:
        user_input = input("输入'1'强行摘下分院帽，或输入你的回答：")
        
        if user_input == '1':
            print("你强行摘下了分院帽。")
            llm.clear_memory()
            break
        qustion=user_input
        response = chain.invoke(qustion)
        print(response)
        
        if "决定了" in response:
            print("分院帽已决定你的归宿！")
            llm.clear_memory()
            break

