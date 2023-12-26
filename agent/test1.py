from typing import Optional, List

import openai, os
from langchain.agents import initialize_agent

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAIChat
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain_core.language_models import LLM
from langchain_core.prompts import StringPromptTemplate
from langchain_core.tools import Tool
from modelscope import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

from agent.QianWenChatLLM import QianWenChatLLM
from agent.tool import tools

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.environ.get("OPENAI_API_KEY")

# llm = ChatOpenAI(temperature=0,model="gpt-3.5-turbo")


llm = QianWenChatLLM()

multiple_choice = """
请针对 >>> 和 <<< 中间的用户问题，选择一个合适的工具去回答它的问题。只要用A、B、C的选项字母告诉我答案。
如果你觉得都不合适，就选D。
>>>{question}<<<
我们有的工具包括：
A. 一个能够查询商品信息，为用户进行商品导购的工具
B. 一个能够查询订单信息，获得最新的订单情况的工具
C. 一个能够搜索商家的退换货政策、运费、物流时长、支付渠道、覆盖国家的工具
D. 都不合适
"""


multiple_choice_prompt = PromptTemplate(template=multiple_choice, input_variables=["question"])
choice_chain = LLMChain(llm=llm, prompt=multiple_choice_prompt, output_key="answer")
# question = "我想买一件衣服，但是不知道哪个款式好看，你能帮我推荐一下吗？"
# print(choice_chain(question))

# question = "我有一张订单，订单号是 2022ABCDE，一直没有收到，能麻烦帮我查一下吗？"
# print(choice_chain(question))
#
# question = "请问你们的货，能送到三亚吗？大概需要几天？"
# print(choice_chain(question))
#
# question = "今天天气怎么样？"
# print(choice_chain(question))

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

question = "我想买一件衣服，但是不知道哪个款式好看，你能帮我推荐一下吗？"
result=agent.run(question)
print(result)


