from langchain.agents import initialize_agent, Tool
from agent.QianWenChatLLM import QianWenChatLLM

llm = QianWenChatLLM()


# 模拟问关于订单
def search_order(input: str) -> str:
    return "订单状态：已发货；发货日期：2023-09-15；预计送达时间：2023-09-18"


# 模拟问关于推荐产品
def recommend_product(input: str) -> str:
    return "红色连衣裙"


# 模拟问电商faq
def faq(input: str) -> str:
    return "7天无理由退货"


# 创建了一个 Tool 对象的数组，把这三个函数分别封装在了三个 Tool 对象里面
# 并且定义了描述，这个 description 就是告诉 AI，这个 Tool 是干什么用的，会根据描述做出选择
tools = [
    Tool(
        name="Search Order", func=search_order,
        description="useful for when you need to answer questions about customers orders"
    ),
    Tool(
        name="Recommend Product", func=recommend_product,
        description="useful for when you need to answer questions about product recommendations"
    ),
    Tool(
        name="FAQ", func=faq,
        description="useful for when you need to answer questions about shopping policies, like return policy, shipping policy, etc."
    ),
]
# 指定使用tools，llm，agent则是zero-shot"零样本分类"，不给案例自己推理
# 而 react description，指的是根据你对于 Tool 的描述（description）进行推理（Reasoning）并采取行动（Action）
