from langchain import SerpAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.utilities import ArxivAPIWrapper

from typing import Dict, Tuple
import os
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from agent.qwen_langchain_agent.create_campaign import CreateCampaign
from agent.qwen_langchain_agent.qa import QA
qa=QA()
createCampaign=CreateCampaign()

def tool_wrapper_for_qwen(tool):
    def tool_(query):
        # print(f"query:{query}")
        query = json.loads(query)["param"]
        return tool.run(query)
    return tool_

# 以下是给千问看的工具描述：
TOOLS = [
    {
        'name_for_human':
            '其他问题',
        'name_for_model':
            'QA',
        'description_for_model':
            '回答其他所有问题使用',
        'parameters': [{
            "name": "param",
            "type": "string",
            "description": "知识问答系统",
            'required': True
        }],
        'tool_api': tool_wrapper_for_qwen(qa)
    },{
        'name_for_human':
            '创建活动',
        'name_for_model':
            'CreateCampaign',
        'description_for_model':
            '需要创建营销活动时使用',
        'parameters': [{
            "name": "param",
            "type": "string",
            "description": "活动名称",
            'required': True
        }],
        'tool_api': tool_wrapper_for_qwen(createCampaign)
    }

]
