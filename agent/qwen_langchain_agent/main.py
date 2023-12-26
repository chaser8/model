# 国内连 hugginface 网络不好，这段代码可能需要多重试
from typing import Tuple

from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from agent.qwen_langchain_agent.prompt_template import build_planning_prompt
from agent.qwen_langchain_agent.tools import TOOLS

model_dir = snapshot_download('qwen/Qwen-7B-Chat')

# checkpoint = "Qwen/Qwen-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)
# model.generation_config.do_sample = False  # greedy

# stop = ["Observation:", "Observation:\n"]
# react_stop_words_tokens = [tokenizer.encode(stop_) for stop_ in stop]

# prompt_1 = build_planning_prompt(TOOLS, query="怎么创建一个4升5的营销活动？")
# print(prompt_1)
# response_1, _ = model.chat(tokenizer, prompt_1, history=None, stop_words_ids=react_stop_words_tokens)
# print(response_1)


def parse_latest_plugin_call(text: str) -> Tuple[str, str]:
    i = text.rfind('\nAction:')
    j = text.rfind('\nAction Input:')
    k = text.rfind('\nObservation:')
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is ommited by the LLM,
            # because the output text may have discarded the stop word.
            text = text.rstrip() + '\nObservation:'  # Add it back.
            k = text.rfind('\nObservation:')
    if 0 <= i < j < k:
        plugin_name = text[i + len('\nAction:'):j].strip()
        plugin_args = text[j + len('\nAction Input:'):k].strip()
        return plugin_name, plugin_args
    return '', ''


def use_api(tools, response):
    use_toolname, action_input = parse_latest_plugin_call(response)
    if use_toolname == "":
        return "no tool founds"

    used_tool_meta = list(filter(lambda x: x["name_for_model"] == use_toolname, tools))
    if len(used_tool_meta) == 0:
        return "no tool founds"

    api_output = used_tool_meta[0]["tool_api"](action_input)
    return api_output


# api_output = use_api(TOOLS, response_1)
# print(api_output)
#
# prompt_2 = prompt_1 + response_1 + ' ' + api_output
# stop = ["Observation:", "Observation:\n"]
# react_stop_words_tokens = [tokenizer.encode(stop_) for stop_ in stop]
# response_2, _ = model.chat(tokenizer, prompt_2, history=None, stop_words_ids=react_stop_words_tokens)
# print(prompt_2, response_2)


def main(query, choose_tools):
    prompt = build_planning_prompt(choose_tools, query) # 组织prompt
    print(prompt)
    stop = ["Observation:", "Observation:\n"]
    react_stop_words_tokens = [tokenizer.encode(stop_) for stop_ in stop]
    response, _ = model.chat(tokenizer, prompt, history=None, stop_words_ids=react_stop_words_tokens)

    while "Final Answer:" not in response: # 出现final Answer时结束
        api_output = use_api(choose_tools, response) # 抽取入参并执行api
        api_output = str(api_output) # 部分api工具返回结果非字符串格式需进行转化后输出
        if "no tool founds" == api_output:
            break
        print("\033[32m" + response + "\033[0m" + "\033[34m" + ' ' + api_output + "\033[0m")
        prompt = prompt + response + ' ' + api_output # 合并api输出
        response, _ = model.chat(tokenizer, prompt, history=None, stop_words_ids=react_stop_words_tokens) # 继续生成

    print("\033[32m" + response + "\033[0m")

query = "怎么创建一个4升5的营销活动？" # 所提问题
choose_tools = TOOLS # 选择备选工具
print("=" * 10)
main(query, choose_tools)