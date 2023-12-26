from typing import Optional, List

from langchain_core.language_models import LLM
from modelscope import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen-7B-Chat",trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("qwen/Qwen-7B-Chat", device_map='cpu', low_cpu_mem_usage=True, trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-7B-Chat",  trust_remote_code=True)

class QianWenChatLLM(LLM):
    # global history = None
    max_length = 10000
    temperature: float = 0.01
    top_p = 0.9

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self):
        return "ChatLLM"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        print(f"=====>{prompt}")
        response, history = model.chat(tokenizer, prompt, history=None)
        # print(history)
        #torch_gc()
        return response