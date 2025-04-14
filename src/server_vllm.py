from vllm import LLM, SamplingParams
import logging
from typing import AsyncIterable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self, model_path="/app/models"):
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self.sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.90,
            max_tokens=32768,
        )
    # 在 ModelService 类中添加此方法
    def update_sampling_params(self, max_tokens=None, temperature=None, top_p=None):
        """更新采样参数"""
        if max_tokens is not None:
            self.sampling_params.max_tokens = max_tokens
        if temperature is not None:
            self.sampling_params.temperature = temperature
        if top_p is not None:
            self.sampling_params.top_p = top_p

    async def load_model(self):
        try:
            logger.info(f"正在从 {self.model_path} 加载模型...")
            # self.model = LLM(
            #     model=self.model_path,
            #     tensor_parallel_size=2,  # 设置为 2 卡并行
            #     max_seq_len_to_capture=8096,
            #     gpu_memory_utilization=0.95,
            #     quantization = "gptq"
            # )
            
            # Create an LLM.
            self.model = LLM(
                    model=self.model_path,
                    trust_remote_code=True,
                    tensor_parallel_size=4,  # 设置为 4 卡并行
                    enforce_eager=True,  # 强制使用 eager execution 模式
                    max_model_len=32768,  # 或者 16384 以减少显存需求
                    gpu_memory_utilization=0.65,  # 调高显存利用率
                    max_num_seqs=4  # 减小并发序列数量
                    )
            
            # vllm 会自动加载相应的 tokenizer
            self.tokenizer = self.model.get_tokenizer()
            logger.info("模型加载成功")
            return True
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False

    # def create_prompt(self, messages: str) -> str:
    #     """
    #     根据消息数组生成提示。
    #     消息格式为：
    #     [
    #         { "role": "system", "content": "你是一个AI助手" },
    #         { "role": "user", "content": "今天天气怎么样？" }
    #     ]
    #     """
    #     prompt = ""
    #     for message in messages:
    #         role = message["role"]
    #         content = message["content"]
    #         if role == "system":
    #             prompt += f"<|system|>\n{content}\n"
    #         elif role == "user":
    #             prompt += f"<|user|>\n{content}\n"
    #         elif role == "assistant":
    #             prompt += f"<|assistant|>\n{content}\n"
    #     print(f"生成的提示: {prompt}")
    #     return prompt
    def create_prompt(self, messages: list) -> str:
        if self.tokenizer is None:
            raise ValueError("Tokenizer 尚未加载")
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(f"生成的提示: {prompt}")
        return prompt
    
    async def generate_response(self, user_input: str) -> str:
        try:
            prompt = self.create_prompt(user_input)
            outputs = self.model.generate(
                prompts=[prompt],
                sampling_params=self.sampling_params,
            )
            print(f"生成的输出: {outputs[0].outputs[0].text.strip()}")
            return outputs[0].outputs[0].text.strip()
        except Exception as e:
            logger.error(f"生成响应失败: {e}")
            return "生成响应失败，发生错误。"


    async def generate_stream_response(self, user_input: str) -> AsyncIterable[str]:
        prompt = self.create_prompt(user_input)
        
        # 使用 vLLM 的流式生成
        request_id = 0
        generator = self.model.generate(
            prompts=[prompt],
            sampling_params=self.sampling_params,
            stream=True,
            request_id=request_id,
        )
        
        for request_output in generator:
            if request_output.outputs:
                yield request_output.outputs[0].text