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
        """更新采样参数，只更新非 None 的参数"""
        if max_tokens is not None:
            self.sampling_params.max_tokens = max_tokens
        else:
            self.sampling_params.max_tokens = 32768
        if temperature is not None:
            self.sampling_params.temperature = temperature
        else:
            self.sampling_params.temperature = 0.8
        if top_p is not None:
            self.sampling_params.top_p = top_p
        else:
            self.sampling_params.top_p = 0.90
        logger.info(f"更新后的采样参数: max_tokens={self.sampling_params.max_tokens}, temperature={self.sampling_params.temperature}, top_p={self.sampling_params.top_p}")

    async def load_model(self):
        try:
            logger.info(f"正在从 {self.model_path} 加载模型...")
            
            # Create an LLM.
            self.model = LLM(
                    model=self.model_path,
                    trust_remote_code=True,
                    tensor_parallel_size=4,  # 设置为 4 卡并行
                    enforce_eager=True,  # 改为 True 以避免动态图问题
                    max_seq_len_to_capture=32768,  # 降低序列长度
                    gpu_memory_utilization=0.8,  # 降低显存利用率
                    max_num_seqs=4,  # 进一步减少并发序列数量
                    # dtype="float16",  # 使用 FP16 进行推理
                    # quantization="awq",  # 添加量化支持
                    max_model_len=32768  # 限制最大模型长度
                    )
            
            # vllm 会自动加载相应的 tokenizer
            self.tokenizer = self.model.get_tokenizer()
            logger.info("模型加载成功")
            return True
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False

    def create_prompt(self, messages: list) -> str:
        if self.tokenizer is None:
            raise ValueError("Tokenizer 尚未加载")
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        logger.info(f"生成的提示: {prompt}")
        return prompt
    
    async def generate_response(self, user_input: str) -> tuple[str, int]:
        try:
            prompt = self.create_prompt(user_input)
            # 计算输入token数量
            input_tokens = len(self.tokenizer.encode(prompt))
            
            outputs = self.model.generate(
                prompts=[prompt],
                sampling_params=self.sampling_params,
            )
            response_text = outputs[0].outputs[0].text.strip()
            # 计算输出token数量
            output_tokens = len(outputs[0].outputs[0].token_ids)
            # 总token数量为输入和输出之和
            total_tokens = input_tokens + output_tokens
            
            print(f"生成的输出: {response_text}")
            print(f"Token统计 - 输入: {input_tokens}, 输出: {output_tokens}, 总计: {total_tokens}")
            return response_text, input_tokens, output_tokens, total_tokens
        except Exception as e:
            logger.error(f"生成响应失败: {e}")
            return "生成响应失败，发生错误。", 0