from unsloth import FastLanguageModel
from vllm import SamplingParams
import logging
from typing import AsyncIterable
# import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self, model_path="xmindai/xm-phi-medical"):
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self.sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=8096,
        )

    async def load_model(self):
        try:
            logger.info(f"正在从 {self.model_path} 加载模型...")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name="unsloth/phi-4",  # 修改为当前文件夹下的 model
                max_seq_length=8096,
                load_in_4bit=False,  # False for LoRA 16bit
                fast_inference=True,  # Enable vLLM fast inference
                max_lora_rank=32,
                gpu_memory_utilization=0.9,  # Reduce if out of memory
                # tensor_parallel_size=2,  # 设置为 2 卡并行
            )

            try:
                logger.info("正在加载 LoRA 权重...")
                self.model.load_adapter("xmindai/xm-phi-medical")
                logger.info("LoRA 权重加载成功")
                # 合并并保存模型到 model 文件夹下
                logger.info("正在合并并保存模型...")
                self.model.merge_and_unload()
                self.model.save_pretrained("model")
                logger.info("模型已成功保存到 'model' 文件夹")
                # 添加多 GPU 支持
                # if torch.cuda.device_count() > 1:
                #     logger.info(f"检测到 {torch.cuda.device_count()} 张 GPU，启用 DataParallel 模式")
                #     self.model = torch.nn.DataParallel(self.model)
            except Exception as e:
                logger.warning(f"LoRA 加载失败: {e}")
            return True
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False

    async def generate_response(self, user_input: str) -> str:
        prompt = self.tokenizer.apply_chat_template([
            {"role": "system", "content": """
            请以一下格式回答问题：:
            <reasoning>
            ...
            </reasoning>
            <answer>
            ...
            </answer>
            """},
            {"role": "user", "content": user_input}
        ], tokenize=False, add_generation_prompt=True)

        output = self.model.fast_generate(
            prompt,
            sampling_params=self.sampling_params,
        )[0].outputs[0].text
        
        return output.strip()

    async def generate_stream_response(self, user_input: str) -> AsyncIterable[str]:
        prompt = self.tokenizer.apply_chat_template([
            {"role": "system", "content": """
            请以一下格式回答问题：:
            <reasoning>
            ...
            </reasoning>
            <answer>
            ...
            </answer>
            """},
            {"role": "user", "content": user_input}
        ], tokenize=False, add_generation_prompt=True)

        # 使用 vLLM 的流式生成
        async for output in self.model.fast_generate_stream(
            prompt,
            sampling_params=self.sampling_params,
        ):
            if output.outputs:
                yield output.outputs[0].text