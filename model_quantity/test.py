from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.6, top_p=0.9)

# Create an LLM.
llm = LLM(model="/home/ubuntu/.cache/modelscope/hub/models/qwen/Qwen-72B-Chat-Int4",
          trust_remote_code=True,
          tensor_parallel_size=2,  # 设置为 2 卡并行
          max_model_len=8192,  # 或者 16384 以减少显存需求
          gpu_memory_utilization=0.95,  # 调高显存利用率
          )
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")