
[LLM Class — vLLM](https://docs.vllm.ai/en/latest/api/offline_inference/llm.html)

[Quantization — vLLM](https://docs.vllm.ai/en/latest/features/quantization/index.html)

[GPTQModel — vLLM](https://docs.vllm.ai/en/latest/features/quantization/gptqmodel.html)

[ModelCloud/GPTQModel: Production ready LLM model compression/quantization toolkit with hw accelerated inference support for both cpu/gpu via HF, vLLM, and SGLang.](https://github.com/ModelCloud/GPTQModel)

## GPTQModel 笔记

### 概述

**GPTQModel** 是一个用于大规模语言模型（LLM）量化的工具，旨在通过将模型权重从 16 位（BF16/FP16）量化到 4 位（INT4）或 8 位（INT8），显著减少模型的内存占用，并提高推理性能。

### 量化技术

量化是将高精度（BF16/FP16）的浮点权重转化为低精度的整数（INT4 或 INT8），从而减小内存占用并加速模型推理。通过这种方式，GPTQModel 能够在保证性能的前提下，优化内存使用，并提升推理速度。

### 主要特点

1. **动态模块量化**：

	- GPTQModel 提供了 **动态** 模块量化功能，允许对不同层或模块应用不同的量化参数。这种方式更灵活，能够根据每个模块的特点进行定制优化，最大化性能。
	- 动态量化完全集成在 vLLM 中，并得到了 ModelCloud.AI 团队的支持。
	- 这种量化技术为开发者提供了精细的控制，使得用户能够进一步优化模型的每个部分。

2. **支持的硬件与优化**：

	- GPTQModel 支持通过 vLLM 的 **Marlin** 和 **Machete** 自定义内核，优化 Nvidia 的 Ampere（A100+）和 Hopper（H100+）系列 GPU 上的批处理事务每秒（tps）和令牌延迟性能。
	- 这两个内核是经过 vLLM 和 NeuralMagic（现为 Redhat 一部分）优化的，能够提供世界级的推理性能，特别适用于量化 GPTQ 模型。

### 使用方法

您可以通过以下方式开始使用 GPTQModel 量化自己的模型：

1. **安装 GPTQModel**： 安装 GPTQModel 工具包，以便开始量化：

	```bash

	pip install -U gptqmodel --no-build-isolation -v
    ```

2. **量化模型**： 使用以下 Python 示例代码，您可以量化特定模型，例如 `meta-llama/Llama-3.2-1B-Instruct`：

	```python
    from datasets import load_dataset
    from gptqmodel import GPTQModel, QuantizeConfig
    
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    quant_path = "Llama-3.2-1B-Instruct-gptqmodel-4bit"
    
    calibration_dataset = load_dataset(
        "allenai/c4",
        data_files="en/c4-train.00001-of-01024.json.gz",
        split="train"
      ).select(range(1024))["text"]
    
    quant_config = QuantizeConfig(bits=4, group_size=128)
    
    model = GPTQModel.load(model_id, quant_config)
    
    # 根据 GPU/VRAM 配置调整批量大小，加速量化
    model.quantize(calibration_dataset, batch_size=2)
    
    model.save(quant_path)
    ```

3. **运行量化后的模型**： 量化完成后，您可以通过以下命令运行量化的 GPTQModel 模型：

	```bash
    python examples/offline_inference/llm_engine_example.py --model DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2
    ```

4. **LLM 接口支持**： 量化模型还可以直接通过 vLLM 的接口进行推理。以下是一个示例，展示如何生成文本：

	```python
    from vllm import LLM, SamplingParams
    
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9)
    
    llm = LLM(model="DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2")
    outputs = llm.generate(prompts, sampling_params)
    
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    ```

### 优势与应用

- **内存优化**：通过将权重量化为低精度整数（如 INT4 或 INT8），大大减少了内存占用，适合部署在资源受限的环境中。
- **推理性能提升**：量化后，模型的推理速度得到显著提升，能够处理更多的请求，适用于高并发的实时应用场景。
- **灵活的量化配置**：动态量化功能允许开发者针对模型的不同层或模块应用定制化的量化策略，从而进一步优化模型性能。

GPTQModel 提供了一种高效、灵活的量化方法，使得大规模语言模型能够在保持良好推理性能的同时，减少内存占用。通过动态模块量化和优化的推理内核，GPTQModel 是一种适用于大规模生产环境的量化工具，能够显著提升模型在实际应用中的表现。

---

## 共享内存大小设置的依据

设置Docker容器共享内存大小（`shm_size`）的依据主要基于以下几个因素：

### 1. 模型规模

- **大型模型需求**：Qwen-72B即使是Int4量化版本，仍是超大规模模型
- **参数数量**：模型越大，在GPU之间传输状态需要的缓冲区越大

### 2. GPU数量和通信模式

- **GPU数量**：使用的GPU越多，需要的共享内存越大
- **多GPU通信**：使用`count: all`时，所有GPU都需要相互通信
- **NCCL缓冲区**：每对GPU之间的通信需要专用缓冲区

### 3. NCCL要求

- **每个进程通信**：NCCL为每个并行进程分配缓冲区
- **中间状态存储**：tensor并行计算过程中的中间结果存储

### 4. 经验值和最佳实践

- **默认值不足**：Docker默认只有64MB共享内存
- **安全边界**：16GB是大型模型的常见推荐值
- **预留空间**：为突发需求预留足够空间

如果16GB仍然不够（继续出现共享内存错误），可以考虑增加到32GB或更高。共享内存设置过大基本没有负面影响，除了占用主机内存。