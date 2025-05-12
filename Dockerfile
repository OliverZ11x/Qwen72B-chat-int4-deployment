FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install --upgrade setuptools
RUN pip3 install -v gptqmodel --no-build-isolation

COPY src/ src/

# 环境变量
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0,1,2,3


# 运行服务
# CMD ["python3", "src/openai_server.py"]

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"， "--access-logfile", "/app/logs/access.log"]