sudo docker run -d --runtime nvidia --gpus all \
-v /home/ubuntu/.cache/modelscope/hub/models/Qwen/Qwen3-14B:/Qwen/Qwen3-14B \
-v /home/ubuntu/code/Qwen72B-chat-int4-deployment/chat_template:/chat_template \
-p 8000:8000 \
--ipc=host \
--log-driver json-file \
--log-opt max-size=10m \
--log-opt max-file=3 \
vllm/vllm-openai:latest \
--model /Qwen/Qwen3-14B \
--tensor-parallel-size 4 \
--api-key token-abc123 \
--dtype 'float16' \
--port 8000 \
--chat-template /chat_template/template_custom.jinja \
--no-enable-chunked-prefill \
--rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
--max-model-len 131072