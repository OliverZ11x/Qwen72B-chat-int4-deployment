import os
from openai import OpenAI

client = OpenAI(
    base_url="http://172.32.1.163:8000/v1",
    api_key="token-abc123",
)

stream = client.chat.completions.create(
    model="/Qwen/Qwen3-14B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁？"}
    ],
    extra_body={"chat_template_kwargs": {"enable_thinking": False},
                "stream_options": {"include_usage": True,
                                   "continuous_usage_stats": True}},
    stream=True,
)

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
