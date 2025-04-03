from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from server_vllm import ModelService
import logging
import json
import asyncio
from typing import Optional, AsyncIterable, Dict, Any, List
import uvicorn
import time

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Chat Service")
model_service: Optional[ModelService] = None

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源，生产环境请指定确切的源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    stream: bool = True
    max_tokens: Optional[int] = Field(default=4096, ge=1, le=8192, description="生成的最大token数")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0, description="top-p采样参数")

class ChatResponse(BaseModel):
    response: str
    usage: Optional[Dict[str, int]] = None

@app.on_event("startup")
async def startup_event():
    global model_service
    try:
        logger.info("正在启动模型服务...")
        model_service = ModelService()
        success = await model_service.load_model()
        if not success:
            logger.error("模型加载失败")
            raise Exception("模型加载失败")
        logger.info("模型服务启动成功")
    except Exception as e:
        logger.error(f"启动失败: {str(e)}")
        raise e

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    start_time = time.time()
    try:
        if not model_service:
            logger.error("服务未就绪")
            raise HTTPException(status_code=503, detail="服务未就绪")
        
        # 更新采样参数
        model_service.update_sampling_params(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # 如果请求流式响应，重定向到流式端点
        if request.stream:
            return await chat_stream_endpoint(request)
            
        response = await model_service.generate_response(request.message)
        
        # 计算耗时，仅用于日志
        elapsed = time.time() - start_time
        logger.info(f"非流式请求处理完成，耗时: {elapsed:.2f}秒")
        
        return ChatResponse(response=response)
    except Exception as e:
        logger.error(f"处理请求时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    start_time = time.time()
    try:
        if not model_service:
            logger.error("服务未就绪")
            raise HTTPException(status_code=503, detail="服务未就绪")
            
        # 更新采样参数
        model_service.update_sampling_params(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        async def generate_stream() -> AsyncIterable[str]:
            try:
                # 发送开始标志
                yield f"data: {json.dumps({'type': 'start'})}\n\n"
                
                full_response = ""
                token_count = 0
                
                # 流式生成文本
                async for text_chunk in model_service.generate_stream_response(request.message):
                    if text_chunk:
                        token_count += 1
                        full_response += text_chunk
                        yield f"data: {json.dumps({'type': 'content', 'content': text_chunk})}\n\n"
                        
                        # 添加周期性的心跳，防止连接超时
                        if token_count % 50 == 0:
                            yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                            await asyncio.sleep(0.01)  # 让出控制权
                
                # 计算耗时，用于日志和返回
                elapsed = time.time() - start_time
                logger.info(f"流式请求处理完成，生成 {token_count} 个tokens，耗时: {elapsed:.2f}秒")
                
                # 发送结束标志
                yield f"data: {json.dumps({'type': 'end', 'stats': {'tokens': token_count, 'time': elapsed}})}\n\n"
            except Exception as e:
                logger.error(f"生成流式响应时出错: {str(e)}")
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
                
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'  # 禁用Nginx的缓冲，确保流式传输
            }
        )
        
    except Exception as e:
        logger.error(f"处理流式请求时出错: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/health")
async def health_check():
    if model_service and model_service.model:
        return {"status": "healthy", "timestamp": time.time()}
    return {"status": "unhealthy", "timestamp": time.time()}

# 如果需要在此文件中直接运行服务
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)