from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.server_vllm import ModelService
import logging
from typing import Optional, AsyncIterable
import uvicorn
import asyncio

app = FastAPI(title="AI Chat Service")
model_service: Optional[ModelService] = None
logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    message: list[dict]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    
class ChatResponse(BaseModel):
    response: str

@app.on_event("startup")
async def startup_event():
    global model_service
    model_service = ModelService()
    success = await model_service.load_model()
    if not success:
        raise Exception("模型加载失败")

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        if not model_service:
            raise HTTPException(status_code=503, detail="服务未就绪")
        
        # 更新采样参数
        model_service.update_sampling_params(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        response = await model_service.generate_response(request.message)
        return ChatResponse(response=response)
    except Exception as e:
        logger.error(f"处理请求时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    try:
        if not model_service:
            raise HTTPException(status_code=503, detail="服务未就绪")
            
        async def generate_stream() -> AsyncIterable[str]:
            async for text_chunk in model_service.generate_stream_response(request.message):
                yield f"data: {text_chunk}\n\n"
                
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        logger.error(f"处理流式请求时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    if model_service and model_service.model:
        return {"status": "healthy"}
    return {"status": "unhealthy"}