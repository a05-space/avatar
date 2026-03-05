import os
import base64
import asyncio
import tempfile
from contextlib import asynccontextmanager

import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, StreamingResponse

from models.utils import TTSRequest, LLMRequest
from utils.app_state import load_config_and_model, get_config, get_infer_engine, get_tts_model
from utils.infer import _process_audio_inference, _process_text2audio_task, _process_text2avatar_task, _process_llm_inference


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_config_and_model()
    yield

app = FastAPI(title="经纬", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


@app.post("/api/audio2avatar")
async def audio2avatar(file: UploadFile = File(...)) -> dict:
    """上传整段音频，返回对应的表情动画 JSON 及音频 Base64。"""

    cfg = get_config()
    infer_engine = get_infer_engine()

    raw_data = await file.read()
    original_filename = file.filename or ""
    ext = os.path.splitext(original_filename)[1].lower()
    
    if not ext:
        mime_type = (file.content_type or "").lower()
        if "mpeg" in mime_type:
            ext = ".mp3"
        elif "flac" in mime_type:
            ext = ".flac"
        else:
            ext = ".wav"

    audio_b64 = base64.b64encode(raw_data).decode("ascii")
    mime_type_for_b64 = file.content_type or "audio/wav"
    audio_base64 = f"data:{mime_type_for_b64};base64,{audio_b64}"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(raw_data)
            tmp_path = tmp.name
        
        loop = asyncio.get_running_loop()
        animation_json = await loop.run_in_executor(
            None, 
            _process_audio_inference, 
            tmp_path, 
            cfg, 
            infer_engine
        )   
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {
        "expression": animation_json,
        "audioBase64": audio_base64,
    }


@app.post("/api/text2audio")
async def text2audio(request: TTSRequest) -> dict:
    tts_model = get_tts_model()
        
    loop = asyncio.get_running_loop()
    sentences, times, uuid = await loop.run_in_executor(
        None,
        _process_text2audio_task,
        tts_model,
        request.text,
        request.language,
        request.ref_audio,
        request.ref_text,
    )

    return {
        "sentences": sentences,
        "times": times,
        "uuid": uuid,
    }


@app.post("/api/text2avatar")
async def text2avatar(request: TTSRequest) -> dict:
    tts_model = get_tts_model()
    cfg = get_config()
    infer_engine = get_infer_engine()

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        _process_text2avatar_task,
        tts_model,
        cfg,
        infer_engine,
        request.text,
        request.language,
        request.ref_audio,
        request.ref_text,
    )

    return result


@app.post("/api/llm2text")
async def llm2text(request: LLMRequest):
    response = await _process_llm_inference(request)
    
    if request.stream:
        async def generate():
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        return {"content": response.choices[0].message.content}


@app.post("/api/llm2audio")
async def llm2audio(request: LLMRequest):
    if request.stream:
        req_dict = request.model_dump()
        req_dict["stream"] = False
        request_non_stream = LLMRequest(**req_dict)
    else:
        request_non_stream = request

    response = await _process_llm_inference(request_non_stream)
    content = response.choices[0].message.content if response and response.choices else ""
    if not content or not content.strip():
        return JSONResponse(status_code=400, content={"detail": "LLM 返回内容为空，无法进行语音合成"})

    tts_model = get_tts_model()
    loop = asyncio.get_running_loop()
    sentences, times, uuid = await loop.run_in_executor(
        None,
        _process_text2audio_task,
        tts_model,
        content,
        TTSRequest.language,
        TTSRequest.ref_audio,
        TTSRequest.ref_text,
    )

    return {
        "sentences": sentences,
        "times": times,
        "uuid": uuid,
    }


@app.post("/api/llm2avatar")
async def llm2avatar(request: LLMRequest):
    pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8109)
