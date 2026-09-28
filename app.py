"""
OmniSnap AI — Main Application Server
Qualcomm Snapdragon® HP OmniBook Copilot & Edge AI Platform.
"""

import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional

from core.npu_engine import npu_engine
from core.speech_transcriber import speech_transcriber
from core.reasoning_llm import reasoning_llm
from core.document_rag import document_rag
from core.vision_engine import vision_engine
from core.privacy_shield import privacy_shield
from core.telemetry import telemetry_engine

app = FastAPI(
    title="OmniSnap AI",
    description="Zero-Cloud Autonomous Multimodal Copilot for Snapdragon HP OmniBook PCs",
    version="1.0.0",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Mount static files if present
static_dir = os.path.join(BASE_DIR, "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Pydantic Request Models
class ReasonRequest(BaseModel):
    prompt: str
    system_context: Optional[str] = None
    rag_context: Optional[str] = None


class SpeechRequest(BaseModel):
    scenario: Optional[str] = "product_sync"


class RAGSearchRequest(BaseModel):
    query: str


class RAGIndexRequest(BaseModel):
    title: str
    content: str
    category: Optional[str] = "User Uploaded"


class VisionRequest(BaseModel):
    frame_type: Optional[str] = "workspace_screen"


class PrivacySanitizeRequest(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    """Render main application dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/status")
async def get_system_status():
    """Hardware health and Qualcomm Hexagon NPU status."""
    return npu_engine.get_system_status()


@app.post("/api/copilot/reason")
async def copilot_reason(req: ReasonRequest):
    """Generate reasoning response on Hexagon NPU."""
    # If rag_context is not provided, automatically search local vector store
    rag_ctx = req.rag_context
    if not rag_ctx:
        hits = document_rag.search_similar(req.prompt, top_k=1)
        if hits and hits[0]["score"] > 0.70:
            rag_ctx = hits[0]["content"]

    res = reasoning_llm.generate_response(
        prompt=req.prompt,
        system_context=req.system_context,
        rag_context=rag_ctx,
    )
    return res


@app.post("/api/speech/transcribe")
async def speech_transcribe(req: SpeechRequest):
    """Transcribe audio on Whisper Hexagon NPU."""
    res = speech_transcriber.transcribe_audio_chunk(mock_scenario=req.scenario)
    return res


@app.get("/api/rag/documents")
async def get_rag_documents():
    """List indexed local documents."""
    return document_rag.get_all_documents()


@app.post("/api/rag/search")
async def search_rag(req: RAGSearchRequest):
    """Semantic vector search on local documents."""
    return document_rag.search_similar(req.query)


@app.post("/api/rag/index")
async def index_rag_document(req: RAGIndexRequest):
    """Index a new document into local vector database."""
    return document_rag.index_document(
        title=req.title,
        content=req.content,
        category=req.category or "User Uploaded",
    )


@app.post("/api/vision/analyze")
async def analyze_vision(req: VisionRequest):
    """Analyze display or document with YOLOv11 on Hexagon NPU."""
    return vision_engine.analyze_screen_frame(frame_type=req.frame_type or "workspace_screen")


@app.post("/api/privacy/sanitize")
async def sanitize_privacy(req: PrivacySanitizeRequest):
    """Sanitize confidential PII on-device."""
    return privacy_shield.inspect_and_sanitize(text=req.text)


@app.get("/api/telemetry/live")
async def get_live_telemetry():
    """Live NPU telemetry, power draw, and cloud cost metrics."""
    return telemetry_engine.get_live_metrics()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 Launching OmniSnap AI on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
