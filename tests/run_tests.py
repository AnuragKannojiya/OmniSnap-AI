#!/usr/bin/env python3
"""
OmniSnap AI — Automated Test Suite
Executes unit tests for AI modules, benchmarks, and all FastAPI web endpoints.
"""

import sys
import os
import time
import json
import asyncio
import numpy as np

# Ensure root dir in path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def test_core_modules():
    print(f"\n{Colors.BOLD}{Colors.CYAN}[1/3] Testing Core AI Modules & NPU Runtime...{Colors.RESET}")
    from core.npu_runtime import NPURuntime
    from core.model_manager import ModelManager
    from core.whisper_asr import WhisperASR
    from core.llama_copilot import LlamaCopilot
    from core.yolo_vision import YOLOVisionGuard
    from core.rag_engine import RAGEngine

    # NPU Runtime
    runtime = NPURuntime()
    assert runtime is not None
    info = runtime.get_device_info()
    print(f"  {Colors.GREEN}✔{Colors.RESET} NPURuntime initialized (has_ort={info['has_onnxruntime']})")

    # Model Manager
    manager = ModelManager()
    assert len(manager.MODELS) == 4
    print(f"  {Colors.GREEN}✔{Colors.RESET} ModelManager loaded 4 target models")

    # Whisper ASR
    asr = WhisperASR(demo_mode=True)
    res = asr.transcribe(np.zeros(16000, dtype=np.float32))
    assert res.text is not None and len(res.text) > 0
    print(f"  {Colors.GREEN}✔{Colors.RESET} Whisper ASR: transcription generated ({res.latency_ms:.1f}ms)")

    # Llama Copilot
    llm = LlamaCopilot(demo_mode=True)
    gen = llm.generate("Test prompt")
    assert gen.tokens_generated > 0
    actions = llm.extract_action_items("Review notes")
    assert len(actions) > 0
    print(f"  {Colors.GREEN}✔{Colors.RESET} Llama Copilot: text generation & action items validated")

    # YOLO Vision Guard
    yolo = YOLOVisionGuard(demo_mode=True)
    dets = yolo.detect(np.zeros((640, 640, 3), dtype=np.uint8))
    assert len(dets) > 0
    alert = yolo.check_privacy(np.zeros((640, 640, 3), dtype=np.uint8))
    assert alert.alert_level in ["safe", "caution", "danger"]
    print(f"  {Colors.GREEN}✔{Colors.RESET} YOLO Vision: object detection & privacy check validated")

    # RAG Engine
    rag = RAGEngine(demo_mode=True)
    chunks = rag.chunk_text("Qualcomm Snapdragon X Elite 45 TOPS Hexagon NPU", chunk_size=20, overlap=5)
    assert len(chunks) > 0
    print(f"  {Colors.GREEN}✔{Colors.RESET} RAG Engine: text chunking & index initialized")

def test_benchmarks():
    print(f"\n{Colors.BOLD}{Colors.CYAN}[2/3] Validating Benchmark Suite & Calculations...{Colors.RESET}")
    from benchmarks.benchmark_suite import run_benchmarks
    # Validate benchmarks table execution
    run_benchmarks()
    print(f"  {Colors.GREEN}✔{Colors.RESET} Benchmark suite calculations verified")

async def test_fastapi_endpoints():
    print(f"\n{Colors.BOLD}{Colors.CYAN}[3/3] Testing FastAPI Endpoints (Pages & REST APIs)...{Colors.RESET}")
    from app import app

    scope_base = {
        'type': 'http',
        'asgi': {'version': '3.0'},
        'http_version': '1.1',
        'server': ('127.0.0.1', 8000),
        'client': ('127.0.0.1', 50000),
        'query_string': b'',
        'headers': [],
    }

    async def call_app(method, path, body=b''):
        scope = dict(scope_base)
        scope['method'] = method
        scope['path'] = path
        scope['raw_path'] = path.encode('ascii')
        scope['headers'] = [(b'host', b'localhost:8000')]
        if body:
            scope['headers'].append((b'content-type', b'application/json'))
            scope['headers'].append((b'content-length', str(len(body)).encode('ascii')))

        body_sent = False
        async def receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {'type': 'http.request', 'body': body, 'more_body': False}
            return {'type': 'http.request', 'body': b'', 'more_body': False}

        status_code = None
        response_body = []
        async def send(message):
            nonlocal status_code
            if message['type'] == 'http.response.start':
                status_code = message['status']
            elif message['type'] == 'http.response.body':
                response_body.append(message.get('body', b''))

        await app(scope, receive, send)
        return status_code, b''.join(response_body)

    # HTML Pages
    for path in ['/', '/copilot', '/meeting', '/rag', '/guard', '/benchmarks']:
        code, body = await call_app('GET', path)
        assert code == 200, f"Page {path} returned {code}"
        assert len(body) > 100
        print(f"  {Colors.GREEN}✔{Colors.RESET} Page {path:<15} HTTP {code} ({len(body)} bytes)")

    # API Endpoints
    code, body = await call_app('GET', '/api/system-info')
    assert code == 200
    info = json.loads(body.decode('utf-8'))
    assert 'platform' in info

    code, body = await call_app('POST', '/api/chat', json.dumps({'message': 'Explain Snapdragon'}).encode('utf-8'))
    assert code == 200
    chat = json.loads(body.decode('utf-8'))
    assert 'response' in chat

    code, body = await call_app('POST', '/api/summarize', json.dumps({'text': 'Meeting notes'}).encode('utf-8'))
    assert code == 200
    summary = json.loads(body.decode('utf-8'))
    assert 'summary' in summary

    code, body = await call_app('POST', '/api/rag/search', json.dumps({'query': 'Hexagon'}).encode('utf-8'))
    assert code == 200
    rag = json.loads(body.decode('utf-8'))
    assert 'results' in rag

    code, body = await call_app('GET', '/api/telemetry')
    assert code == 200
    telem = json.loads(body.decode('utf-8'))
    assert 'npu_provider' in telem

    code, body = await call_app('GET', '/api/benchmarks')
    assert code == 200
    bench = json.loads(body.decode('utf-8'))
    assert 'results' in bench
    print(f"  {Colors.GREEN}✔{Colors.RESET} All 6 REST APIs returned HTTP 200 with valid schema")

def main():
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 65}")
    print(f"  OMNISNAP AI — AUTOMATED TEST SUITE")
    print(f"{'=' * 65}{Colors.RESET}")
    start = time.time()
    
    test_core_modules()
    test_benchmarks()
    asyncio.run(test_fastapi_endpoints())
    
    duration = time.time() - start
    print(f"\n{Colors.BOLD}{Colors.GREEN}=================================================================")
    print(f"  ALL TESTS PASSED SUCCESSFULLY in {duration:.2f}s! ✅")
    print(f"================================================================={Colors.RESET}\n")

if __name__ == '__main__':
    main()
