#!/usr/bin/env python3
"""
OmniSnap AI — Demo Runner & System Verification
Snapdragon® AI Lab Build & Present Challenge

Usage:
    python run_demo.py                  # Full system check + launch server
    python run_demo.py --test-only      # Run checks only
    python run_demo.py --demo           # Force demo mode (no real models)
    python run_demo.py --port 8080      # Launch on custom port
"""

import os
import sys
import argparse
import platform
import time
import subprocess
from datetime import datetime

# ANSI colors
class C:
    CYAN = '\033[96m'; GREEN = '\033[92m'; YELLOW = '\033[93m'; RED = '\033[91m'
    BLUE = '\033[94m'; BOLD = '\033[1m'; DIM = '\033[2m'; RESET = '\033[0m'


def banner():
    print(f"""{C.CYAN}{C.BOLD}
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   ⚡  OmniSnap AI  —  Zero-Cloud Multimodal Copilot          ║
    ║                                                               ║
    ║   Powered by Qualcomm® Snapdragon® X Elite Hexagon NPU       ║
    ║   REAL End-to-End AI Inference                                ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    {C.RESET}""")


def check_system():
    print(f"\n{C.BOLD}{'═' * 60}")
    print(f"  SYSTEM CHECK")
    print(f"{'═' * 60}{C.RESET}\n")

    plat = f"{platform.system()} {platform.release()}"
    machine = platform.machine()
    print(f"  {C.BLUE}Platform:{C.RESET}    {plat}")
    print(f"  {C.BLUE}Machine:{C.RESET}     {machine}")
    print(f"  {C.BLUE}Python:{C.RESET}      {sys.version.split(' ')[0]}")

    is_arm = machine.lower() in ('aarch64', 'arm64')
    if is_arm:
        print(f"  {C.GREEN}✅ ARM64 — Snapdragon compatible{C.RESET}")
    else:
        print(f"  {C.YELLOW}⚠️  Non-ARM ({machine}) — CPU fallback{C.RESET}")

    print()

    # PyTorch
    try:
        import torch
        device = "MPS" if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else ("CUDA" if torch.cuda.is_available() else "CPU")
        print(f"  {C.GREEN}✅ PyTorch {torch.__version__} ({device}){C.RESET}")
    except ImportError:
        print(f"  {C.RED}❌ PyTorch not installed{C.RESET}")

    # Transformers
    try:
        import transformers
        print(f"  {C.GREEN}✅ Transformers {transformers.__version__}{C.RESET}")
    except ImportError:
        print(f"  {C.RED}❌ Transformers not installed{C.RESET}")

    # Sentence Transformers
    try:
        import sentence_transformers
        print(f"  {C.GREEN}✅ Sentence-Transformers {sentence_transformers.__version__}{C.RESET}")
    except ImportError:
        print(f"  {C.RED}❌ Sentence-Transformers not installed{C.RESET}")

    # Ultralytics
    try:
        import ultralytics
        print(f"  {C.GREEN}✅ Ultralytics {ultralytics.__version__}{C.RESET}")
    except ImportError:
        print(f"  {C.RED}❌ Ultralytics not installed{C.RESET}")

    # ONNX Runtime
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        has_qnn = 'QnnExecutionProvider' in providers
        print(f"  {C.GREEN}✅ ONNX Runtime {ort.__version__} ({', '.join(providers[:3])}){C.RESET}")
        if has_qnn:
            print(f"  {C.GREEN}✅ QNN Execution Provider — Hexagon NPU READY!{C.RESET}")
    except ImportError:
        print(f"  {C.YELLOW}⚠️  ONNX Runtime not installed{C.RESET}")

    # FastAPI
    try:
        import fastapi
        print(f"  {C.GREEN}✅ FastAPI {fastapi.__version__}{C.RESET}")
    except ImportError:
        print(f"  {C.RED}❌ FastAPI not installed{C.RESET}")

    print()


def test_modules(demo_mode=False):
    import numpy as np

    print(f"\n{C.BOLD}{'═' * 60}")
    print(f"  AI MODULE TESTS ({'demo mode' if demo_mode else 'REAL INFERENCE'})")
    print(f"{'═' * 60}{C.RESET}\n")

    results = []

    # YOLO
    print(f"  {C.BOLD}YOLOv11-Nano Vision{C.RESET}")
    try:
        from core.yolo_vision import YOLOVisionGuard
        t0 = time.time()
        yolo = YOLOVisionGuard(demo_mode=demo_mode)
        load_time = time.time() - t0
        print(f"    Loaded in {load_time:.1f}s (demo={yolo.demo_mode})")
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        t0 = time.time()
        dets = yolo.detect(img)
        lat = (time.time()-t0)*1000
        print(f"    {C.GREEN}✅ {len(dets)} detections in {lat:.0f}ms{C.RESET}")
        results.append(("YOLOv11-Nano", True))
    except Exception as e:
        print(f"    {C.RED}❌ {e}{C.RESET}")
        results.append(("YOLOv11-Nano", False))

    print()

    # RAG / Embeddings
    print(f"  {C.BOLD}MiniLM-L6-v2 (RAG Embeddings){C.RESET}")
    try:
        from core.rag_engine import RAGEngine
        t0 = time.time()
        rag = RAGEngine(demo_mode=demo_mode)
        load_time = time.time() - t0
        print(f"    Loaded in {load_time:.1f}s (demo={rag.demo_mode})")
        embs = rag.embed(["Hello world", "Test query"])
        print(f"    {C.GREEN}✅ Embeddings: shape={embs.shape}{C.RESET}")
        results.append(("MiniLM-L6-v2", True))
    except Exception as e:
        print(f"    {C.RED}❌ {e}{C.RESET}")
        results.append(("MiniLM-L6-v2", False))

    print()

    # Whisper
    print(f"  {C.BOLD}Whisper-Base EN (ASR){C.RESET}")
    try:
        from core.whisper_asr import WhisperASR
        t0 = time.time()
        whisper = WhisperASR(demo_mode=demo_mode)
        load_time = time.time() - t0
        print(f"    Loaded in {load_time:.1f}s (demo={whisper.demo_mode})")
        audio = np.random.randn(16000).astype(np.float32) * 0.01
        t0 = time.time()
        result = whisper.transcribe(audio)
        lat = (time.time()-t0)*1000
        print(f"    {C.GREEN}✅ \"{result.text[:60]}\" ({lat:.0f}ms){C.RESET}")
        results.append(("Whisper-Base", True))
    except Exception as e:
        print(f"    {C.RED}❌ {e}{C.RESET}")
        results.append(("Whisper-Base", False))

    print()

    # LLM
    print(f"  {C.BOLD}LLM Copilot{C.RESET}")
    try:
        from core.llama_copilot import LlamaCopilot
        t0 = time.time()
        llm = LlamaCopilot(demo_mode=demo_mode)
        load_time = time.time() - t0
        print(f"    Loaded in {load_time:.1f}s (backend={llm.backend})")
        t0 = time.time()
        result = llm.generate("What is 2+2?", max_tokens=50)
        lat = (time.time()-t0)*1000
        print(f"    {C.GREEN}✅ \"{result.text[:80]}\" ({lat:.0f}ms){C.RESET}")
        results.append(("LLM Copilot", True))
    except Exception as e:
        print(f"    {C.RED}❌ {e}{C.RESET}")
        results.append(("LLM Copilot", False))

    print()
    passed = sum(1 for _, ok in results if ok)
    print(f"  {C.BOLD}{passed}/{len(results)} modules passed{C.RESET}")
    return results


def main():
    parser = argparse.ArgumentParser(description="OmniSnap AI — Demo Runner")
    parser.add_argument("--test-only", action="store_true", help="Run checks only")
    parser.add_argument("--demo", action="store_true", help="Force demo mode")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    args = parser.parse_args()

    banner()
    print(f"  {C.DIM}Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{C.RESET}")

    check_system()
    test_modules(demo_mode=args.demo)

    if args.test_only:
        print(f"\n{C.GREEN}{C.BOLD}  ✅ Test complete!{C.RESET}\n")
        return

    print(f"\n{C.BOLD}{'═' * 60}")
    print(f"  LAUNCHING SERVER")
    print(f"{'═' * 60}{C.RESET}\n")
    print(f"  {C.GREEN}►{C.RESET} http://localhost:{args.port}/")
    print(f"  {C.GREEN}►{C.RESET} Press Ctrl+C to stop\n")

    try:
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "app:app", "--host", args.host, "--port", str(args.port), "--reload"],
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
    except KeyboardInterrupt:
        print(f"\n  {C.YELLOW}🛑 Server stopped.{C.RESET}\n")


if __name__ == "__main__":
    main()
