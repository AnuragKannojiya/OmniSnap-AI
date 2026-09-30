import logging
import time
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Generator

logger = logging.getLogger(__name__)

__all__ = ['LlamaCopilot', 'GenerationResult', 'ActionItem']

# Try importing backends
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@dataclass
class GenerationResult:
    text: str
    tokens_generated: int
    latency_ms: float
    tokens_per_second: float
    backend: str = "demo"


@dataclass  
class ActionItem:
    task: str
    owner: str
    deadline: str
    priority: str


class LlamaCopilot:
    """Reasoning Copilot with multiple LLM backends.
    
    Backend priority:
    1. Ollama (if server running at localhost:11434)
    2. HuggingFace Transformers
    3. Demo mode (mock responses)

    Model Target:
    - Dev/Demo: SmolLM2-360M-Instruct (auto-downloads, works everywhere)
    - Production: Llama-3.2-3B-Instruct via Qualcomm AI Hub + QNN EP
    """
    
    OLLAMA_URL = "http://localhost:11434"
    DEFAULT_HF_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
    SYSTEM_PROMPT = """You are OmniSnap AI, an intelligent on-device copilot running on a Qualcomm Snapdragon Hexagon NPU. You are helpful, concise, and professional. All your processing happens 100% on-device with zero cloud dependency."""
    
    def __init__(self, model_id: str = None, backend: str = "auto", demo_mode: bool = False):
        self.demo_mode = demo_mode
        self.backend = "demo"
        self.model = None
        self.tokenizer = None
        self.ollama_model = "llama3.2:3b"  # Default Ollama model
        self._inference_count = 0
        self._total_latency = 0.0
        
        if demo_mode:
            logger.info("LlamaCopilot initialized in demo mode")
            return
        
        # Auto-detect best backend
        if backend == "auto":
            # 1. Try Ollama first
            if self._check_ollama():
                self.backend = "ollama"
                logger.info(f"✅ Using Ollama backend (model: {self.ollama_model})")
                return
            
            # 2. Try HuggingFace Transformers
            if HAS_TRANSFORMERS:
                try:
                    self._init_transformers(model_id or self.DEFAULT_HF_MODEL)
                    self.backend = "transformers"
                    logger.info(f"✅ Using Transformers backend (model: {model_id or self.DEFAULT_HF_MODEL})")
                    return
                except Exception as e:
                    logger.warning(f"Transformers init failed: {e}")
            
            # 3. Fall back to demo
            self.backend = "demo"
            self.demo_mode = True
            logger.info("Using demo mode (no LLM backend available)")
        elif backend == "ollama":
            if not self._check_ollama():
                raise RuntimeError("Ollama server not available")
            self.backend = "ollama"
        elif backend == "transformers":
            self._init_transformers(model_id or self.DEFAULT_HF_MODEL)
            self.backend = "transformers"
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is running locally."""
        if not HAS_REQUESTS:
            return False
        try:
            resp = requests.get(f"{self.OLLAMA_URL}/api/tags", timeout=2)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                logger.info(f"Ollama available with models: {models}")
                # Pick best available model
                preferred = ["llama3.2:3b", "llama3.2:1b", "llama3.1:8b", "phi3:mini", "mistral:7b", "qwen2.5:0.5b"]
                for pref in preferred:
                    if pref in models:
                        self.ollama_model = pref
                        return True
                # Use whatever is available
                if models:
                    self.ollama_model = models[0]
                    return True
            return False
        except Exception:
            return False
    
    def _init_transformers(self, model_id: str):
        """Initialize HuggingFace Transformers model."""
        logger.info(f"Loading HuggingFace model '{model_id}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info(f"Model loaded: {sum(p.numel() for p in self.model.parameters())/1e6:.0f}M params")
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> GenerationResult:
        """Generate a response to the prompt."""
        start = time.time()
        
        if self.backend == "ollama":
            text = self._generate_ollama(prompt, max_tokens, temperature)
        elif self.backend == "transformers":
            text = self._generate_transformers(prompt, max_tokens, temperature)
        else:
            text = self._generate_demo(prompt)
        
        latency = (time.time() - start) * 1000
        tokens = len(text.split())
        tps = tokens / max(latency / 1000, 0.001)
        self._inference_count += 1
        self._total_latency += latency
        
        return GenerationResult(
            text=text, tokens_generated=tokens,
            latency_ms=round(latency, 1), tokens_per_second=round(tps, 1),
            backend=self.backend
        )
    
    def _generate_ollama(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using Ollama API."""
        try:
            resp = requests.post(f"{self.OLLAMA_URL}/api/generate", json={
                "model": self.ollama_model,
                "prompt": prompt,
                "system": self.SYSTEM_PROMPT,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                }
            }, timeout=120)
            return resp.json().get("response", "[No response from Ollama]")
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return f"[Ollama error: {str(e)[:100]}]"
    
    def _generate_transformers(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using HuggingFace Transformers."""
        try:
            # Format with chat template if available
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            try:
                input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                input_text = f"System: {self.SYSTEM_PROMPT}\n\nUser: {prompt}\n\nAssistant:"
            
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            # Decode only the generated tokens (not the input)
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            return response
        except Exception as e:
            logger.error(f"Transformers generation error: {e}")
            return f"[Generation error: {str(e)[:100]}]"
    
    def _generate_demo(self, prompt: str) -> str:
        """Generate demo responses."""
        # ... (keep rich demo responses for various prompt types)
        prompt_lower = prompt.lower()
        if "summarize" in prompt_lower or "summary" in prompt_lower:
            return "Here's a concise summary: The document outlines key strategies for optimizing on-device AI inference using the Qualcomm Hexagon NPU, achieving 7x speedup over CPU with 90% power savings."
        elif "email" in prompt_lower or "draft" in prompt_lower:
            return "Subject: AI Initiative Update\n\nDear Team,\n\nOur on-device AI deployment using Snapdragon Hexagon NPU has exceeded expectations. Key metrics: 7.2x speedup, 90% power reduction, 100% privacy compliance.\n\nBest regards"
        elif "code" in prompt_lower or "explain" in prompt_lower:
            return "This code initializes an ONNX Runtime session with the QNN Execution Provider, targeting the Hexagon NPU. The provider chain ensures graceful fallback across hardware."
        elif "action" in prompt_lower or "item" in prompt_lower or "task" in prompt_lower:
            return "Action Items:\n1. [HIGH] Finalize Q3 performance report - Analytics Team - Due: Next Friday\n2. [HIGH] Schedule Snapdragon NPU demo - Engineering Lead - Due: This Week\n3. [MEDIUM] Draft board presentation - Product Manager - Due: Oct 15"
        else:
            return f"I've analyzed your request using on-device AI. Here's my response:\n\n{prompt[:200]}\n\nThis was processed entirely on-device with zero cloud dependency."
    
    def stream_generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> Generator[str, None, None]:
        """Stream-generate tokens one at a time (for WebSocket streaming)."""
        if self.backend == "ollama":
            yield from self._stream_ollama(prompt, max_tokens, temperature)
        elif self.backend == "transformers":
            yield from self._stream_transformers(prompt, max_tokens, temperature)
        else:
            # Demo: yield words one at a time
            for word in self._generate_demo(prompt).split():
                yield word + " "
    
    def _stream_ollama(self, prompt, max_tokens, temperature):
        try:
            resp = requests.post(f"{self.OLLAMA_URL}/api/generate", json={
                "model": self.ollama_model, "prompt": prompt,
                "system": self.SYSTEM_PROMPT, "stream": True,
                "options": {"num_predict": max_tokens, "temperature": temperature}
            }, stream=True, timeout=120)
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    if data.get("done", False):
                        break
        except Exception as e:
            yield f"[Ollama stream error: {e}]"
    
    def _stream_transformers(self, prompt, max_tokens, temperature):
        from threading import Thread
        try:
            messages = [{"role": "system", "content": self.SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
            try:
                input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                input_text = f"System: {self.SYSTEM_PROMPT}\nUser: {prompt}\nAssistant:"
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True, skip_prompt=True)
            gen_kwargs = {**inputs, "max_new_tokens": max_tokens, "temperature": temperature, "do_sample": temperature > 0, "streamer": streamer, "pad_token_id": self.tokenizer.pad_token_id}
            thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
            thread.start()
            for text in streamer:
                yield text
            thread.join()
        except Exception as e:
            yield f"[Stream error: {e}]"
    
    def summarize(self, text: str) -> str:
        result = self.generate(f"Summarize the following text concisely:\n\n{text[:2000]}", max_tokens=200)
        return result.text
    
    def extract_action_items(self, transcript: str) -> List[ActionItem]:
        result = self.generate(f"Extract action items from this meeting transcript. For each item, provide: task, owner, deadline, priority (High/Medium/Low). Format as a numbered list.\n\nTranscript:\n{transcript[:2000]}", max_tokens=300)
        # Parse the response into ActionItems
        items = []
        for line in result.text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('*')):
                items.append(ActionItem(task=line.lstrip('0123456789.-*) '), owner="Unassigned", deadline="TBD", priority="Medium"))
        if not items:
            items.append(ActionItem(task=result.text[:200], owner="Unassigned", deadline="TBD", priority="Medium"))
        return items
    
    def draft_response(self, context: str, instruction: str) -> str:
        result = self.generate(f"Context: {context[:1000]}\n\nInstruction: {instruction}", max_tokens=300)
        return result.text
