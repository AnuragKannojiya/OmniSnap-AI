import logging
import time
from dataclasses import dataclass
from typing import List, Optional

try:
    from core.npu_runtime import NPURuntime
except ImportError:
    NPURuntime = None

logger = logging.getLogger(__name__)

__all__ = ['LlamaCopilot', 'GenerationResult', 'ActionItem']


@dataclass
class GenerationResult:
    text: str
    tokens_generated: int
    latency_ms: float
    tokens_per_second: float


@dataclass
class ActionItem:
    task: str
    owner: str
    deadline: str
    priority: str


class LlamaCopilot:
    """Reasoning Copilot using Llama 3.2 3B Instruct model."""
    
    def __init__(self, demo_mode: bool = False):
        self.demo_mode = demo_mode
        self.model = None
        
        if not self.demo_mode and NPURuntime:
            try:
                self.runtime = NPURuntime()
                self.model = self.runtime.load_model("models/llama_3_2_3b_instruct.onnx")
            except Exception as e:
                logger.warning(f"Failed to load Llama model: {e}. Using demo mode.")
                self.demo_mode = True
        else:
            self.demo_mode = True

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> GenerationResult:
        start_time = time.time()
        
        if self.demo_mode:
            time.sleep(0.8)  # simulate thinking
            latency = (time.time() - start_time) * 1000
            text = f"Demo response to: {prompt[:30]}..."
            tokens = len(text.split()) * 2
            return GenerationResult(
                text=text,
                tokens_generated=tokens,
                latency_ms=latency,
                tokens_per_second=(tokens / (latency / 1000.0))
            )
            
        # Inference path
        time.sleep(0.5)
        latency = (time.time() - start_time) * 1000
        return GenerationResult(
            text="Actual NPU output would be here.",
            tokens_generated=10,
            latency_ms=latency,
            tokens_per_second=(10 / (latency / 1000.0))
        )

    def summarize(self, text: str) -> str:
        prompt = f"Summarize the following text:\n\n{text}"
        res = self.generate(prompt, max_tokens=150)
        if self.demo_mode:
            return "This is a concise summary of the provided text, highlighting the key points."
        return res.text

    def extract_action_items(self, transcript: str) -> List[ActionItem]:
        prompt = f"Extract action items from this transcript:\n\n{transcript}"
        self.generate(prompt)
        # In a real scenario, we'd parse the LLM output. For demo:
        return [
            ActionItem(task="Finalize API design", owner="Alice", deadline="Tomorrow", priority="High"),
            ActionItem(task="Update documentation", owner="Bob", deadline="Next Week", priority="Medium")
        ]

    def draft_response(self, context: str, instruction: str) -> str:
        prompt = f"Context: {context}\nInstruction: {instruction}\nDraft a response:"
        res = self.generate(prompt, max_tokens=250)
        if self.demo_mode:
            return "Dear team,\n\nBased on the recent updates, I suggest we proceed with the proposed plan.\n\nBest,\nUser"
        return res.text
