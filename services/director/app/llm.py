"""Thin async client for Ollama: pull the model if needed, keep it warm, ask for JSON."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time

import httpx

log = logging.getLogger("director.llm")

THINKING_MODELS = re.compile(r"qwen3|qwq|deepseek-r1|magistral|thinking|gpt-oss", re.I)


class Ollama:
    def __init__(self, base_url: str, model: str, keep_alive: str = "2h", num_ctx: int = 4096,
                 temperature: float = 0.8, timeout: float = 90.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout = timeout
        self.state = "connecting"      # connecting | pulling | loading | ready | error
        self.detail = f"waiting for Ollama at {self.base_url}"
        self.progress = 0.0
        self.last_latency = None
        self.last_tokens_per_s = None
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=5.0))

    async def close(self):
        await self._client.aclose()

    # ------------------------------------------------------------------ lifecycle
    async def prepare(self) -> None:
        """Runs in the background at startup: wait for Ollama, pull the model, load it."""
        while True:
            try:
                r = await self._client.get(f"{self.base_url}/api/version")
                r.raise_for_status()
                break
            except httpx.HTTPError:
                self.state, self.detail = "connecting", f"waiting for Ollama at {self.base_url}"
                await asyncio.sleep(3)
        while True:
            try:
                if not await self._has_model():
                    await self._pull()
                self.state, self.detail = "loading", f"loading {self.model} into the GPU"
                await self._client.post(f"{self.base_url}/api/generate",
                                        json={"model": self.model, "prompt": "",
                                              "keep_alive": self.keep_alive},
                                        timeout=600)
                self.state, self.detail = "ready", self.model
                return
            except Exception as e:  # noqa: BLE001 - report and retry, never die
                self.state, self.detail = "error", f"{type(e).__name__}: {e}"[:300]
                log.warning("model preparation failed: %s", self.detail)
                await asyncio.sleep(10)

    async def _has_model(self) -> bool:
        r = await self._client.post(f"{self.base_url}/api/show", json={"model": self.model})
        return r.status_code == 200

    async def _pull(self) -> None:
        self.state, self.detail, self.progress = "pulling", f"downloading {self.model}", 0.0
        async with self._client.stream("POST", f"{self.base_url}/api/pull",
                                       json={"model": self.model, "stream": True},
                                       timeout=None) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                msg = json.loads(line)
                if msg.get("error"):
                    raise RuntimeError(msg["error"])
                total, done = msg.get("total"), msg.get("completed")
                if total and done:
                    self.progress = round(100.0 * done / total, 1)
                    self.detail = f"downloading {self.model}: {self.progress}%"
                if msg.get("status") == "success":
                    self.progress = 100.0

    # ------------------------------------------------------------------ inference
    async def chat_json(self, messages: list[dict], schema: dict, max_tokens: int = 420) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "format": schema,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "top_p": 0.92,
                "repeat_penalty": 1.08,
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
            },
        }
        if THINKING_MODELS.search(self.model):
            body["think"] = False
        t0 = time.perf_counter()
        r = await self._client.post(f"{self.base_url}/api/chat", json=body)
        r.raise_for_status()
        data = r.json()
        self.last_latency = round(time.perf_counter() - t0, 2)
        if data.get("eval_count") and data.get("eval_duration"):
            self.last_tokens_per_s = round(data["eval_count"] / (data["eval_duration"] / 1e9), 1)
        content = (data.get("message") or {}).get("content", "")
        return parse_json_loose(content)


def parse_json_loose(text: str) -> dict:
    """Structured outputs should give clean JSON; this still copes with stray text."""
    text = (text or "").strip()
    try:
        out = json.loads(text)
        return out if isinstance(out, dict) else {}
    except ValueError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            out = json.loads(text[start:end + 1])
            return out if isinstance(out, dict) else {}
        except ValueError:
            return {}
    return {}
