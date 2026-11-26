"""Thin async client for Ollama: pull the model if needed, keep it warm, ask for JSON."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time

import httpx

log = logging.getLogger("director.llm")

THINKING_MODELS = re.compile(r"qwen3|qwq|deepseek-r1|magistral|thinking|gpt-oss|gemma4|dictalm", re.I)


class Ollama:
    def __init__(self, base_url: str, model: str, keep_alive: str = "2h", num_ctx: int = 4096,
                 temperature: float = 0.8, timeout: float = 90.0, num_thread: int = 0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        # Ollama reads a bare number as seconds (negative: keep loaded); "2h" style strings too.
        try:
            self.keep_alive = int(keep_alive)
        except (TypeError, ValueError):
            self.keep_alive = keep_alive
        self.num_ctx = num_ctx
        # CPU threads for the parts of a model that run on the processor (0: Ollama decides).
        self.num_thread = num_thread
        self.temperature = temperature
        self.timeout = timeout
        self.state = "connecting"      # connecting | pulling | loading | ready | error
        self.detail = f"waiting for Ollama at {self.base_url}"
        self.progress = 0.0
        self.last_latency = None
        self.last_tokens_per_s = None
        self.last_prompt_tokens_per_s = None
        self.side_pulls: dict[str, dict] = {}     # model -> {"state", "progress", "detail"}
        self.hold_until = 0.0                     # the watchdog leaves Ollama alone until then
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=5.0))

    async def close(self):
        await self._client.aclose()

    # ------------------------------------------------------------------ lifecycle
    async def prepare(self, warmup_messages: list[dict] | None = None,
                      warmup_schema: dict | None = None) -> None:
        """Runs in the background at startup: wait for Ollama, pull the model, load it, and
        (optionally) run one tiny chat so the fixed prompt prefix is already cached."""
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
                # Same context size as every turn: a different one makes Ollama reload the model.
                await self._client.post(f"{self.base_url}/api/generate",
                                        json={"model": self.model, "prompt": "",
                                              "keep_alive": self.keep_alive,
                                              "options": self._load_options()},
                                        timeout=600)
                if warmup_messages:
                    self.detail = f"warming up {self.model}"
                    t0 = time.perf_counter()
                    # Exactly the request a turn makes (answer format and all), so the
                    # prompt Ollama caches is the one the first turn starts with.
                    body = self._body(warmup_messages, warmup_schema, max_tokens=1, temperature=None)
                    await self._client.post(f"{self.base_url}/api/chat", timeout=600, json=body)
                    log.info("prompt prefix cached in %.1fs", time.perf_counter() - t0)
                self.state, self.detail = "ready", self.model
                return
            except Exception as e:  # noqa: BLE001 - report and retry, never die
                self.state, self.detail = "error", f"{type(e).__name__}: {e}"[:300]
                log.warning("model preparation failed: %s", self.detail)
                await asyncio.sleep(10)

    async def watch(self, warmup_messages: list[dict] | None = None,
                    warmup_schema: dict | None = None, every: float = 10.0) -> None:
        """Prepare, then keep watching: if Ollama restarts or drops the model, load and warm
        it again before the next turn arrives, and say so in the heartbeat meanwhile."""
        await self.prepare(warmup_messages, warmup_schema)
        while True:
            await asyncio.sleep(every)
            if time.time() < self.hold_until:     # a model trial is using the GPU
                continue
            try:
                r = await self._client.get(f"{self.base_url}/api/ps", timeout=5)
                if r.status_code == 404:          # an Ollama too old to list loaded models
                    log.info("Ollama has no /api/ps; the watchdog stops here")
                    return
                r.raise_for_status()
                loaded = {m.get("name") for m in r.json().get("models", [])} | \
                         {m.get("model") for m in r.json().get("models", [])}
                if self.model in loaded or f"{self.model}:latest" in loaded:
                    continue
                log.warning("%s is no longer loaded in Ollama; loading it again", self.model)
            except Exception as e:  # noqa: BLE001
                log.warning("Ollama unreachable (%s); waiting for it", type(e).__name__)
            await self.prepare(warmup_messages, warmup_schema)

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

    async def pull_on_the_side(self, model: str) -> None:
        """Download another model while the current one keeps serving (for trials)."""
        rec = self.side_pulls[model] = {"state": "pulling", "progress": 0.0, "detail": "starting"}
        try:
            async with self._client.stream("POST", f"{self.base_url}/api/pull",
                                           json={"model": model, "stream": True}, timeout=None) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    msg = json.loads(line)
                    if msg.get("error"):
                        raise RuntimeError(msg["error"])
                    rec["detail"] = msg.get("status", "")
                    total, done = msg.get("total"), msg.get("completed")
                    if total and done:
                        rec["progress"] = round(100.0 * done / total, 1)
                        rec["detail"] = f"{msg.get('status', '')} {rec['progress']}% of {total / 1e9:.1f} GB"
                    if msg.get("status") == "success":
                        rec["state"], rec["progress"] = "done", 100.0
        except Exception as e:  # noqa: BLE001
            rec["state"], rec["detail"] = "error", f"{type(e).__name__}: {e}"[:300]

    async def unload(self, model: str) -> None:
        """Ask Ollama to drop a model from memory now."""
        r = await self._client.post(f"{self.base_url}/api/generate",
                                    json={"model": model, "keep_alive": 0}, timeout=30)
        r.raise_for_status()

    async def loaded(self) -> list[dict]:
        r = await self._client.get(f"{self.base_url}/api/ps", timeout=5)
        r.raise_for_status()
        return r.json().get("models", [])

    # ------------------------------------------------------------------ inference
    def _load_options(self) -> dict:
        """Options that decide how Ollama loads the model; every request repeats them, since a
        different value makes Ollama load the model again."""
        opts = {"num_ctx": self.num_ctx}
        if self.num_thread:
            opts["num_thread"] = self.num_thread
        return opts

    def _body(self, messages: list[dict], schema: dict, max_tokens: int,
              temperature: float | None, stream: bool = False, model: str | None = None) -> dict:
        model = model or self.model
        body = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "top_p": 0.92,
                "repeat_penalty": 1.08,
                **self._load_options(),
                "num_predict": max_tokens,
            },
        }
        if schema is not None:
            body["format"] = schema
        if THINKING_MODELS.search(model):
            body["think"] = False
        return body

    def _measure(self, data: dict, t0: float) -> None:
        self.last_latency = round(time.perf_counter() - t0, 2)
        if data.get("eval_count") and data.get("eval_duration"):
            self.last_tokens_per_s = round(data["eval_count"] / (data["eval_duration"] / 1e9), 1)
        if data.get("prompt_eval_count") and data.get("prompt_eval_duration"):
            self.last_prompt_tokens_per_s = round(
                data["prompt_eval_count"] / (data["prompt_eval_duration"] / 1e9), 1)

    async def chat_json(self, messages: list[dict], schema: dict, max_tokens: int = 420,
                        temperature: float | None = None) -> dict:
        t0 = time.perf_counter()
        r = await self._client.post(f"{self.base_url}/api/chat",
                                    json=self._body(messages, schema, max_tokens, temperature))
        r.raise_for_status()
        data = r.json()
        self._measure(data, t0)
        return parse_json_loose((data.get("message") or {}).get("content", ""))

    async def chat_stream(self, messages: list[dict], schema: dict, max_tokens: int = 420,
                          temperature: float | None = None, model: str | None = None):
        """Like chat_json, but yields the answer's text piece by piece as it is generated."""
        t0 = time.perf_counter()
        body = self._body(messages, schema, max_tokens, temperature, stream=True, model=model)
        async with self._client.stream("POST", f"{self.base_url}/api/chat", json=body) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                piece = (data.get("message") or {}).get("content", "")
                if piece:
                    yield piece
                if data.get("done"):
                    self._measure(data, t0)


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
