"""Image generation, importable on the API container and on remote Dask workers alike.

The pipeline is loaded once per process and kept; a Dask worker started from the same
image imports this module, so a submitted task reuses the worker's warm pipeline.
"""
from __future__ import annotations

import io
import os
import threading

_PIPE = None
_LOCK = threading.Lock()

STYLE = {
    "backdrop": "{p}, dreamlike, ethereal, soft focus, drifting mist, painterly, cinematic light, wide view",
    "sprite": "{p}, a single object in the centre, glowing, dreamlike, ethereal, plain black background",
}


def _pipeline():
    global _PIPE
    if _PIPE is None:
        with _LOCK:
            if _PIPE is None:
                import torch
                from diffusers import AutoPipelineForText2Image
                model = os.environ.get("IMAGERY_MODEL", "stabilityai/sd-turbo")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if device == "cuda" else torch.float32
                kwargs = {"torch_dtype": dtype}
                if device == "cuda":
                    kwargs["variant"] = "fp16"
                try:
                    pipe = AutoPipelineForText2Image.from_pretrained(model, **kwargs)
                except (OSError, ValueError):
                    kwargs.pop("variant", None)
                    pipe = AutoPipelineForText2Image.from_pretrained(model, **kwargs)
                pipe = pipe.to(device)
                pipe.set_progress_bar_config(disable=True)
                _PIPE = pipe
    return _PIPE


def warm() -> str:
    pipe = _pipeline()
    return str(pipe.device)


def render_png(prompt: str, kind: str = "backdrop", seed: int = 0, steps: int | None = None) -> bytes:
    import torch
    pipe = _pipeline()
    steps = steps or int(os.environ.get("IMAGERY_STEPS", "2"))
    size = int(os.environ.get("IMAGERY_SIZE", "512"))
    gen = torch.Generator(device=pipe.device).manual_seed(seed or 1234)
    text = STYLE.get(kind, STYLE["backdrop"]).format(p=prompt.strip()[:300])
    with _LOCK:
        image = pipe(prompt=text, num_inference_steps=steps, guidance_scale=0.0,
                     width=size, height=size, generator=gen).images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
