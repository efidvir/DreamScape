"""Image generation, importable on the API container and on remote Dask workers alike.

The pipeline is loaded once per process and kept; a Dask worker started from the same
image imports this module, so a submitted task reuses the worker's warm pipeline.

Where it runs: IMAGERY_DEVICE=cuda or cpu, or auto (the graphics card when it has room
to spare, else the processor). On the processor it keeps to IMAGERY_THREADS threads and
decodes with the tiny autoencoder, so a picture takes a few seconds.
"""
from __future__ import annotations

import io
import os
import threading

_PIPE = None
_DEVICE = None
LOCK = threading.RLock()        # one picture at a time in a process

STYLE = {
    "backdrop": "{p}, dreamlike, ethereal, soft focus, drifting mist, painterly, cinematic light, wide view",
    "sprite": "{p}, a single object in the centre, glowing, dreamlike, ethereal, plain black background",
}


def threads() -> int:
    n = int(os.environ.get("IMAGERY_THREADS", "0") or 0)
    return n if n > 0 else max(2, (os.cpu_count() or 4) // 2 - 2)


def device() -> str:
    """Where the models run, decided once per process."""
    global _DEVICE
    if _DEVICE is None:
        import torch
        want = os.environ.get("IMAGERY_DEVICE", "auto").strip().lower()
        if want == "cpu" or not torch.cuda.is_available():
            _DEVICE = "cpu"
        elif want == "cuda":
            _DEVICE = "cuda"
        else:   # auto: only with room to spare beside the storyteller and the voices
            try:
                free, _ = torch.cuda.mem_get_info()
                _DEVICE = "cuda" if free > 3.5e9 else "cpu"
            except Exception:  # noqa: BLE001
                _DEVICE = "cpu"
        if _DEVICE == "cpu":
            torch.set_num_threads(threads())
    return _DEVICE


def pipeline():
    global _PIPE
    if _PIPE is None:
        with LOCK:
            if _PIPE is None:
                import torch
                from diffusers import AutoPipelineForText2Image
                dev = device()
                model = os.environ.get("IMAGERY_MODEL", "stabilityai/sd-turbo")
                dtype = torch.float16 if dev == "cuda" else torch.float32
                try:   # the half-size weights download in half the time; widened on load
                    pipe = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=dtype, variant="fp16")
                except (OSError, ValueError):
                    pipe = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=dtype)
                if dev == "cpu" and os.environ.get("IMAGERY_TINY_VAE", "1") == "1":
                    from diffusers import AutoencoderTiny
                    pipe.vae = AutoencoderTiny.from_pretrained(
                        os.environ.get("IMAGERY_TINY_VAE_MODEL", "madebyollin/taesd"), torch_dtype=dtype)
                pipe = pipe.to(dev)
                pipe.set_progress_bar_config(disable=True)
                _PIPE = pipe
    return _PIPE


def warm() -> str:
    pipeline()
    return device()


def generate(text: str, width: int, height: int, steps: int, seed: int):
    """One picture from the prompt as given (a PIL image)."""
    import torch
    pipe = pipeline()
    gen = torch.Generator(device="cpu").manual_seed(seed or 1234)
    with LOCK:
        return pipe(prompt=text, num_inference_steps=steps, guidance_scale=0.0,
                    width=width, height=height, generator=gen).images[0]


def render_png(prompt: str, kind: str = "backdrop", seed: int = 0, steps: int | None = None) -> bytes:
    steps = steps or int(os.environ.get("IMAGERY_STEPS", "2"))
    size = int(os.environ.get("IMAGERY_SIZE", "512"))
    text = STYLE.get(kind, STYLE["backdrop"]).format(p=prompt.strip()[:300])
    image = generate(text, size, size, steps, seed)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
