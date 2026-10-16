"""ComfyUI nodes for MiniMax Music 3 inference with MLX on Apple Silicon.

The model implementation comes from ``mlx-audio`` (``mlx_audio.music``). These
nodes drive its generation stages directly instead of calling
``Model.generate`` so ComfyUI can show progress and cancel between
autoregressive frames and between diffusion chunks.

Model lifecycle: the loader node only resolves (and, if asked, downloads) the
weights. The model is materialized in unified memory on the first generation
and stays resident inside the loader's cached output until ``unload_after`` is
set, the loader inputs change, or ComfyUI drops its node cache. ComfyUI's own
model manager does not track MLX memory, so "Unload models" in the UI does not
free it.
"""

from __future__ import annotations

import gc
import os
import platform
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

CATEGORY = "MiniMax Music3 (MLX)"
MODEL_FOLDER = "minimax_music3"

# Native MLX conversions published by mlx-community. Each repo carries
# config.json, the converted safetensors and the official tokenizer.
HF_PRESETS = {
    "mlx-community/MiniMax-Music3-mxfp8": "Recommended: memory/lyric-fidelity balance",
    "mlx-community/MiniMax-Music3-8bit": "Affine 8-bit",
    "mlx-community/MiniMax-Music3-6bit": "Affine 6-bit",
    "mlx-community/MiniMax-Music3-4bit": "Affine 4-bit",
    "mlx-community/MiniMax-Music3-mxfp4": "Experimental; may drop lyrics",
    "mlx-community/MiniMax-Music3-bf16": "Dense reference",
}

DEFAULT_CAPTION = (
    "Warm acoustic pop, 96 BPM, intimate female vocal, fingerpicked guitar, "
    "soft brushed drums, gentle piano in the chorus"
)
DEFAULT_LYRICS = (
    "[verse]\n"
    "Morning light across the room\n"
    "Coffee cooling, quiet tune\n"
    "[chorus]\n"
    "Sing with me, the day is new\n"
    "Every color coming through"
)


def _models_dir() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.models_dir)
    except (ImportError, AttributeError):
        return Path.cwd() / "models"


def _music3_dir() -> Path:
    return _models_dir() / MODEL_FOLDER


def _register_model_folder() -> None:
    try:
        import folder_paths

        folder_paths.add_model_folder_path(MODEL_FOLDER, str(_music3_dir()))
    except (ImportError, AttributeError):
        pass


_register_model_folder()


def _is_model_dir(path: Path) -> bool:
    return (path / "config.json").is_file() and any(path.glob("*.safetensors"))


def _local_model_names() -> list[str]:
    roots = [_music3_dir()]
    try:
        import folder_paths

        roots = [Path(p) for p in folder_paths.get_folder_paths(MODEL_FOLDER)] or roots
    except (ImportError, AttributeError, KeyError):
        pass

    names: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        # Accept <root>/<name> and <root>/<org>/<name> layouts.
        for candidate in [*root.iterdir(), *root.glob("*/*")]:
            if candidate.is_dir() and _is_model_dir(candidate):
                names.add(candidate.relative_to(root).as_posix())
    return sorted(names)


def _model_choices() -> list[str]:
    # Offer a downloaded preset under its repo id, not its directory name, so a
    # workflow saved before the download still validates after it.
    preset_ids = {_preset_dir(name).name: name for name in HF_PRESETS}
    local = [preset_ids.get(name, name) for name in _local_model_names()]
    presets = [name for name in HF_PRESETS if name not in local]
    return [*local, *presets] or list(HF_PRESETS)


def _preset_dir(repo_id: str) -> Path:
    return _music3_dir() / repo_id.split("/", 1)[-1]


def _resolve_model(choice: str, custom_path: str, allow_download: bool) -> Path:
    raw = os.path.expandvars(os.path.expanduser(custom_path.strip()))
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            cwd_candidate = Path.cwd() / path
            path = cwd_candidate if cwd_candidate.exists() else _music3_dir() / path
        path = path.resolve()
        if not _is_model_dir(path):
            raise FileNotFoundError(
                f"custom_path is not an MLX Music3 model directory (needs config.json "
                f"and *.safetensors): {path}"
            )
        return path

    local = _music3_dir() / choice
    if _is_model_dir(local):
        return local.resolve()

    if choice in HF_PRESETS:
        target = _preset_dir(choice)
        if _is_model_dir(target):
            return target.resolve()
        if not allow_download:
            raise FileNotFoundError(
                f"{choice} is not downloaded. Enable allow_download or run:\n"
                f"  hf download {choice} --local-dir {target}"
            )
        from huggingface_hub import snapshot_download

        target.mkdir(parents=True, exist_ok=True)
        print(f"[MiniMax Music3] downloading {choice} -> {target}", flush=True)
        snapshot_download(repo_id=choice, local_dir=str(target))
        if not _is_model_dir(target):
            raise FileNotFoundError(f"Download of {choice} did not produce a model at {target}")
        return target.resolve()

    raise FileNotFoundError(f"Unknown MiniMax Music3 model: {choice}")


def _check_runtime() -> None:
    if os.environ.get("MINIMAX_MUSIC3_ALLOW_ANY_PLATFORM") != "1" and (
        platform.system() != "Darwin"
        or platform.machine().lower() not in {"arm64", "aarch64"}
    ):
        raise RuntimeError(
            "MiniMax Music3 MLX nodes require an Apple Silicon Mac (arm64 macOS); "
            f"detected {platform.system()} {platform.machine()}."
        )
    try:
        import mlx.core  # noqa: F401
        from mlx_audio.music.models.minimax_music3 import Model  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "MiniMax Music3 needs mlx-audio>=0.5.9. Run the ComfyUI Python "
            "environment's `pip install -r custom_nodes/ComfyUI-MiniMax-H3-MLX/requirements.txt`."
        ) from exc


@dataclass
class MiniMaxMusic3Handle:
    """Lazily loaded MLX model shared by every generate node wired to a loader."""

    model_path: Path
    model: Any = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def get(self) -> Any:
        if self.model is None:
            from mlx_audio.music import load_model

            started = time.perf_counter()
            print(f"[MiniMax Music3] loading {self.model_path}", flush=True)
            self.model = load_model(self.model_path)
            print(
                f"[MiniMax Music3] loaded in {time.perf_counter() - started:.1f}s",
                flush=True,
            )
        return self.model

    def unload(self) -> None:
        self.model = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except ImportError:
            pass


# --------------------------------------------------------------------------- #
# Generation, staged so the host can report progress and interrupt.
# --------------------------------------------------------------------------- #


def generate_audio(
    model: Any,
    caption: str,
    lyrics: str,
    duration: float,
    steps: int,
    seed: int,
    on_progress: Callable[[str, int, int], None] | None = None,
    check_interrupt: Callable[[], None] | None = None,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Run MiniMax Music 3 and return ``(waveform[channels, samples], sr, stats)``.

    Mirrors ``mlx_audio.music.models.minimax_music3.Model.generate`` (AR frames,
    then chunked flow-matching DiT + vocoder) with hooks between units of work.
    """
    import mlx.core as mx
    from mlx_audio.music.models.minimax_music3 import ar
    from mlx_audio.music.models.minimax_music3 import minimax_music3 as mm
    from mlx_audio.music.models.minimax_music3.config import (
        CHUNK_FRAMES,
        DIT_CFG_SCALE,
        OVERLAP_LATENT_LENGTH,
    )
    from mlx_audio.music.models.minimax_music3.euler import denoise_chunk

    if not caption.strip():
        raise ValueError("A music description (caption) is required")
    if not lyrics.strip():
        raise ValueError("Lyrics are required; use [instrumental] for instrumental tracks")
    if not 0 < duration <= 360:
        raise ValueError("duration must be between 0 and 360 seconds")
    if not 1 <= steps <= 30:
        raise ValueError("steps must be between 1 and 30")

    tick = on_progress or (lambda *_: None)
    poll = check_interrupt or (lambda: None)
    config = model.config
    started = time.perf_counter()

    # Stage 1: hierarchical autoregressive song writing.
    max_frames = max(1, int(duration * config.frame_rate))
    text_ids = model._text_ids(caption, lyrics)
    mx.random.seed(seed)
    key = mx.random.key(seed)
    embeddings = model.language_model.model.embed_tokens(text_ids)
    hidden, cache = ar.qwen3_hidden(model.language_model, embeddings)
    last_hidden = hidden[:, -1]
    frames = []
    tick("ar", 0, max_frames)
    for frame_index in range(max_frames + 1):
        poll()
        key, subkey = mx.random.split(key)
        result = ar.ar_one_frame(
            model.language_model,
            model.rvq_depth_decoder,
            config,
            last_hidden,
            cache,
            subkey,
            emit_frame=frame_index > 0,
        )
        last_hidden, cache = result.last_hidden, result.cache
        if result.ended:
            break
        if frame_index > 0:
            frames.append(result.frame_hidden)
            tick("ar", len(frames), max_frames)
            if len(frames) >= max_frames:
                break
    if not frames:
        raise ValueError("MiniMax Music 3 generated zero audio frames")
    frame_hiddens = mx.stack(frames, axis=1)
    mx.eval(frame_hiddens)
    del cache, frames
    ar_seconds = time.perf_counter() - started

    # Stage 2: chunked flow-matching DiT and DAC vocoder.
    starts = mm._chunk_starts(frame_hiddens.shape[1])
    waves = []
    previous_latent = None
    previous_condition = None
    mx.random.seed(seed + 7)
    tick("dit", 0, len(starts))
    for chunk_index, start in enumerate(starts):
        poll()
        end = min(start + CHUNK_FRAMES, frame_hiddens.shape[1])
        condition = model.condition_encoder(frame_hiddens[:, start:end])
        noise = mx.random.normal(
            (1, config.dit_in_channels, condition.shape[1])
        ).astype(condition.dtype)
        latents, condition = denoise_chunk(
            model.transformer,
            noise,
            condition,
            num_inference_steps=steps,
            guidance_scale=DIT_CFG_SCALE,
            previous_latent=previous_latent,
            previous_condition=previous_condition,
        )
        carry_start = max(0, latents.shape[-1] - 2 * OVERLAP_LATENT_LENGTH)
        carry_end = max(carry_start, latents.shape[-1] - OVERLAP_LATENT_LENGTH)
        previous_latent = latents[..., carry_start:carry_end]
        previous_condition = condition[:, carry_start:carry_end]
        wave = model.vocoder(latents)
        mx.eval(wave, previous_latent, previous_condition)
        waves.append(wave)
        tick("dit", chunk_index + 1, len(starts))

    audio = mx.concatenate(
        [mm._crop_waveform(wave, i, len(waves)) for i, wave in enumerate(waves)],
        axis=-1,
    )
    # [1, channels, samples] -> [channels, samples]
    waveform = np.asarray(mx.clip(audio[0].astype(mx.float32), -1.0, 1.0))
    if not np.isfinite(waveform).all():
        raise RuntimeError("MiniMax Music 3 produced non-finite audio")

    elapsed = time.perf_counter() - started
    sample_rate = int(model.sample_rate)
    seconds = waveform.shape[-1] / sample_rate
    stats = {
        "frames": int(frame_hiddens.shape[1]),
        "requested_frames": max_frames,
        "chunks": len(starts),
        "audio_seconds": seconds,
        "ar_seconds": ar_seconds,
        "dit_seconds": elapsed - ar_seconds,
        "total_seconds": elapsed,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
    }
    return waveform, sample_rate, stats


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #


class MiniMaxMusic3ModelLoader:
    """Resolve MiniMax Music 3 MLX weights. Loading happens on first generate."""

    @classmethod
    def INPUT_TYPES(cls):
        choices = _model_choices()
        default = (
            "mlx-community/MiniMax-Music3-mxfp8"
            if "mlx-community/MiniMax-Music3-mxfp8" in choices
            else choices[0]
        )
        return {
            "required": {
                "model": (
                    choices,
                    {
                        "default": default,
                        "tooltip": (
                            "Local MLX conversions in models/minimax_music3, or an "
                            "mlx-community repo to download there. mxfp8 is the "
                            "recommended quant when lyric fidelity matters."
                        ),
                    },
                ),
                "allow_download": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Download a missing mlx-community preset into models/minimax_music3.",
                    },
                ),
            },
            "optional": {
                "custom_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Override with any MLX Music3 directory (config.json, *.safetensors, tokenizer/).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MINIMAX_MUSIC3_MODEL", "STRING")
    RETURN_NAMES = ("model", "load_info")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, model: str, allow_download: bool, custom_path: str = ""):
        _check_runtime()
        path = _resolve_model(model, custom_path, bool(allow_download))
        if not (path / "tokenizer").is_dir():
            # Without the official tokenizer mlx-audio silently falls back to its
            # test tokenizer, which produces noise. Fail loudly instead.
            raise FileNotFoundError(
                f"{path} has no tokenizer/ directory. Copy it from "
                "MiniMaxAI/MiniMax-Music3 or use an mlx-community conversion."
            )
        handle = MiniMaxMusic3Handle(model_path=path)
        return (handle, f"model: {path}\nloads on first generation")


class MiniMaxMusic3Generate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MINIMAX_MUSIC3_MODEL",),
                "caption": (
                    "STRING",
                    {
                        "default": DEFAULT_CAPTION,
                        "multiline": True,
                        "tooltip": "Style, mood, tempo, instrumentation, vocal and arrangement.",
                    },
                ),
                "lyrics": (
                    "STRING",
                    {
                        "default": DEFAULT_LYRICS,
                        "multiline": True,
                        "tooltip": (
                            "Section tags ([intro], [verse], [chorus], [bridge], [outro]) "
                            "on their own lines. Use [instrumental] for no vocals."
                        ),
                    },
                ),
                "duration_seconds": (
                    "FLOAT",
                    {
                        "default": 30.0,
                        "min": 1.0,
                        "max": 360.0,
                        "step": 1.0,
                        "tooltip": "Upper bound. The model may end the song earlier.",
                    },
                ),
                "steps": (
                    "INT",
                    {"default": 30, "min": 1, "max": 30, "step": 1, "tooltip": "Flow-matching Euler steps per chunk."},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFF,
                        "tooltip": "Seeds the autoregressive stage, which decides most of the song.",
                    },
                ),
                "unload_after": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Release the model's unified memory after this run; the next run reloads it.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "generation_info")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(
        self,
        model: MiniMaxMusic3Handle,
        caption: str,
        lyrics: str,
        duration_seconds: float,
        steps: int,
        seed: int,
        unload_after: bool,
    ):
        import torch

        progress, check_interrupt = _comfy_hooks(duration_seconds)
        with model.lock:
            try:
                waveform, sample_rate, stats = generate_audio(
                    model.get(),
                    str(caption),
                    str(lyrics),
                    float(duration_seconds),
                    int(steps),
                    int(seed),
                    on_progress=progress,
                    check_interrupt=check_interrupt,
                )
            finally:
                if unload_after:
                    model.unload()
                else:
                    import mlx.core as mx

                    mx.clear_cache()

        audio = {
            "waveform": torch.from_numpy(np.ascontiguousarray(waveform)).unsqueeze(0),
            "sample_rate": sample_rate,
        }
        info = (
            f"audio: {stats['audio_seconds']:.1f}s @ {sample_rate} Hz, {waveform.shape[0]} ch\n"
            f"frames: {stats['frames']}/{stats['requested_frames']}, chunks: {stats['chunks']}\n"
            f"time: AR {stats['ar_seconds']:.0f}s + DiT/vocoder {stats['dit_seconds']:.0f}s "
            f"= {stats['total_seconds'] / 60.0:.2f} min\n"
            f"peak MLX memory: {stats['peak_memory_gb']:.1f} GB"
        )
        print(f"[MiniMax Music3] {info.replace(chr(10), ' | ')}", flush=True)
        return (audio, info)


def _comfy_hooks(duration: float):
    """Progress bar over AR frames (80%) and DiT chunks (20%) plus cancel polling."""
    try:
        import comfy.model_management as mm
        import comfy.utils

        bar = comfy.utils.ProgressBar(1000)
        check = mm.throw_exception_if_processing_interrupted
    except ImportError:
        return None, None

    def progress(stage: str, done: int, total: int) -> None:
        fraction = done / max(total, 1)
        value = 800 * fraction if stage == "ar" else 800 + 200 * fraction
        bar.update_absolute(int(value), 1000)

    return progress, check


NODE_CLASS_MAPPINGS = {
    "MiniMaxMusic3ModelLoader": MiniMaxMusic3ModelLoader,
    "MiniMaxMusic3Generate": MiniMaxMusic3Generate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxMusic3ModelLoader": "Load MiniMax Music3 (MLX)",
    "MiniMaxMusic3Generate": "Generate MiniMax Music3 Song",
}
