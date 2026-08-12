"""ComfyUI integration for Argus-AiTeam/minimax-h3-mac.

Heavy MLX and MiniMax imports are intentionally delayed until a node executes. This
lets ComfyUI start and display actionable platform/dependency errors even when the
runtime has not been installed yet.
"""

from __future__ import annotations

import os
import platform
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

CATEGORY = "MiniMax H3 (MLX)"
DENSE_DEQUANT_PROFILES = (
    "off",
    "qkv-only-tiled",
    "ffn-fc2-tiled",
    "qkv-fc2-out-resident",
    "qkv-fc2-out-tiled",
)


def _models_dir() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.models_dir)
    except (ImportError, AttributeError):
        return Path.cwd() / "models"


def _default_checkpoint() -> str:
    return str(_models_dir() / "minimax_h3" / "MiniMax-H3" / "FL2VA")


def _expand_path(value: str, *, required: bool, label: str) -> Path | None:
    raw = os.path.expandvars(os.path.expanduser(str(value).strip()))
    if not raw:
        if required:
            raise ValueError(f"{label} is required")
        return None

    path = Path(raw)
    if not path.is_absolute():
        # Accept both paths relative to ComfyUI and paths relative to
        # ComfyUI/models/minimax_h3.
        cwd_candidate = Path.cwd() / path
        model_candidate = _models_dir() / "minimax_h3" / path
        path = cwd_candidate if cwd_candidate.exists() else model_candidate
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _require_file(root: Path, relative: str, label: str) -> None:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"{label} is incomplete; missing {path}")


def _check_runtime() -> None:
    if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        raise RuntimeError(
            "MiniMax-H3 MLX nodes require an Apple Silicon Mac (arm64 macOS); "
            f"detected {platform.system()} {platform.machine()}."
        )
    if sys.version_info < (3, 11):
        raise RuntimeError(
            f"minimax-h3-mlx requires Python 3.11 or newer; detected {sys.version.split()[0]}."
        )
    try:
        import mlx.core  # noqa: F401
        from .minimax_h3_mlx.pipeline import MiniMaxH3Pipeline  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "MiniMax-H3 MLX dependencies are not installed. Run the ComfyUI Python "
            "environment's `pip install -r custom_nodes/ComfyUI-MiniMax-H3-MLX/requirements.txt`."
        ) from exc


@dataclass
class MiniMaxH3PipelineHandle:
    pipeline: Any
    low_memory: bool
    load_vision: bool
    checkpoint: str
    transformer: str
    text_encoder: str | None
    turbo_lora: str | None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


@dataclass(frozen=True)
class MiniMaxH3Result:
    video: np.ndarray
    audio: np.ndarray
    sample_rate: int
    fps: int
    seconds_per_step: float
    total_seconds: float
    block_cache_stats: dict[str, int | float] | None = None


class MiniMaxH3ModelLoader:
    """Create a reusable pipeline handle.

    In low-memory mode this operation only validates paths and builds a lightweight
    pipeline shell. Components are loaded and released one phase at a time during
    generation by the upstream implementation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint_dir": (
                    "STRING",
                    {
                        "default": _default_checkpoint(),
                        "tooltip": "Official MiniMax-H3 FL2VA directory containing model_index.json, tokenizer and VAEs.",
                    },
                ),
                "transformer_dir": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional BF16/INT8 transformer directory. Empty uses checkpoint_dir/transformer.",
                    },
                ),
                "text_encoder_dir": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional quantized text encoder. Empty uses checkpoint_dir/text_encoder.",
                    },
                ),
                "turbo_lora_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional Turbo safetensors file or native MLX Turbo directory.",
                    },
                ),
                "low_memory": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Stage the text encoder, DiT and VAEs. Required for typical 24GB Macs; text-to-video only upstream.",
                    },
                ),
                "stream_blocks": ("BOOLEAN", {"default": True}),
                "stream_block_group_size": (
                    "INT",
                    {"default": 1, "min": 1, "max": 8, "step": 1},
                ),
                "memory_limit_gb": (
                    "FLOAT",
                    {"default": 16.0, "min": 4.0, "max": 192.0, "step": 0.5},
                ),
                "memory_pressure_guard": ("BOOLEAN", {"default": False}),
                "load_vision": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Load image-conditioning modules. Upstream low-memory mode currently cannot use keyframes.",
                    },
                ),
                "turbo_lora_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05},
                ),
                "turbo_lora_alpha": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 4096.0,
                        "step": 1.0,
                        "tooltip": "0 reads metadata/uses native MLX alpha. Set only for legacy PEFT adapters.",
                    },
                ),
                "sigma_shift_video": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": "0 uses model default; LightX2V 768p Turbo recommends 6.",
                    },
                ),
                "sigma_shift_audio": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": "0 uses model default; LightX2V Turbo recommends 3.",
                    },
                ),
                "dense_dequant_profile": (DENSE_DEQUANT_PROFILES, {"default": "off"}),
            }
        }

    RETURN_TYPES = ("MINIMAX_H3_PIPELINE", "STRING")
    RETURN_NAMES = ("pipeline", "load_info")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(
        self,
        checkpoint_dir: str,
        transformer_dir: str,
        text_encoder_dir: str,
        turbo_lora_path: str,
        low_memory: bool,
        stream_blocks: bool,
        stream_block_group_size: int,
        memory_limit_gb: float,
        memory_pressure_guard: bool,
        load_vision: bool,
        turbo_lora_scale: float,
        turbo_lora_alpha: float,
        sigma_shift_video: float,
        sigma_shift_audio: float,
        dense_dequant_profile: str,
    ):
        _check_runtime()
        if low_memory and load_vision:
            raise ValueError(
                "The upstream pipeline does not support keyframes in low-memory mode. "
                "Disable load_vision or disable low_memory."
            )

        checkpoint = _expand_path(checkpoint_dir, required=True, label="checkpoint_dir")
        assert checkpoint is not None
        _require_file(checkpoint, "model_index.json", "checkpoint_dir")
        for component in ("video_vae", "audio_vae", "tokenizer", "processor"):
            if not (checkpoint / component).is_dir():
                raise FileNotFoundError(
                    f"checkpoint_dir is incomplete; missing directory {checkpoint / component}"
                )

        transformer = _expand_path(transformer_dir, required=False, label="transformer_dir")
        if transformer is None:
            transformer = (checkpoint / "transformer").resolve()
        _require_file(transformer, "config.json", "transformer_dir")
        _require_file(transformer, "model.safetensors.index.json", "transformer_dir")

        text_encoder = _expand_path(text_encoder_dir, required=False, label="text_encoder_dir")
        if text_encoder is not None:
            _require_file(text_encoder, "config.json", "text_encoder_dir")
            _require_file(text_encoder, "model.safetensors.index.json", "text_encoder_dir")
        else:
            _require_file(checkpoint, "text_encoder/config.json", "checkpoint_dir")
            _require_file(
                checkpoint,
                "text_encoder/model.safetensors.index.json",
                "checkpoint_dir",
            )

        turbo_lora = _expand_path(turbo_lora_path, required=False, label="turbo_lora_path")

        from .minimax_h3_mlx.pipeline import MiniMaxH3Pipeline

        pipe = MiniMaxH3Pipeline.from_pretrained(
            checkpoint,
            transformer_dir=transformer,
            load_vision=bool(load_vision),
            stream_blocks=bool(stream_blocks or low_memory),
            low_memory=bool(low_memory),
            text_encoder_dir=text_encoder,
            turbo_lora_path=turbo_lora,
            turbo_lora_alpha=float(turbo_lora_alpha) if turbo_lora_alpha > 0 else None,
            turbo_lora_scale=float(turbo_lora_scale),
            sigma_shift_video=float(sigma_shift_video) if sigma_shift_video > 0 else None,
            sigma_shift_audio=float(sigma_shift_audio) if sigma_shift_audio > 0 else None,
            memory_limit_gb=float(memory_limit_gb),
            stream_block_group_size=int(stream_block_group_size),
            dense_dequant_profile=dense_dequant_profile,
            memory_pressure_guard=bool(memory_pressure_guard),
        )
        handle = MiniMaxH3PipelineHandle(
            pipeline=pipe,
            low_memory=bool(low_memory),
            load_vision=bool(load_vision),
            checkpoint=str(checkpoint),
            transformer=str(transformer),
            text_encoder=str(text_encoder) if text_encoder else None,
            turbo_lora=str(turbo_lora) if turbo_lora else None,
        )
        info = (
            f"checkpoint: {checkpoint}\n"
            f"transformer: {transformer}\n"
            f"text encoder: {text_encoder or checkpoint / 'text_encoder'}\n"
            f"turbo: {turbo_lora or 'disabled'}\n"
            f"low memory: {bool(low_memory)}, streamed blocks: {bool(stream_blocks or low_memory)}"
        )
        return (handle, info)


class MiniMaxH3Generate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("MINIMAX_H3_PIPELINE",),
                "prompt": (
                    "STRING",
                    {"default": "A cinematic red panda running through a misty bamboo forest", "multiline": True},
                ),
                "width": ("INT", {"default": 768, "min": 32, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 448, "min": 32, "max": 2048, "step": 32}),
                "duration_seconds": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.1, "max": 15.0, "step": 0.1},
                ),
                "steps": (
                    "INT",
                    {
                        "default": 5,
                        "min": 2,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Sigma grid points. Denoiser forwards (NFE) = steps - 1.",
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "drop_adaln": ("BOOLEAN", {"default": True}),
                "block_cache": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Approximate residual cache; disable for full computation."},
                ),
                "block_cache_threshold": (
                    "FLOAT",
                    {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "block_cache_depth": (
                    "FLOAT",
                    {"default": 0.75, "min": 0.0, "max": 0.99, "step": 0.01},
                ),
                "block_cache_max_consecutive": (
                    "INT",
                    {"default": 2, "min": 0, "max": 100, "step": 1},
                ),
                "cache_text_conditioning": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "start_image": ("IMAGE",),
                "end_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("MINIMAX_H3_RESULT", "STRING")
    RETURN_NAMES = ("result", "generation_info")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    @staticmethod
    def _tensor_to_pil(image):
        from PIL import Image

        if image.ndim == 4:
            image = image[0]
        array = image.detach().cpu().float().clamp(0, 1).numpy()
        return Image.fromarray((array * 255.0 + 0.5).astype(np.uint8), mode="RGB")

    def generate(
        self,
        pipeline: MiniMaxH3PipelineHandle,
        prompt: str,
        width: int,
        height: int,
        duration_seconds: float,
        steps: int,
        seed: int,
        drop_adaln: bool,
        block_cache: bool,
        block_cache_threshold: float,
        block_cache_depth: float,
        block_cache_max_consecutive: int,
        cache_text_conditioning: bool,
        start_image=None,
        end_image=None,
    ):
        if width % 32 or height % 32:
            raise ValueError(f"width and height must be multiples of 32; got {width}x{height}")
        if not str(prompt).strip():
            raise ValueError("prompt cannot be empty")

        images = []
        anchors = []
        if start_image is not None:
            images.append(self._tensor_to_pil(start_image))
            anchors.append("first")
        if end_image is not None:
            images.append(self._tensor_to_pil(end_image))
            anchors.append("last")
        if images and pipeline.low_memory:
            raise ValueError(
                "Image conditioning is not supported by the upstream low-memory path. "
                "Remove the image inputs or reload with low_memory=false and load_vision=true."
            )
        if images and not pipeline.load_vision:
            raise ValueError("Reload the model with load_vision=true before connecting keyframe images.")

        cache_config = None
        if block_cache:
            from .minimax_h3_mlx.block_cache import BlockCacheConfig

            cache_config = BlockCacheConfig(
                sigma_threshold=float(block_cache_threshold),
                max_consecutive=int(block_cache_max_consecutive),
                cache_depth=float(block_cache_depth),
            )

        # A pipeline owns mutable phase caches and is not thread-safe. ComfyUI normally
        # executes one queue item at a time, but the lock also protects API callers.
        with pipeline.lock:
            output = pipeline.pipeline(
                str(prompt),
                duration_seconds=float(duration_seconds),
                num_inference_steps=int(steps),
                seed=int(seed),
                images=images or None,
                keyframe_anchors=tuple(anchors),
                height=int(height),
                width=int(width),
                drop_adaln=bool(drop_adaln),
                block_cache_config=cache_config,
                cache_text_conditioning=bool(cache_text_conditioning),
                verbose=True,
            )

        result = MiniMaxH3Result(
            video=output.video,
            audio=output.audio,
            sample_rate=int(output.sample_rate),
            fps=int(output.fps),
            seconds_per_step=float(output.seconds_per_step),
            total_seconds=float(output.total_seconds),
            block_cache_stats=output.block_cache_stats,
        )
        frame_count = int(result.video.shape[0])
        audio_seconds = result.audio.shape[-1] / result.sample_rate
        info = (
            f"{result.video.shape[2]}x{result.video.shape[1]}, {frame_count} frames @ {result.fps} FPS\n"
            f"audio: {audio_seconds:.2f}s @ {result.sample_rate} Hz stereo\n"
            f"time: {result.total_seconds / 60.0:.2f} min total, "
            f"{result.seconds_per_step:.1f}s per denoising step"
        )
        if result.block_cache_stats is not None:
            stats = result.block_cache_stats
            info += (
                f"\nblock cache: full={stats['full_steps']}, cached={stats['cache_steps']}, "
                f"saved={100.0 * float(stats['saved_fraction']):.1f}%"
            )
        return (result, info)


class MiniMaxH3ResultToComfyUI:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "result": ("MINIMAX_H3_RESULT",),
                "frames": (("all", "first", "last"), {"default": "all"}),
                "frame_stride": ("INT", {"default": 1, "min": 1, "max": 120, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT")
    RETURN_NAMES = ("images", "audio", "fps")
    FUNCTION = "convert"
    CATEGORY = CATEGORY

    def convert(self, result: MiniMaxH3Result, frames: str, frame_stride: int):
        import torch

        if frames == "first":
            selected = result.video[:1]
        elif frames == "last":
            selected = result.video[-1:]
        else:
            selected = result.video[:: int(frame_stride)]
        selected = np.ascontiguousarray(selected)
        images = torch.from_numpy(selected).to(dtype=torch.float32).div_(255.0)

        waveform = np.ascontiguousarray(result.audio, dtype=np.float32)
        if waveform.ndim == 1:
            waveform = waveform[None, :]
        audio = {
            "waveform": torch.from_numpy(waveform).unsqueeze(0),
            "sample_rate": int(result.sample_rate),
        }
        return (images, audio, int(result.fps))


class MiniMaxH3SaveVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "result": ("MINIMAX_H3_RESULT",),
                "filename_prefix": ("STRING", {"default": "minimax_h3/ComfyUI"}),
                "crf": (
                    "INT",
                    {"default": 18, "min": 0, "max": 51, "step": 1, "tooltip": "Lower is higher H.264 quality."},
                ),
                "ffmpeg_path": (
                    "STRING",
                    {"default": "", "tooltip": "Empty searches PATH (brew install ffmpeg)."},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def save(
        self,
        result: MiniMaxH3Result,
        filename_prefix: str,
        crf: int,
        ffmpeg_path: str,
    ):
        import folder_paths
        from .minimax_h3_mlx.media import save_mp4

        output_dir = folder_paths.get_output_directory()
        height, width = result.video.shape[1:3]
        full_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix,
            output_dir,
            width,
            height,
        )
        full_folder = Path(full_folder)
        full_folder.mkdir(parents=True, exist_ok=True)
        filename = filename.replace("%batch_num%", "0")
        path = full_folder / f"{filename}_{counter:05}_.mp4"
        wav_sidecar = path.with_suffix(".wav")
        while path.exists() or wav_sidecar.exists():
            counter += 1
            path = full_folder / f"{filename}_{counter:05}_.mp4"
            wav_sidecar = path.with_suffix(".wav")

        try:
            save_mp4(
                path,
                result.video,
                result.fps,
                result.audio,
                result.sample_rate,
                crf=int(crf),
                ffmpeg=ffmpeg_path.strip() or None,
            )
        finally:
            # The upstream muxer uses a temporary WAV beside the MP4.
            wav_sidecar.unlink(missing_ok=True)

        descriptor = {
            "filename": path.name,
            "subfolder": subfolder,
            "type": "output",
        }
        return {
            "ui": {"minimax_h3_video": [descriptor], "text": [str(path)]},
            "result": (str(path),),
        }


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3ModelLoader": MiniMaxH3ModelLoader,
    "MiniMaxH3Generate": MiniMaxH3Generate,
    "MiniMaxH3ResultToComfyUI": MiniMaxH3ResultToComfyUI,
    "MiniMaxH3SaveVideo": MiniMaxH3SaveVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3ModelLoader": "Load MiniMax H3 (MLX)",
    "MiniMaxH3Generate": "Generate MiniMax H3 Video + Audio",
    "MiniMaxH3ResultToComfyUI": "MiniMax H3 Result to Images + Audio",
    "MiniMaxH3SaveVideo": "Save MiniMax H3 MP4",
}
