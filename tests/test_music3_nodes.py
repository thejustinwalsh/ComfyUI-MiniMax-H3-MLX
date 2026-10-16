"""Tests for the MiniMax Music3 MLX nodes.

Runs on any platform with ``mlx`` (``mlx[cpu]`` on Linux) and ``mlx-audio``
installed, using mlx-audio's randomly initialized tiny Music3 config.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_audio.music.models.minimax_music3")


def _load_module():
    spec = importlib.util.spec_from_file_location("music3_nodes", ROOT / "music3_nodes.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["music3_nodes"] = module
    spec.loader.exec_module(module)
    return module


music3 = _load_module()

CAPTION = "Genre: acoustic pop. BPM: 96. Warm female vocal."
LYRICS = "[verse]\nMorning light\n[chorus]\nSing with me"


@pytest.fixture(scope="module")
def tiny_model():
    from mlx_audio.music.models.minimax_music3.minimax_music3 import Model, ModelConfig

    mx.random.seed(1234)
    model = Model(ModelConfig.tiny())
    mx.eval(model.parameters())
    return model


# (duration, seed, chunks): seed 3 runs long enough on the tiny model to cross
# the 200-frame window, covering latent carry-over and waveform stitching.
@pytest.mark.parametrize(("duration", "seed", "chunks"), [(0.08, 7, 1), (12.0, 3, 2)])
def test_staged_generation_matches_upstream_generate(tiny_model, duration, seed, chunks):
    reference = next(
        tiny_model.generate(text=CAPTION, lyrics=LYRICS, duration=duration, steps=2, seed=seed)
    )
    waveform, sample_rate, stats = music3.generate_audio(
        tiny_model, CAPTION, LYRICS, duration, steps=2, seed=seed
    )

    expected = np.asarray(reference.audio).T  # upstream yields [samples, channels]
    assert sample_rate == reference.sample_rate == 44_100
    assert waveform.shape == expected.shape
    np.testing.assert_allclose(waveform, expected, rtol=1e-5, atol=1e-6)
    assert stats["frames"] == reference.token_count
    assert stats["chunks"] == chunks


def test_progress_is_monotonic_per_stage_and_completes(tiny_model):
    events: list[tuple[str, int, int]] = []
    music3.generate_audio(
        tiny_model,
        CAPTION,
        LYRICS,
        0.4,
        steps=1,
        seed=0,
        on_progress=lambda stage, done, total: events.append((stage, done, total)),
    )
    stages = [stage for stage, *_ in events]
    assert stages[0] == "ar" and stages[-1] == "dit"
    assert stages.index("dit") > max(i for i, s in enumerate(stages) if s == "ar")
    for stage in ("ar", "dit"):
        done = [d for s, d, _ in events if s == stage]
        assert done == sorted(done)
    _, last_done, last_total = events[-1]
    assert last_done == last_total


def test_interrupt_stops_generation(tiny_model):
    class Interrupted(Exception):
        pass

    calls = {"n": 0}

    def check():
        calls["n"] += 1
        if calls["n"] > 3:
            raise Interrupted

    with pytest.raises(Interrupted):
        music3.generate_audio(
            tiny_model, CAPTION, LYRICS, 2.0, steps=1, seed=0, check_interrupt=check
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"caption": " "}, "caption"),
        ({"lyrics": ""}, "instrumental"),
        ({"duration": 0}, "duration"),
        ({"duration": 361}, "duration"),
        ({"steps": 0}, "steps"),
        ({"steps": 31}, "steps"),
    ],
)
def test_input_validation(tiny_model, kwargs, message):
    args = {"caption": CAPTION, "lyrics": LYRICS, "duration": 1.0, "steps": 1, "seed": 0}
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        music3.generate_audio(tiny_model, **args)


def _fake_model_dir(path: Path, tokenizer: bool = True) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"model_type": "minimax_music3"}))
    (path / "model.safetensors").write_bytes(b"")
    if tokenizer:
        (path / "tokenizer").mkdir()
    return path


def test_model_resolution(tmp_path, monkeypatch):
    monkeypatch.setattr(music3, "_music3_dir", lambda: tmp_path)
    local = _fake_model_dir(tmp_path / "MiniMax-Music3-mxfp8")
    nested = _fake_model_dir(tmp_path / "org" / "custom")
    (tmp_path / "not-a-model").mkdir()

    assert music3._local_model_names() == ["MiniMax-Music3-mxfp8", "org/custom"]
    # A downloaded preset resolves to its local folder without hitting the network.
    assert music3._resolve_model("mlx-community/MiniMax-Music3-mxfp8", "", False) == local
    assert music3._resolve_model("org/custom", "", False) == nested
    assert music3._resolve_model("ignored", str(nested), False) == nested
    # Presets already on disk are not offered twice.
    choices = music3._model_choices()
    assert "mlx-community/MiniMax-Music3-mxfp8" not in choices
    assert "mlx-community/MiniMax-Music3-4bit" in choices

    with pytest.raises(FileNotFoundError, match="hf download"):
        music3._resolve_model("mlx-community/MiniMax-Music3-4bit", "", False)
    with pytest.raises(FileNotFoundError, match="custom_path"):
        music3._resolve_model("x", str(tmp_path / "not-a-model"), False)


def test_loader_requires_tokenizer(tmp_path, monkeypatch):
    monkeypatch.setattr(music3, "_music3_dir", lambda: tmp_path)
    monkeypatch.setattr(music3, "_check_runtime", lambda: None)
    _fake_model_dir(tmp_path / "no-tok", tokenizer=False)
    _fake_model_dir(tmp_path / "ok")

    loader = music3.MiniMaxMusic3ModelLoader()
    with pytest.raises(FileNotFoundError, match="tokenizer"):
        loader.load("no-tok", False)
    handle, info = loader.load("ok", False)
    assert handle.model is None  # loading is deferred to the first generation
    assert str(tmp_path / "ok") in info


def test_generate_node_returns_comfy_audio(tiny_model, monkeypatch):
    torch = pytest.importorskip("torch")
    handle = music3.MiniMaxMusic3Handle(model_path=Path("unused"), model=tiny_model)

    node = music3.MiniMaxMusic3Generate()
    audio, info = node.generate(handle, CAPTION, LYRICS, 0.08, 1, 7, unload_after=True)

    assert audio["sample_rate"] == 44_100
    assert isinstance(audio["waveform"], torch.Tensor)
    assert audio["waveform"].ndim == 3 and audio["waveform"].shape[:2] == (1, 2)
    assert "peak MLX memory" in info
    assert handle.model is None  # unload_after released it


def test_node_mappings_are_consistent():
    assert set(music3.NODE_CLASS_MAPPINGS) == set(music3.NODE_DISPLAY_NAME_MAPPINGS)
    for cls in music3.NODE_CLASS_MAPPINGS.values():
        inputs = cls.INPUT_TYPES()
        assert "required" in inputs
        assert hasattr(cls, cls.FUNCTION)
