"""Shared test fixtures and lightweight stubs for optional runtime deps."""

import sys
import types


if "torch" not in sys.modules:
    torch_module = types.ModuleType("torch")

    def _from_numpy(array):  # pragma: no cover - helper used during import time
        return array

    torch_module.from_numpy = _from_numpy  # type: ignore[attr-defined]
    sys.modules["torch"] = torch_module


if "silero_vad" not in sys.modules:
    silero_module = types.ModuleType("silero_vad")

    class _DummyPrediction:
        def item(self) -> float:
            return 0.0

    class _DummyVadModel:
        def __call__(self, *_args, **_kwargs):
            return _DummyPrediction()

        def reset_states(self) -> None:
            pass

    def _load_silero_vad():
        return _DummyVadModel()

    silero_module.load_silero_vad = _load_silero_vad  # type: ignore[attr-defined]
    sys.modules["silero_vad"] = silero_module
