"""The local embedding model: all-MiniLM-L6-v2, 384 dimensions, on this machine.

Used in local mode only, for the `memory` table's vectors; once signed in, the
platform embeds with its own model. The Infino client embeds its queries over
the same local table with the same model file through transformers.js, so the
two must agree: they do exactly with the full-precision file (`model.onnx`),
and differ in the third decimal with the 8-bit one, whose integer kernels are
not the same in the two ONNX runtimes. Hence fp32, here and in the client.

The model is downloaded from Hugging Face on first use and cached there. It is
loaded on a background thread (`load_in_background`), never on a turn: until
it is ready, `ready()` gives None, memories are saved without vectors (and
embedded once it is), and recall is keyword-only. A load that failed (offline
on first use) is not tried again for FAILED_RETRY_S.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

MODEL = "Xenova/all-MiniLM-L6-v2"
MODEL_FILE = "onnx/model.onnx"
TOKENIZER_FILE = "tokenizer.json"
DIM = 384
# The model's longest input, in tokens; longer text is cut, as transformers.js does.
MAX_TOKENS = 512
# Texts per model call.
BATCH = 32
# Seconds before a failed load (no network, no cached file) is tried again.
FAILED_RETRY_S = 600.0

Embedder = Callable[[list[str]], list[list[float]]]

_lock = threading.Lock()
_model: Optional[Embedder] = None
_loading = False
_failed_at: Optional[float] = None


def _load() -> Embedder:
    import numpy as np
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(hf_hub_download(MODEL, TOKENIZER_FILE))
    tokenizer.enable_truncation(MAX_TOKENS)
    session = ort.InferenceSession(hf_hub_download(MODEL, MODEL_FILE), providers=["CPUExecutionProvider"])
    inputs = {i.name for i in session.get_inputs()}

    def embed_batch(texts: list[str]) -> list[list[float]]:
        encoded = tokenizer.encode_batch(texts)
        width = max(len(e.ids) for e in encoded)
        ids = np.array([e.ids + [0] * (width - len(e.ids)) for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask + [0] * (width - len(e.ids)) for e in encoded], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in inputs:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = session.run(None, feeds)[0]
        # Mean over the real tokens, then unit length: transformers.js's
        # feature-extraction with {pooling: "mean", normalize: true}.
        weights = mask[..., None].astype(np.float32)
        pooled = (hidden * weights).sum(1) / np.clip(weights.sum(1), 1e-9, None)
        pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
        return pooled.astype(np.float32).tolist()

    def embed(texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), BATCH):
            out += embed_batch([t or " " for t in texts[start:start + BATCH]])
        return out

    return embed


def ready() -> Optional[Embedder]:
    """The model if it is loaded, else None. Never loads, never waits."""
    return _model


def load_in_background(then: Optional[Callable[[], None]] = None) -> None:
    """Load the model on a thread of its own, then call `then` (when it is
    ready). Does nothing while a load runs, or soon after one failed; calls
    `then` at once when the model is loaded already."""
    global _loading
    with _lock:
        if _model is None:
            if _loading or (_failed_at is not None and time.monotonic() - _failed_at < FAILED_RETRY_S):
                return
            _loading = True
    if _model is not None:
        _then(then)
        return

    def run() -> None:
        global _model, _loading, _failed_at
        try:
            model = _load()
        except Exception as e:
            logger.warning("infino: the local model could not load (memories are found by keyword): %s", e)
            with _lock:
                _loading, _failed_at = False, time.monotonic()
            return
        with _lock:
            _model, _loading, _failed_at = model, False, None
        _then(then)

    threading.Thread(target=run, name="infino-model-load", daemon=True).start()


def _then(then: Optional[Callable[[], None]]) -> None:
    if then is None:
        return
    try:
        then()
    except Exception as e:
        logger.warning("infino: after the model loaded: %s", e)


def use(model: Optional[Embedder]) -> None:
    """Replace the model (tests), or None to load the real one again."""
    global _model, _loading, _failed_at
    with _lock:
        _model, _loading, _failed_at = model, False, None
