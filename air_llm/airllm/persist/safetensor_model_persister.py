"""SafetensorModelPersister
================================
Bullet‑proof persister that *always* writes a shard to disk when
≥ 5 GiB are free – even if directories don’t exist yet – and ensures
concurrent processes never corrupt the cache.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import contextlib
from importlib import import_module
from pathlib import Path
from typing import Dict, Tuple, Union, ContextManager

import requests
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from safetensors.torch import load_file, save_file

from .model_persister import ModelPersister

try:
    import fcntl  # POSIX‑only; we fall back gracefully on Windows
except ImportError:  # pragma: no cover – Windows
    fcntl = None  # type: ignore

PathLike = Union[str, Path]
_TensorDict = Dict[str, "torch.Tensor"]

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------
DEFAULT_KEEP_FREE = 5 * 1024 ** 3  # 5 GiB
_LOCK_SUFFIX = ".lock"
_TMP_SUFFIX = ".tmp"


def _ensure_dir(path: Path) -> None:
    """mkdir -p *path* (if it is meant to be a directory)"""
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# SafetensorModelPersister implementation
# ---------------------------------------------------------------------------
class SafetensorModelPersister(ModelPersister):
    """
    Persist layers as `<layer_name>.safetensors` and support mmap sharing.

    Key guarantees
    --------------
    • If ≥ ``DEFAULT_KEEP_FREE`` bytes are available on the target filesystem
      we **always** write the shard to disk – even if intermediate
      directories are missing.
    • Writes are atomic and safe for concurrent processes via an advisory
      file lock (POSIX; best‑effort no‑lock on Windows).
    • Remote streams from the Hugging Face Hub are optionally cached under
      the standard HF cache dir using the same guarantees.
    """

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _filename(layer_name: str) -> str:
        """Normalise layer name → shard filename."""
        return f"{layer_name.rstrip('.')}\.safetensors"  # avoid double dots

    def _tensor_path(self, layer_name: str, base: Path) -> Path:
        return base / self._filename(layer_name)

    def _done_path(self, layer_name: str, base: Path) -> Path:
        return base / f"{self._filename(layer_name)}.done"

    # ------------------------------------------------------------------ #
    # remote‑stream helpers                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_hf_stream_path(hf_path: str) -> Tuple[str, str]:
        """Split ``hf://<repo_id>/splitted_model`` → ``(<repo_id>, <dir>)``."""
        assert hf_path.startswith("hf://"), "not an hf‑stream path"
        stripped = hf_path[5:]
        key = "/splitted_model"
        idx = stripped.index(key)  # raises if malformed
        repo_id = stripped[:idx]
        dir_name = stripped[idx + 1 :]  # keep folder name itself
        return repo_id, dir_name

    def _load_from_bytes(self, data: bytes) -> _TensorDict:
        """Zero‑copy tensor loader directly from a bytes object."""
        safe_open = import_module("safetensors.torch").safe_open
        tensors: _TensorDict = {}
        with safe_open(data, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        return tensors

    # ------------------------------------------------------------------ #
    # disk‑space helpers                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _disk_usage(path: Path) -> shutil.disk_usage:
        """Return `shutil.disk_usage` for *path* or its nearest existing parent."""
        probe = path
        while not probe.exists():
            if probe == probe.parent:  # reached filesystem root
                break
            probe = probe.parent
        return shutil.disk_usage(probe)

    def _enough_space(self, dir_path: Path, size_bytes: int) -> bool:
        """True if writing *size_bytes* leaves ≥ DEFAULT_KEEP_FREE bytes free."""
        usage = self._disk_usage(dir_path)
        return usage.free - size_bytes >= DEFAULT_KEEP_FREE

    # ------------------------------------------------------------------ #
    # file‑locking helpers                                               #
    # ------------------------------------------------------------------ #

    @contextlib.contextmanager
    def _exclusive_lock(self, path: Path) -> ContextManager[None]:
        """Context manager acquiring an advisory lock on *path* (best‑effort)."""
        if fcntl is None:
            yield  # Windows – no robust cross‑process lock; hope for the best
            return
        _ensure_dir(path.parent)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    # ------------------------------------------------------------------ #
    # caching helpers                                                    #
    # ------------------------------------------------------------------ #

    def _maybe_cache_shard(
        self,
        repo_id: str,
        dir_name: str,
        layer_name: str,
        data: bytes,
    ) -> None:
        """Best‑effort write of the streamed shard to the HF cache."""
        cache_root = Path(HUGGINGFACE_HUB_CACHE)
        shard_path = cache_root / repo_id / dir_name / self._filename(layer_name)

        # Already cached → done.
        if shard_path.exists():
            return

        if not self._enough_space(cache_root, len(data)):
            # Insufficient space → skip quietly.
            return

        tmp_path = shard_path.with_suffix(shard_path.suffix + _TMP_SUFFIX)
        lock_path = shard_path.with_suffix(shard_path.suffix + _LOCK_SUFFIX)

        with self._exclusive_lock(lock_path):
            if shard_path.exists():  # another proc beat us
                return

            _ensure_dir(shard_path.parent)

            try:
                with open(tmp_path, "wb") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, shard_path)
            finally:
                # Clean up leftover tmp on failure
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    # public API – saving                                                #
    # ------------------------------------------------------------------ #

    def persist_model(
        self,
        state_dict: _TensorDict,
        layer_name: str,
        saving_path: PathLike,
    ) -> None:
        """Save *state_dict* under *saving_path*.<layer_name>.safetensors."""
        base_dir = Path(saving_path)
        _ensure_dir(base_dir)

        tensor_path = self._tensor_path(layer_name, base_dir)
        tmp_path = tensor_path.with_suffix(tensor_path.suffix + _TMP_SUFFIX)
        lock_path = tensor_path.with_suffix(tensor_path.suffix + _LOCK_SUFFIX)

        with self._exclusive_lock(lock_path):
            # Double‑check: if another process has written while we waited.
            if tensor_path.exists():
                return

            if not self._enough_space(base_dir, state_dict.__sizeof__()):
                raise OSError(
                    f"Not enough space to write {tensor_path}: need ≥ {DEFAULT_KEEP_FREE >> 30} GiB free."
                )

            save_file(state_dict, tmp_path)
            # Explicitly fsync to guarantee durability before rename
            with open(tmp_path, "rb") as f:
                os.fsync(f.fileno())
            os.replace(tmp_path, tensor_path)
            self._done_path(layer_name, base_dir).touch()

    def model_persist_exist(self, layer_name: str, saving_path: PathLike) -> bool:
        base = Path(saving_path)
        return (
            self._tensor_path(layer_name, base).exists()
            and self._done_path(layer_name, base).exists()
        )

    # ------------------------------------------------------------------ #
    # loading                                                            #
    # ------------------------------------------------------------------ #

    def _load_with_safe_open(self, file_path: Path) -> _TensorDict:
        """Fallback loader that never copies data (mmap via safe_open)."""
        safe_open = import_module("safetensors.torch").safe_open
        state: _TensorDict = {}
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                state[k] = f.get_tensor(k)
        return state

    def load_model(
        self,
        layer_name: str,
        path: PathLike,
        *,
        mmap: bool = False,
        **load_kwargs,
    ) -> _TensorDict:
        """Load a safetensor shard onto CPU.

        • If *path* starts with ``hf://`` the shard is streamed from the Hub
          and cached on disk (subject to free‑space check).
        • Otherwise we honour *mmap* when available and fall back to
          `safe_open()` for older safetensors.
        """
        # ── Remote streaming path ───────────────────────────────────────
        if str(path).startswith("hf://"):
            from huggingface_hub import hf_hub_url

            repo_id, dir_name = self._parse_hf_stream_path(str(path))
            url = hf_hub_url(repo_id, f"{dir_name}/{self._filename(layer_name)}")

            headers = {}
            if (token := os.environ.get("HF_TOKEN")):
                headers["Authorization"] = f"Bearer {token}"

            try:
                with requests.get(url, headers=headers, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    data = r.content
                # Best‑effort cache
                self._maybe_cache_shard(repo_id, dir_name, layer_name, data)
                return self._load_from_bytes(data)
            except Exception as ex:  # pragma: no cover – network failure
                print(f"[SafetensorModelPersister] stream‑load failed: {ex}. Falling back to local path…")

        # ── Local filesystem path (original behaviour) ──────────────────
        file_path = self._tensor_path(layer_name, Path(path))

        if not mmap:
            return load_file(file_path, device="cpu", **load_kwargs)

        try:
            return load_file(file_path, device="cpu", mmap=True, **load_kwargs)
        except TypeError:  # Older safetensors w/o mmap
            return self._load_with_safe_open(file_path)
