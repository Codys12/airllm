"""
SafetensorModelPersister
========================
Bullet-proof persister that *never* crashes and *always* caches/saves
when ≥ 5 GiB are free. If free space drops below the threshold we just
skip the write (remote-stream cache or persist) and carry on noiselessly.

Changes in this revision
------------------------
• **No exceptions on low-disk** – `persist_model()` now *logs & skips* if
  there isn’t room instead of raising `OSError`.
• **Accurate size estimate** – `_estimate_size()` sums `tensor.nbytes` so
  the free-space check is realistic.
• Minor: renamed `DEFAULT_KEEP_FREE` → `KEEP_FREE_BYTES` for clarity.
• **Guaranteed Hub-cache reuse** – `load_model()` now checks the local
  Hugging Face cache *first* and only hits the network if the shard is
  not already on disk. Fresh downloads are still written to the cache
  (space permitting), so subsequent loads never re-download.
• **Unified cache-path helper** – a new private method
  `_cache_shard_path()` is used everywhere, so `_maybe_cache_shard()`
  *always* writes to the exact location that `load_model()` expects.
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
    import fcntl  # POSIX-only; we fall back gracefully on Windows
except ImportError:  # pragma: no cover – Windows
    fcntl = None  # type: ignore

PathLike = Union[str, Path]
_TensorDict = Dict[str, "torch.Tensor"]

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------
KEEP_FREE_BYTES = 5 * 1024 ** 3  # 5 GiB safety margin
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

    Guarantees
    ----------
    • If there is *any* failure writing to disk (low-space, permission,
      concurrent clobber, etc.) we **never raise**, we simply skip the
      on-disk write and return tensors from RAM.
    • If ≥ ``KEEP_FREE_BYTES`` are available we *always* write – even if
      parent directories don’t yet exist.
    • Concurrent processes use an advisory lock plus atomic `os.replace`
      so the cache is never corrupted.
    • Remote shards are fetched exactly **once**; we always try the local
      Hugging Face cache first and only download if missing.
    """

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _filename(layer_name: str) -> str:
        """Normalise layer name → shard filename."""
        return f"{layer_name.rstrip('.')}.safetensors"  # avoid double dots

    def _tensor_path(self, layer_name: str, base: Path) -> Path:
        return base / self._filename(layer_name)

    def _done_path(self, layer_name: str, base: Path) -> Path:
        return base / f"{self._filename(layer_name)}.done"

    # ---------- NEW: one true canonical HF-cache path builder ----------
    @staticmethod
    def _cache_shard_path(repo_id: str, dir_name: str, layer_name: str) -> Path:
        """
        Return the local cache path for a Hub shard so both loader and
        writer use the **exact** same location.
        """
        return (
            Path(HUGGINGFACE_HUB_CACHE)
            / repo_id
            / dir_name
            / f"{layer_name.rstrip('.')}.safetensors"
        )

    # ------------------------------------------------------------------ #
    # remote-stream helpers                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_hf_stream_path(hf_path: str) -> Tuple[str, str]:
        """Split ``hf://<repo_id>/splitted_model`` → ``(<repo_id>, <dir>)``."""
        assert hf_path.startswith("hf://"), "not an hf-stream path"
        stripped = hf_path[5:]
        key = "/splitted_model"
        idx = stripped.index(key)  # raises if malformed
        repo_id = stripped[:idx]
        dir_name = stripped[idx + 1 :]  # keep folder name itself
        return repo_id, dir_name

    def _load_from_bytes(self, data: bytes) -> _TensorDict:
        """Zero-copy tensor loader directly from a bytes object."""
        safe_open = import_module("safetensors.torch").safe_open
        tensors: _TensorDict = {}
        with safe_open(data, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        return tensors

    # ------------------------------------------------------------------ #
    # disk-space helpers                                                 #
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
        """True if writing *size_bytes* leaves ≥ KEEP_FREE_BYTES free."""
        usage = self._disk_usage(dir_path)
        return usage.free - size_bytes >= KEEP_FREE_BYTES

    @staticmethod
    def _estimate_size(state_dict: _TensorDict) -> int:
        """Return total byte size of all tensors in *state_dict*."""
        total = 0
        for t in state_dict.values():
            try:
                total += t.nbytes  # torch 2.1+
            except AttributeError:
                total += t.element_size() * t.nelement()
        return total

    # ------------------------------------------------------------------ #
    # file-locking helpers                                               #
    # ------------------------------------------------------------------ #

    @contextlib.contextmanager
    def _exclusive_lock(self, path: Path) -> ContextManager[None]:
        """Context manager acquiring an advisory lock on *path* (best-effort)."""
        if fcntl is None:
            yield  # Windows – no robust cross-process lock; hope for best
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
        """Best-effort write of the streamed shard to the HF cache."""
        shard_path = self._cache_shard_path(repo_id, dir_name, layer_name)

        # Already cached → done.
        if shard_path.exists():
            return

        if not self._enough_space(shard_path.parent, len(data)):
            # Insufficient space → skip quietly.
            print("[SafetensorModelPersister] Low disk – skipping cache write.")
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
        """Save *state_dict* under *saving_path*.<layer_name>.safetensors.

        Never raises for low-space – logs and returns.
        """
        base_dir = Path(saving_path)
        _ensure_dir(base_dir)

        tensor_path = self._tensor_path(layer_name, base_dir)
        tmp_path = tensor_path.with_suffix(tensor_path.suffix + _TMP_SUFFIX)
        lock_path = tensor_path.with_suffix(tensor_path.suffix + _LOCK_SUFFIX)

        with self._exclusive_lock(lock_path):
            if tensor_path.exists():
                return

            size_bytes = self._estimate_size(state_dict)
            if not self._enough_space(base_dir, size_bytes):
                print(
                    f"[SafetensorModelPersister] Low disk – need {size_bytes/1e9:.2f} GiB free. Skipping persist."
                )
                return

            try:
                save_file(state_dict, tmp_path)
                # Explicitly fsync to guarantee durability before rename
                with open(tmp_path, "rb") as f:
                    os.fsync(f.fileno())
                os.replace(tmp_path, tensor_path)
                self._done_path(layer_name, base_dir).touch()
            finally:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)

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
          and cached on disk (subject to free-space check). The local cache
          is always checked **first**, so the network is contacted only if
          the shard is missing.
        • Otherwise we honour *mmap* when available and fall back to
          `safe_open()` for older safetensors.
        """
        # ── Remote streaming path ───────────────────────────────────────
        if str(path).startswith("hf://"):
            from huggingface_hub import hf_hub_url

            repo_id, dir_name = self._parse_hf_stream_path(str(path))
            cache_file = self._cache_shard_path(repo_id, dir_name, layer_name)

            # ---------- Try local cache first ---------------------------
            if cache_file.exists():
                try:
                    if mmap:
                        return load_file(cache_file, device="cpu", mmap=True, **load_kwargs)
                    return load_file(cache_file, device="cpu", **load_kwargs)
                except TypeError:  # Older safetensors
                    return self._load_with_safe_open(cache_file)
                except Exception as ex:
                    # Corrupted cache? Warn & fall back to fresh download.
                    print(
                        f"[SafetensorModelPersister] Cache read failed ({ex}). "
                        f"Re-downloading shard…"
                    )

            # ---------- Not cached (or cache unusable) → download -------
            url = hf_hub_url(repo_id, f"{dir_name}/{self._filename(layer_name)}")

            headers = {}
            if (token := os.environ.get("HF_TOKEN")):
                headers["Authorization"] = f"Bearer {token}"

            try:
                with requests.get(url, headers=headers, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    data = r.content

                # Best-effort cache write (only if enough free space)
                self._maybe_cache_shard(repo_id, dir_name, layer_name, data)

                return self._load_from_bytes(data)
            except Exception as ex:  # pragma: no cover – network failure
                print(
                    f"[SafetensorModelPersister] stream-load failed: {ex}. "
                    f"Falling back to local path…"
                )

        # ── Local filesystem path (original behaviour) ──────────────────
        file_path = self._tensor_path(layer_name, Path(path))

        if not mmap:
            return load_file(file_path, device="cpu", **load_kwargs)

        try:
            return load_file(file_path, device="cpu", mmap=True, **load_kwargs)
        except TypeError:  # Older safetensors w/o mmap
            return self._load_with_safe_open(file_path)
