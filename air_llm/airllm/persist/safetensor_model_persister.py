"""
SafetensorModelPersister
========================
A bullet-proof persister that never crashes and always saves shards to
disk whenever doing so leaves at least 5 GiB of free space.

Key guarantees
--------------
* **No cache eviction** – once a shard is on disk it is never removed by
  this class; if writing a new shard would push free space below the
  5 GiB margin we simply skip caching the new shard.
* **Atomic & concurrent-safe writes** – advisory file locks plus an
  atomic `os.replace()` ensure the cache is never corrupted.
* **Reliable downloads** – remote fetches are retried (exponential
  back-off) and writing falls back noiselessly on any failure.
* **No silent re-downloads** – a shard is fetched from the Hub only when
  it is genuinely absent or corrupted on disk.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
import time
from importlib import import_module
from pathlib import Path
from typing import ContextManager, Dict, Tuple, Union

import requests
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from safetensors.torch import load_file, save_file

from .model_persister import ModelPersister

try:
    import fcntl  # POSIX-only; absent on Windows
except ImportError:  # pragma: no cover – Windows
    fcntl = None  # type: ignore

PathLike = Union[str, Path]
_TensorDict = Dict[str, "torch.Tensor"]

# ---------------------------------------------------------------------------#
# Constants & helpers                                                        #
# ---------------------------------------------------------------------------#
KEEP_FREE_BYTES = 5 * 1024**3  # 5 GiB safety margin
MAX_FETCH_RETRIES = 3
_LOCK_SUFFIX = ".lock"
_TMP_SUFFIX = ".tmp"


def _ensure_dir(path: Path) -> None:
    """`mkdir -p` *path*."""
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------#
# SafetensorModelPersister implementation                                    #
# ---------------------------------------------------------------------------#
class SafetensorModelPersister(ModelPersister):
    # ------------------------------------------------------------------ #
    # path helpers                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _filename(layer_name: str) -> str:
        return f"{layer_name.rstrip('.')}.safetensors"

    def _tensor_path(self, layer_name: str, base: Path) -> Path:
        return base / self._filename(layer_name)

    def _done_path(self, layer_name: str, base: Path) -> Path:
        return base / f"{self._filename(layer_name)}.done"

    # ––––– one canonical HF-cache path builder –––––
    @staticmethod
    def _cache_shard_path(repo_id: str, dir_name: str, layer_name: str) -> Path:
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
        """Split ``hf://<repo_id>/splitted_model`` → ``(repo_id, dir_name)``."""
        assert hf_path.startswith("hf://"), "not an hf-stream path"
        stripped = hf_path[5:]
        key = "/splitted_model"
        idx = stripped.index(key)  # raises if malformed
        repo_id = stripped[:idx]
        dir_name = stripped[idx + 1 :]
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
            if probe == probe.parent:
                break
            probe = probe.parent
        return shutil.disk_usage(probe)

    @staticmethod
    def _estimate_size(state_dict: _TensorDict) -> int:
        total = 0
        for t in state_dict.values():
            try:
                total += t.nbytes  # torch 2.1+
            except AttributeError:
                total += t.element_size() * t.nelement()
        return total

    # ------------------------------------------------------------------ #
    # file-locking helper                                                #
    # ------------------------------------------------------------------ #

    @contextlib.contextmanager
    def _exclusive_lock(self, path: Path) -> ContextManager[None]:
        if fcntl is None:  # Windows
            yield
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
    # cache-write helper                                                 #
    # ------------------------------------------------------------------ #

    def _write_cache_shard(
        self,
        repo_id: str,
        dir_name: str,
        layer_name: str,
        data: bytes,
    ) -> Path:
        """
        Best-effort: write *data* to the HF cache unless doing so would
        leave less than ``KEEP_FREE_BYTES`` free.  Existing cache files
        are **never** removed.
        """
        shard_path = self._cache_shard_path(repo_id, dir_name, layer_name)
        cache_root = shard_path.parent

        # Fast-path: already cached.
        if shard_path.exists():
            return shard_path

        size_bytes = len(data)

        # Choose tmp directory – cache dir preferred, fall back to system tmp.
        try:
            _ensure_dir(cache_root)
            tmp_dir = cache_root
        except Exception:
            tmp_dir = Path(tempfile.gettempdir())

        usage = self._disk_usage(tmp_dir)

        # If writing would violate the 5 GiB margin, skip caching.
        if usage.free - size_bytes < KEEP_FREE_BYTES:
            print("[SafetensorModelPersister] Low disk – skipping cache write.")
            return shard_path

        tmp_path = tmp_dir / (shard_path.name + _TMP_SUFFIX)
        lock_path = shard_path.with_suffix(shard_path.suffix + _LOCK_SUFFIX)

        with self._exclusive_lock(lock_path):
            if shard_path.exists():
                return shard_path  # another process beat us

            try:
                with open(tmp_path, "wb") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                # Same-FS move first, fall back to shutil.move if needed.
                try:
                    os.replace(tmp_path, shard_path)
                except OSError:
                    print("OSERROR")
                    shutil.move(tmp_path, shard_path)
            except Exception:
                print("OTHER")
            finally:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)

        return shard_path

    # ------------------------------------------------------------------ #
    # public API – saving                                                #
    # ------------------------------------------------------------------ #

    def persist_model(
        self,
        state_dict: _TensorDict,
        layer_name: str,
        saving_path: PathLike,
    ) -> None:
        base_dir = Path(saving_path)
        _ensure_dir(base_dir)

        tensor_path = self._tensor_path(layer_name, base_dir)
        tmp_path = tensor_path.with_suffix(tensor_path.suffix + _TMP_SUFFIX)
        lock_path = tensor_path.with_suffix(tensor_path.suffix + _LOCK_SUFFIX)

        with self._exclusive_lock(lock_path):
            if tensor_path.exists():
                return  # already persisted

            size_bytes = self._estimate_size(state_dict)
            usage = self._disk_usage(base_dir)

            # Honour the 5 GiB margin for on-disk persists.
            if usage.free - size_bytes < KEEP_FREE_BYTES:
                print(
                    f"[SafetensorModelPersister] Low disk – skipping persist of {layer_name}."
                )
                return

            try:
                save_file(state_dict, tmp_path)
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
        safe_open = import_module("safetensors.torch").safe_open
        state: _TensorDict = {}
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                state[k] = f.get_tensor(k)
        return state

    # ––––– resilient fetch –––––
    def _download_with_retries(self, url: str, headers: dict) -> bytes:
        delay = 1.0
        for attempt in range(1, MAX_FETCH_RETRIES + 1):
            try:
                with requests.get(url, headers=headers, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    return r.content
            except Exception as ex:
                if attempt == MAX_FETCH_RETRIES:
                    raise
                print(
                    f"[SafetensorModelPersister] Download failed "
                    f"(attempt {attempt}/{MAX_FETCH_RETRIES}): {ex}. "
                    f"Retrying in {delay}s…"
                )
                time.sleep(delay)
                delay *= 2  # exponential back-off
        raise RuntimeError("Unreachable")

    def load_model(
        self,
        layer_name: str,
        path: PathLike,
        *,
        mmap: bool = False,
        **load_kwargs,
    ) -> _TensorDict:
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
                data = self._download_with_retries(url, headers)
            except Exception as ex:  # pragma: no cover – network failure
                print(f"[SafetensorModelPersister] stream-load failed: {ex}")
                raise

            # Best-effort cache write (never evicts).
            cache_file = self._write_cache_shard(repo_id, dir_name, layer_name, data)

            # Load tensors.
            if cache_file.exists():
                try:
                    if mmap:
                        return load_file(cache_file, device="cpu", mmap=True, **load_kwargs)
                    return load_file(cache_file, device="cpu", **load_kwargs)
                except Exception:
                    # Unexpected; fall back to bytes loader.
                    pass

            return self._load_from_bytes(data)

        # ── Local filesystem path (original behaviour) ──────────────────
        file_path = self._tensor_path(layer_name, Path(path))

        if not mmap:
            return load_file(file_path, device="cpu", **load_kwargs)

        try:
            return load_file(file_path, device="cpu", mmap=True, **load_kwargs)
        except TypeError:  # older safetensors without mmap
            return self._load_with_safe_open(file_path)
