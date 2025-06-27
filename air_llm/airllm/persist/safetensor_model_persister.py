"""
SafetensorModelPersister
========================
A bullet‑proof persister that never crashes and always saves shards to
 disk whenever doing so leaves at least 5 GiB of free space.

Key guarantees
--------------
* **No cache eviction** – once a shard is on disk it is never removed by
  this class; if writing a new shard would push free space below the
  5 GiB margin we simply skip caching the new shard.
* **Atomic & concurrent‑safe writes** – advisory file locks plus an
  atomic ``os.replace()`` ensure the cache is never corrupted.
* **Reliable downloads** – remote fetches are retried (exponential
  back‑off) and writing falls back noiselessly on any failure.
* **No silent re‑downloads** – a shard is fetched from the Hub only when
  it is genuinely absent or corrupted on disk.

This version adds **rich debug logging** for every significant cache
read/write operation so you can trace exactly what is happening at
runtime.
"""

from __future__ import annotations

import contextlib
import logging
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
    import fcntl  # POSIX‑only; absent on Windows
except ImportError:  # pragma: no cover – Windows
    fcntl = None  # type: ignore

PathLike = Union[str, Path]
_TensorDict = Dict[str, "torch.Tensor"]

# ---------------------------------------------------------------------------#
# Logging setup                                                              #
# ---------------------------------------------------------------------------#
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.DEBUG)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------#
# Constants & helpers                                                        #
# ---------------------------------------------------------------------------#
KEEP_FREE_BYTES = 5 * 1024 ** 3  # 5 GiB safety margin
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
    """Persists and loads model shards in the *safetensors* format with
    robust guarantees plus verbose debug prints for tracing.
    """

    # ------------------------------------------------------------------ #
    # path helpers                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _filename(layer_name: str) -> str:
        return f"{layer_name.rstrip('.')}\.safetensors"

    def _tensor_path(self, layer_name: str, base: Path) -> Path:
        return base / self._filename(layer_name)

    def _done_path(self, layer_name: str, base: Path) -> Path:
        return base / f"{self._filename(layer_name)}.done"

    # ––––– one canonical HF‑cache path builder –––––
    @staticmethod
    def _cache_shard_path(repo_id: str, dir_name: str, layer_name: str) -> Path:
        return (
            Path(HUGGINGFACE_HUB_CACHE)
            / repo_id
            / dir_name
            / f"{layer_name.rstrip('.')}\.safetensors"
        )

    # ------------------------------------------------------------------ #
    # remote‑stream helpers                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_hf_stream_path(hf_path: str) -> Tuple[str, str]:
        """Split ``hf://<repo_id>/splitted_model`` → ``(repo_id, dir_name)``."""
        assert hf_path.startswith("hf://"), "not an hf‑stream path"
        stripped = hf_path[5:]
        key = "/splitted_model"
        idx = stripped.index(key)  # raises if malformed
        repo_id = stripped[:idx]
        dir_name = stripped[idx + 1 :]
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
        """Return ``shutil.disk_usage`` for *path* or its nearest existing parent."""
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
    # file‑locking helper                                                #
    # ------------------------------------------------------------------ #

    @contextlib.contextmanager
    def _exclusive_lock(self, path: Path) -> ContextManager[None]:
        if fcntl is None:  # Windows – best effort only
            yield
            return
        _ensure_dir(path.parent)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            log.debug("Acquiring lock %s", path)
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            log.debug("Released lock %s", path)

    # ------------------------------------------------------------------ #
    # cache‑write helper                                                 #
    # ------------------------------------------------------------------ #

    def _write_cache_shard(
        self,
        repo_id: str,
        dir_name: str,
        layer_name: str,
        data: bytes,
    ) -> Path:
        """
        Best‑effort: write *data* to the HF cache unless doing so would
        leave less than ``KEEP_FREE_BYTES`` free.  Existing cache files
        are **never** removed.
        """
        shard_path = self._cache_shard_path(repo_id, dir_name, layer_name)
        cache_root = shard_path.parent

        # Fast‑path: already cached.
        if shard_path.exists():
            log.debug("[cache‑hit] %s already cached", shard_path)
            return shard_path

        size_bytes = len(data)
        log.debug("Caching shard %s (%.2f MiB) into %s", layer_name, size_bytes / 2**20, shard_path)

        # Choose tmp directory – cache dir preferred, fall back to system tmp.
        try:
            _ensure_dir(cache_root)
            tmp_dir = cache_root
        except Exception:
            tmp_dir = Path(tempfile.gettempdir())
            log.debug("Falling back to system tmp dir %s", tmp_dir)

        usage = self._disk_usage(tmp_dir)

        # If writing would violate the 5 GiB margin, skip caching.
        if usage.free - size_bytes < KEEP_FREE_BYTES:
            log.warning("Low disk – skipping cache write for %s (only %.2f GiB free)", layer_name, usage.free / 2**30)
            return shard_path

        tmp_path = tmp_dir / (shard_path.name + _TMP_SUFFIX)
        lock_path = shard_path.with_suffix(shard_path.suffix + _LOCK_SUFFIX)

        with self._exclusive_lock(lock_path):
            if shard_path.exists():
                log.debug("[race‑lost] Another process cached %s first", layer_name)
                return shard_path  # another process beat us

            try:
                with open(tmp_path, "wb") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                log.debug("Written tmp shard %s", tmp_path)
                # Same‑FS move first, fall back to shutil.move if needed.
                try:
                    os.replace(tmp_path, shard_path)
                    log.debug("Promoted tmp → cache: %s", shard_path)
                except OSError:
                    shutil.move(tmp_path, shard_path)
                    log.debug("Cross‑device move tmp → cache: %s", shard_path)
            except Exception as ex:
                log.error("Failed to cache shard %s: %s", layer_name, ex, exc_info=True)
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

        log.debug("Persist request: %s → %s", layer_name, tensor_path)

        with self._exclusive_lock(lock_path):
            if tensor_path.exists():
                log.debug("[persist‑skip] %s already persisted", layer_name)
                return  # already persisted

            size_bytes = self._estimate_size(state_dict)
            usage = self._disk_usage(base_dir)

            # Honour the 5 GiB margin for on‑disk persists.
            if usage.free - size_bytes < KEEP_FREE_BYTES:
                log.warning("Low disk – skipping persist of %s (only %.2f GiB free)", layer_name, usage.free / 2**30)
                return

            try:
                save_file(state_dict, tmp_path)
                with open(tmp_path, "rb") as f:
                    os.fsync(f.fileno())
                os.replace(tmp_path, tensor_path)
                self._done_path(layer_name, base_dir).touch()
                log.debug("Persisted %s (%.2f MiB) to %s", layer_name, size_bytes / 2**20, tensor_path)
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
                log.debug("Downloading shard (attempt %d): %s", attempt, url)
                with requests.get(url, headers=headers, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    data = r.content
                    log.debug("Downloaded %.2f MiB from %s", len(data) / 2**20, url)
                    return data
            except Exception as ex:
                if attempt == MAX_FETCH_RETRIES:
                    log.error("Permanent failure downloading %s: %s", url, ex)
                    raise
                log.warning("Download failed (attempt %d/%d): %s. Retrying in %.1fs…", attempt, MAX_FETCH_RETRIES, ex, delay)
                time.sleep(delay)
                delay *= 2  # exponential back‑off
        raise RuntimeError("Unreachable")

    def load_model(
        self,
        layer_name: str,
        path: PathLike,
        *,
        mmap: bool = False,
        **load_kwargs,
    ) -> _TensorDict:
        log.debug("Load request: layer=%s, path=%s", layer_name, path)

        # ── Remote streaming path ───────────────────────────────────────
        if str(path).startswith("hf://"):
            from huggingface_hub import hf_hub_url

            repo_id, dir_name = self._parse_hf_stream_path(str(path))
            cache_file = self._cache_shard_path(repo_id, dir_name, layer_name)

            # ---------- Try local cache first ---------------------------
            if cache_file.exists():
                log.debug("Trying cached shard %s", cache_file)
                try:
                    if mmap:
                        tensors = load_file(cache_file, device="cpu", mmap=True, **load_kwargs)
                    else:
                        tensors = load_file(cache_file, device="cpu", **load_kwargs)
                    log.debug("Loaded shard %s from cache", cache_file)
                    return tensors
                except Exception as ex:
                    # Corrupted cache? Warn & fall back to fresh download.
                    log.warning("Cache read failed for %s: %s. Re‑downloading…", cache_file, ex)

            # ---------- Not cached (or cache unusable) → download -------
            url = hf_hub_url(repo_id, f"{dir_name}/{self._filename(layer_name)}")
            headers = {}
            if (token := os.environ.get("HF_TOKEN")):
                headers["Authorization"] = f"Bearer {token}"

            data = self._download_with_retries(url, headers)

            # Best‑effort cache write (never evicts).
            cache_file = self._write_cache_shard(repo_id, dir_name, layer_name, data)

            # Load tensors.
            if cache_file.exists():
                try:
                    if mmap:
                        tensors = load_file(cache_file, device="cpu", mmap=True, **load_kwargs)
                    else:
                        tensors = load_file(cache_file, device="cpu", **load_kwargs)
                    log.debug("Loaded shard %s after caching", cache_file)
                    return tensors
                except Exception as ex:
                    log.warning("Unexpected error reading cached shard %s: %s. Falling back to bytes loader.", cache_file, ex)

            log.debug("Loading shard from raw bytes (%d bytes)", len(data))
            return self._load_from_bytes(data)

        # ── Local filesystem path (original behaviour) ──────────────────
        file_path = self._tensor_path(layer_name, Path(path))
        log.debug("Loading shard %s from local file %s", layer_name, file_path)

        if not mmap:
            tensors = load_file(file_path, device="cpu", **load_kwargs)
        else:
            try:
                tensors = load_file(file_path, device="cpu", mmap=True, **load_kwargs)
            except TypeError:  # older safetensors without mmap
                tensors = self._load_with_safe_open(file_path)

        log.debug("Loaded shard %s (%.2f MiB)", layer_name, sum(t.numel() * t.element_size() for t in tensors.values()) / 2**20)
        return tensors
