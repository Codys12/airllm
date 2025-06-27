import os
import requests
import shutil
from pathlib import Path
from typing import Union, Dict, Tuple

from .model_persister import ModelPersister
from safetensors.torch import load_file, save_file
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

# `safe_open` exists in every release; we import lazily in case it’s unused.
from importlib import import_module

PathLike = Union[str, Path]


class SafetensorModelPersister(ModelPersister):
    """
    Persist layers as `<layer_name>.safetensors` and support mmap sharing.

    * If the installed safetensors supports `mmap=…`, we use it.
    * Otherwise we fall back to `safe_open()` which never copies the bytes,
      so multiple processes still share the page‑cache.

    Streaming support (June 2025)
    -----------------------------
    Recognises URIs of the form
        ``hf://<repo_id>/splitted_model[.<compression>]``
    and downloads individual shards straight from the Hugging Face Hub.
    Once the payload is in RAM we try to *cache it on disk* inside the
    standard Hugging Face cache dir so subsequent loads hit the filesystem
    instead of the network—provided there is enough free space.
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

    # ------------------------------------------------------------------ #
    # remote‑stream helpers                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_hf_stream_path(hf_path: str) -> Tuple[str, str]:
        """Split ``hf://<repo_id>/<dir>`` → ``(<repo_id>, <dir>)``."""
        assert hf_path.startswith("hf://"), "not an hf‑stream path"
        stripped = hf_path[5:]
        key = "/splitted_model"
        idx = stripped.index(key)  # raises if malformed
        repo_id = stripped[:idx]
        dir_name = stripped[idx + 1 :]  # keep folder name itself
        return repo_id, dir_name

    def _load_from_bytes(self, data: bytes) -> Dict[str, "torch.Tensor"]:
        """Zero‑copy tensor loader directly from a bytes object."""
        safe_open = import_module("safetensors.torch").safe_open
        tensors: Dict[str, "torch.Tensor"] = {}
        with safe_open(data, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        return tensors

    # ------------------------------------------------------------------ #
    # caching helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _enough_space(dir_path: Path, size_bytes: int, keep_free: int = 2_000_000_000) -> bool:
        """True if writing *size_bytes* leaves ≥ *keep_free* bytes free."""
        try:
            usage = shutil.disk_usage(dir_path)
        except FileNotFoundError:
            # If the path doesn’t exist yet (e.g. first‑time), assume OK.
            return True
        return usage.free - size_bytes > keep_free

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

        shard_path.parent.mkdir(parents=True, exist_ok=True)

        if not self._enough_space(cache_root, len(data)):
            # Insufficient space → skip quietly.
            return

        tmp = shard_path.with_suffix(shard_path.suffix + ".tmp")
        try:
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, shard_path)
        except Exception:
            # Never fail model load because of caching.
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #

    def persist_model(
        self,
        state_dict: Dict[str, "torch.Tensor"],
        layer_name: str,
        saving_path: PathLike,
    ) -> None:
        saving_path = Path(saving_path)
        tensor_path = self._tensor_path(layer_name, saving_path)
        save_file(state_dict, tensor_path)
        print(f"saved as: {tensor_path}")
        self._done_path(layer_name, saving_path).touch()

    def model_persist_exist(self, layer_name: str, saving_path: PathLike) -> bool:
        saving_path = Path(saving_path)
        return (
            self._tensor_path(layer_name, saving_path).exists()
            and self._done_path(layer_name, saving_path).exists()
        )

    # ------------------------------------------------------------------ #
    # loading                                                            #
    # ------------------------------------------------------------------ #

    def _load_with_safe_open(self, file_path: Path) -> Dict[str, "torch.Tensor"]:
        """Fallback loader that never copies data (mmap via safe_open)."""
        safe_open = import_module("safetensors.torch").safe_open
        state = {}
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
    ):
        """Load a safetensor shard onto CPU.

        • If *path* starts with ``hf://`` the shard is streamed from the Hub
          and optionally cached on disk.

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
            except Exception as ex:
                # Gracefully fall back to disk path
                print(f"[SafetensorModelPersister] stream‑load failed: {ex}")

        # ── Local filesystem path (original behaviour) ──────────────────
        file_path = self._tensor_path(layer_name, Path(path))

        if not mmap:
            return load_file(file_path, device="cpu", **load_kwargs)

        try:
            return load_file(file_path, device="cpu", mmap=True, **load_kwargs)
        except TypeError:
            return self._load_with_safe_open(file_path)
