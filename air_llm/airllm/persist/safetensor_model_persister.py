import os, io, requests
from pathlib import Path
from typing import Union, Dict, Tuple

from .model_persister import ModelPersister
from safetensors.torch import load_file, save_file

# `safe_open` exists in every release; we import lazily in case it’s unused.
from importlib import import_module

PathLike = Union[str, Path]


class SafetensorModelPersister(ModelPersister):
    """
    Persist layers as `<layer_name>.safetensors` and support mmap sharing.

    * If the installed safetensors supports `mmap=…`, we use it.
    * Otherwise we fall back to `safe_open()` which never copies the bytes,
      so multiple processes still share the page-cache.

    New in June 2025
    ----------------
    Recognises streaming URIs of the form
        ``hf://<repo_id>/splitted_model[.<compression>]``
    and downloads individual shards straight from the Hugging Face Hub
    into memory, bypassing disk entirely.  Any failure automatically
    falls back to the original on-disk code-path.
    """

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _filename(layer_name: str) -> str:
        # remove a trailing dot to avoid “layer..safetensors”
        return f"{layer_name.rstrip('.')}.safetensors"

    def _tensor_path(self, layer_name: str, base: Path) -> Path:
        return base / self._filename(layer_name)

    def _done_path(self, layer_name: str, base: Path) -> Path:
        return base / f"{self._filename(layer_name)}.done"

    # ------------------------------------------------------------------ #
    # remote-stream helpers                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_hf_stream_path(hf_path: str) -> Tuple[str, str]:
        """
        Split ``hf://<repo_id>/<dir>`` into ``(<repo_id>, <dir>)``.
        """
        assert hf_path.startswith("hf://"), "not an hf-stream path"
        stripped   = hf_path[5:]
        split_key  = "/splitted_model"
        split_at   = stripped.index(split_key)          # raises if malformed
        repo_id    = stripped[:split_at]
        dir_name   = stripped[split_at + 1:]            # keep folder name
        return repo_id, dir_name

    def _load_from_bytes(self, data: bytes) -> Dict[str, "torch.Tensor"]:
        """
        Zero-copy loader identical to `_load_with_safe_open` but operates on an
        in-memory buffer instead of a file.
        """
        safe_open = import_module("safetensors.torch").safe_open
        tensors: Dict[str, "torch.Tensor"] = {}
        with safe_open(data, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        return tensors

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #

    def persist_model(
        self, state_dict: Dict[str, "torch.Tensor"], layer_name: str, saving_path: PathLike
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
        """
        Fallback loader that never copies data: tensors read directly from the
        underlying mmap’d pages (works on *all* safetensors versions).
        """
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
        """
        Load a safetensor shard onto CPU.

        • If `path` begins with ``hf://`` the shard is streamed directly from
          the Hub into RAM ― never written to disk.

        • Otherwise we honour `mmap` when possible and fall back to
          `safe_open()` when the installed safetensors lacks that flag.
        """
        # ── Remote streaming path ───────────────────────────────────────
        if str(path).startswith("hf://"):
            from huggingface_hub import hf_hub_url

            repo_id, dir_name = self._parse_hf_stream_path(str(path))
            url = hf_hub_url(
                repo_id,
                f"{dir_name}/{self._filename(layer_name)}",
            )

            headers = {}
            _tok = os.environ.get("HF_TOKEN")
            if _tok:
                headers["Authorization"] = f"Bearer {_tok}"

            try:
                with requests.get(url, headers=headers, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    data = r.content
                return self._load_from_bytes(data)
            except Exception as ex:
                # Gracefully degrade to the disk-cache loader
                print(f"[SafetensorModelPersister] stream-load failed: {ex}")

        # ── Local filesystem path (original behaviour) ──────────────────
        file_path = self._tensor_path(layer_name, Path(path))

        if not mmap:
            return load_file(file_path, device="cpu", **load_kwargs)

        # Fast path: load_file(..., mmap=True)
        try:
            return load_file(file_path, device="cpu", mmap=True, **load_kwargs)
        except TypeError:
            # Older safetensors: silently switch to safe_open()
            return self._load_with_safe_open(file_path)
