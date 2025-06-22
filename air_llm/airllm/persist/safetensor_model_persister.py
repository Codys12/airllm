import os
from pathlib import Path
from typing import Union, Dict

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
        Load a safetensor shard onto CPU, honouring `mmap` whenever possible.
        Falls back transparently if the installed safetensors lacks the flag.
        """
        file_path = self._tensor_path(layer_name, Path(path))

        if not mmap:
            # caller doesn't care about mmap – use the regular helper
            return load_file(file_path, device="cpu", **load_kwargs)

        # Try the fast path: load_file(..., mmap=True)
        try:
            return load_file(file_path, device="cpu", mmap=True, **load_kwargs)
        except TypeError:
            # Old safetensors – silently switch to safe_open()
            return self._load_with_safe_open(file_path)
