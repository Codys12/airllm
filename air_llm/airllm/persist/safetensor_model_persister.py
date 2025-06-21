import os
from pathlib import Path
from typing import Dict, Union
from importlib import import_module

from .model_persister import ModelPersister
from safetensors.torch import load_file, save_file


PathLike = Union[str, Path]


class SafetensorModelPersister(ModelPersister):
    """
    Persist layers exactly as you already have them on disk:

        <layer_name> + ".safetensors"

    That means if *layer_name* already ends with a trailing dot
    (e.g. ``model.layers.0.``) the file on disk is
    ``model.layers.0..safetensors`` — the double-dot variant you generated
    earlier.  This class also supports mmap-aware loading, with a transparent
    fallback for old *safetensors* versions that lack the keyword.
    """

    # ───────────────────────── helpers ────────────────────────── #

    @staticmethod
    def _filename(layer_name: str) -> str:
        """
        Return the on-disk filename.  Deliberately **does not** strip a trailing
        dot, so callers like `"model.layers.0."` map to
        `"model.layers.0..safetensors"`.
        """
        return f"{layer_name}.safetensors"

    def _tensor_path(self, layer_name: str, base: Path) -> Path:
        return base / self._filename(layer_name)

    def _done_path(self, layer_name: str, base: Path) -> Path:
        return base / f"{self._filename(layer_name)}.done"

    # ───────────────────────── saving ─────────────────────────── #

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

    # ───────────────────── existence check ────────────────────── #

    def model_persist_exist(self, layer_name: str, saving_path: PathLike) -> bool:
        saving_path = Path(saving_path)
        return (
            self._tensor_path(layer_name, saving_path).exists()
            and self._done_path(layer_name, saving_path).exists()
        )

    # ───────────────────────── loading ────────────────────────── #

    def _load_with_safe_open(self, file_path: Path) -> Dict[str, "torch.Tensor"]:
        """
        Fallback loader that works on all safetensors versions and still maps
        the data directly from the file (no extra copies).
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
        Load `<layer_name>.safetensors` to CPU.

        * If safetensors >= 0.4.0 is installed, we honour `mmap=…`.
        * On older versions we transparently fall back to `safe_open()`.
        """
        file_path = self._tensor_path(layer_name, Path(path))

        if not mmap:
            # Caller doesn’t care about mmap; use the plain helper.
            return load_file(file_path, device="cpu", **load_kwargs)

        # Try the fast path with mmap
        try:
            return load_file(file_path, device="cpu", mmap=True, **load_kwargs)
        except TypeError:
            # Old safetensors — silently switch to safe_open()
            return self._load_with_safe_open(file_path)
