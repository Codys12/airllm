import os
from pathlib import Path

from .model_persister import ModelPersister
from safetensors.torch import load_file, save_file


class SafetensorModelPersister(ModelPersister):
    """
    Store every layer as `<layer_name>.safetensors` plus a `.done` marker once
    the write finishes.  Reads now accept `mmap=True` so multiple processes
    map the same file-backed pages instead of re-reading the bytes from disk.
    """

    # ------------------------------------------------------------------ #
    # construction                                                       #
    # ------------------------------------------------------------------ #

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #

    def _tensor_path(self, layer_name: str, base_path: Path) -> Path:
        """Return `<base>/<layer_name>.safetensors`."""
        return base_path / f"{layer_name}.safetensors"

    def _done_path(self, layer_name: str, base_path: Path) -> Path:
        """Return `<base>/<layer_name>.safetensors.done`."""
        return base_path / f"{layer_name}.safetensors.done"

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #

    def model_persist_exist(self, layer_name: str, saving_path: Path) -> bool:
        """True when both the tensor file and its '.done' marker exist."""
        return (
            self._tensor_path(layer_name, saving_path).exists()
            and self._done_path(layer_name, saving_path).exists()
        )

    def persist_model(self, state_dict, layer_name: str, saving_path: Path) -> None:
        """
        Save *state_dict* and drop a `.done` flag so other workers know the
        file is complete.
        """
        tensor_path = self._tensor_path(layer_name, saving_path)
        save_file(state_dict, tensor_path)
        print(f"saved as: {tensor_path}")

        # mark completion
        self._done_path(layer_name, saving_path).touch()

    # ------------------------------------------------------------------ #
    # loading                                                            #
    # ------------------------------------------------------------------ #

    def load_model(
        self,
        layer_name: str,
        path: str | Path,
        *,
        mmap: bool = False,
        **load_kwargs,
    ):
        """
        Load a layer onto CPU.

        Parameters
        ----------
        layer_name : str
            Name of the layer, e.g. "layer11".
        path : str | Path
            Directory containing `<layer_name>.safetensors`.
        mmap : bool, default False
            If True, open the file via mmap so every process shares the same
            file-backed pages (cuts redundant disk I/O).
        **load_kwargs :
            Forwarded unmodified to `safetensors.torch.load_file` so future
            options (dtype, device_map, etc.) remain compatible.
        """
        file_path = self._tensor_path(layer_name, Path(path))
        return load_file(file_path, device="cpu", mmap=mmap, **load_kwargs)
