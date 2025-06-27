import gc
import json
import os
import ctypes
import shutil
import io
import requests
from tqdm import tqdm
from pathlib import Path
from glob import glob
import time
import logging
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple, Union
from sys import platform

import huggingface_hub
from huggingface_hub import hf_hub_url, model_info

is_on_mac_os = platform == "darwin"

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from .persist import ModelPersister

try:
    import bitsandbytes as bnb

    bitsandbytes_installed = True
except ImportError:
    bitsandbytes_installed = False


# ───────────────────────────────────────────────────────────────────────────────
# helpers & utilities
# ───────────────────────────────────────────────────────────────────────────────

class NotEnoughSpaceException(Exception):
    pass


# --------------------------------------------------------------------- #
# memory helpers                                                        #
# --------------------------------------------------------------------- #
def clean_memory() -> None:
    """Free RAM and GPU VRAM."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        # Non-Linux platforms just ignore
        pass
    torch.cuda.empty_cache()


# --------------------------------------------------------------------- #
# (de)compression helpers                                               #
# --------------------------------------------------------------------- #
def save_quant_state_to_dict(self, packed: bool = True):
    """
    Replacement for `bnb.quantstate.as_dict` until upstream bug is fixed.
    Returns dict of tensors (ready for `state_dict`) plus metadata.
    """
    qs_dict = {
        'quant_type':  self.quant_type,
        'absmax':      self.absmax,
        'blocksize':   self.blocksize,
        'quant_map':   self.code,
        'dtype':       str(self.dtype).strip('torch.'),
        'shape':       tuple(self.shape),
    }
    if self.nested:
        qs_dict.update({
            'nested_absmax':   self.state2.absmax,
            'nested_blocksize': self.state2.blocksize,
            'nested_quant_map': self.state2.code,
            'nested_dtype':     str(self.state2.dtype).strip('torch.'),
            'nested_offset':    self.offset.item(),
        })

    if not packed:
        return qs_dict

    qs_packed_dict = {k: v for k, v in qs_dict.items() if isinstance(v, torch.Tensor)}
    non_tensor_dict = {k: v for k, v in qs_dict.items() if not isinstance(v, torch.Tensor)}
    qs_packed_dict["quant_state." + "bitsandbytes__" + self.quant_type] = \
        bnb.utils.pack_dict_to_tensor(non_tensor_dict)
    return qs_packed_dict


def uncompress_layer_state_dict(layer_state_dict: Dict[str, torch.Tensor]):
    """
    Reverse either 4-bit NF4 or 8-bit blockwise compression (if present).
    Otherwise returns `layer_state_dict` unchanged.
    """
    uncompressed_layer_state_dict = None

    if any('4bit' in k for k in layer_state_dict.keys()):
        uncompressed_layer_state_dict = {}
        for k, v in layer_state_dict.items():
            if '4bit' not in k:
                quant_state_dict = {
                    kk[len(k):]: kv
                    for kk, kv in layer_state_dict.items()
                    if kk.startswith(k) and k != kk
                }
                quant_state = bnb.functional.QuantState.from_dict(
                    qs_dict=quant_state_dict, device="cuda"
                )
                dqv = bnb.functional.dequantize_nf4(v.cuda(), quant_state)
                uncompressed_layer_state_dict[k] = dqv
        del layer_state_dict

    elif any('8bit' in k for k in layer_state_dict.keys()):
        uncompressed_layer_state_dict = {}
        for k, v in layer_state_dict.items():
            if '8bit' not in k:
                absmax = layer_state_dict[k + ".8bit.absmax"]
                code   = layer_state_dict[k + ".8bit.code"]
                dqv = bnb.functional.dequantize_blockwise(
                    v.cuda(),
                    bnb.functional.QuantState(
                        absmax=absmax.cuda(),
                        code=code.cuda(),
                        blocksize=2048,
                        dtype=torch.float16,
                    ),
                )
                uncompressed_layer_state_dict[k] = dqv
        del layer_state_dict

    return (
        layer_state_dict
        if uncompressed_layer_state_dict is None
        else uncompressed_layer_state_dict
    )


# --------------------------------------------------------------------- #
# NEW: remote streaming helpers                                         #
# --------------------------------------------------------------------- #
def _stream_layer_from_hub(hf_path: str, layer_name: str) -> Dict[str, torch.Tensor]:
    """
    Stream a single safetensor shard from the Hub **directly into RAM**.
    The path must use the special scheme ``hf://<repo_id>/<splitted_dir>``.
    """
    from safetensors.torch import safe_open

    assert hf_path.startswith("hf://"), "remote path must start with 'hf://'"
    stripped     = hf_path[5:]                       # drop scheme
    split_key    = "/splitted_model"
    split_at     = stripped.index(split_key)         # raises ValueError if malformed
    repo_id      = stripped[:split_at]
    splitted_dir = stripped[split_at + 1:]           # keep leading folder

    filename = f"{layer_name.rstrip('.')}.safetensors"
    url = hf_hub_url(
        repo_id,
        f"{splitted_dir}/{filename}",
    )

    # ── auth header because `hf_hub_url` no longer accepts `token` ──
    headers = {}
    _tok = os.environ.get("HF_TOKEN")
    if _tok:
        headers["Authorization"] = f"Bearer {_tok}"

    # stream into memory
    with requests.get(url, headers=headers, stream=True, timeout=30) as r:
        r.raise_for_status()
        buffer = io.BytesIO(r.content)

    state_dict: Dict[str, torch.Tensor] = {}
    with safe_open(buffer.getvalue(), framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
    return state_dict


# --------------------------------------------------------------------- #
# layer I/O                                                             #
# --------------------------------------------------------------------- #
def load_layer(local_path, layer_name, profiling: bool = False):
    """
    Load one layer shard.  Supports three scenarios:

    1. **Remote split dir** – `local_path` starts with ``hf://`` → stream.
    2. **Local safetensor shard** – use `ModelPersister` with mmap=True.
    3. **Profiling** – return (state_dict, compression_overhead).
    """
    # ── 1️⃣ remote streaming ───────────────────────────────────────────
    if str(local_path).startswith("hf://"):
        t0 = time.process_time() if profiling else None
        layer_state_dict = _stream_layer_from_hub(str(local_path), layer_name)
        uncompressed     = uncompress_layer_state_dict(layer_state_dict)
        if profiling:
            return uncompressed, (time.process_time() - t0)
        return uncompressed

    # ── 2️⃣ local filesystem (original behaviour) ──────────────────────
    layer_state_dict = ModelPersister.get_model_persister().load_model(
        layer_name, local_path, mmap=True
    )

    if profiling:
        t = time.process_time()

    to_return = uncompress_layer_state_dict(layer_state_dict)

    if profiling:
        elapsed_time = time.process_time() - t
        return to_return, elapsed_time
    return to_return


# --------------------------------------------------------------------- #
# disk-space sanity check                                               #
# --------------------------------------------------------------------- #
def check_space(
    checkpoint_path: Path,
    layer_shards_saving_path: Optional[str] = None,
    compression: Optional[str] = None,
    splitted_model_dir_name: str = 'splitted_model',
):
    """
    Ensure there is enough disk space to create (or re-create) split shards.
    Raises NotEnoughSpaceException if insufficient.
    """
    total_shard_files_size_bytes = sum(
        os.path.getsize(p) for p in glob(str(checkpoint_path / '*'))
    )

    total_saved_split_files_size_bytes = 0
    if layer_shards_saving_path is not None:
        total_saved_split_files_size_bytes = sum(
            os.path.getsize(p)
            for p in glob(
                str(Path(layer_shards_saving_path) / splitted_model_dir_name / '*')
            )
        )

    if compression == '4bit':
        total_shard_files_size_bytes = int(total_shard_files_size_bytes / 0.2813)
    elif compression == '8bit':
        total_shard_files_size_bytes = total_shard_files_size_bytes // 2

    _, _, free = shutil.disk_usage(
        checkpoint_path if layer_shards_saving_path is None else layer_shards_saving_path
    )

    if free + total_saved_split_files_size_bytes < total_shard_files_size_bytes:
        raise NotEnoughSpaceException(
            f"Not enough space: free={free/1e9:.02f} GB, "
            f"needed≈{total_shard_files_size_bytes/1e9:.02f} GB"
        )


# --------------------------------------------------------------------- #
# compression helpers                                                   #
# --------------------------------------------------------------------- #
def compress_layer_state_dict(layer_state_dict, compression: Optional[str] = None):
    """
    Compress a state_dict to 4-bit NF4 or 8-bit blockwise if requested.
    """
    if compression not in ('4bit', '8bit'):
        return layer_state_dict  # no compression

    assert bitsandbytes_installed, "bitsandbytes is required for compression"

    compressed_layer_state_dict: Dict[str, torch.Tensor] = {}
    if compression == '4bit':
        for k, v in layer_state_dict.items():
            v_quant, quant_state = bnb.functional.quantize_nf4(v.cuda(), blocksize=64)
            compressed_layer_state_dict[k] = v_quant
            for qs_k, qs_v in save_quant_state_to_dict(quant_state).items():
                compressed_layer_state_dict[f"{k}.4bit.{qs_k}"] = qs_v

    elif compression == '8bit':
        for k, v in layer_state_dict.items():
            v_quant, quant_state = bnb.functional.quantize_blockwise(
                v.cuda(), blocksize=2048
            )
            compressed_layer_state_dict[k] = v_quant
            compressed_layer_state_dict[f"{k}.8bit.absmax"] = quant_state.absmax.clone().contiguous()
            compressed_layer_state_dict[f"{k}.8bit.code"]   = quant_state.code.clone().contiguous()

    return compressed_layer_state_dict


# --------------------------------------------------------------------- #
# file-deletion helper                                                  #
# --------------------------------------------------------------------- #
def remove_real_and_linked_file(to_delete: str) -> None:
    """
    Delete `to_delete`.  If it is a symlink, also delete its real target.
    Any errors are caught and logged instead of propagating.
    """
    targetpath = None
    try:
        if os.path.realpath(to_delete) != to_delete:
            targetpath = os.path.realpath(to_delete)

        os.remove(to_delete)

        if targetpath:
            try:
                os.remove(targetpath)
            except FileNotFoundError:
                logging.debug("Target file %s was already absent", targetpath)

    except FileNotFoundError as e:
        logging.warning("File not found: %s", e.filename)
    except PermissionError as e:
        logging.error("Permission denied while deleting %s", e.filename)
    except OSError:
        logging.exception("Unexpected error while deleting files")


# --------------------------------------------------------------------- #
# splitting & saving                                                    #
# (unchanged except for minor cosmetic tweaks)                          #
# --------------------------------------------------------------------- #
def split_and_save_layers(
    checkpoint_path,
    layer_shards_saving_path: Optional[str] = None,
    splitted_model_dir_name: str = 'splitted_model',
    compression: Optional[str] = None,
    layer_names: Optional[dict] = None,
    delete_original: bool = False,
    repo_id: Optional[str] = None,
    hf_token: Optional[str] = None,
):
    """
    Convert original HF shards into per-layer safetensors (optionally
    compressed) and store them under `<splitted_model_dir_name>[.<compression>]`.
    """
    if compression is not None:
        assert bitsandbytes_installed, "compression requires bitsandbytes"
        splitted_model_dir_name += f".{compression}"

    checkpoint_path = Path(checkpoint_path)
    saving_path = (
        Path(layer_shards_saving_path) / splitted_model_dir_name
        if layer_shards_saving_path is not None
        else checkpoint_path / splitted_model_dir_name
    )

    safetensors_format = checkpoint_path.joinpath('model.safetensors.index.json').exists()
    index_file = (
        checkpoint_path / ('model.safetensors.index.json' if safetensors_format
                           else 'pytorch_model.bin.index.json')
    )
    with open(index_file, 'rb') as f:
        index = json.load(f)['weight_map']

    if layer_names is None:
        n_layers = len({int(k.split('.')[2]) for k in index if 'model.layers' in k})
        layers = (
            ['model.embed_tokens.']
            + [f'model.layers.{i}.' for i in range(n_layers)]
            + ['model.norm.', 'lm_head.']
        )
    else:
        n_layers = len({int(k[len(layer_names['layer_prefix']):].split('.')[1])
                        for k in index if layer_names['layer_prefix'] in k})
        layers = (
            [layer_names['embed']]
            + [f"{layer_names['layer_prefix']}.{i}" for i in range(n_layers)]
            + [layer_names['norm'], layer_names['lm_head']]
        )
        layers = [l + "." for l in layers]

    # short-circuit if all shards already exist
    if saving_path.exists():
        found = {
            layer: ModelPersister.get_model_persister().model_persist_exist(layer, saving_path)
            for layer in layers
        }
        if all(found.values()):
            print(f"[split-layers] using existing split dir: {saving_path}")
            return str(saving_path)
        print("[split-layers] some split files missing – re-saving all layers")

    if not delete_original:
        check_space(checkpoint_path, layer_shards_saving_path, compression,
                    splitted_model_dir_name=splitted_model_dir_name)

    shard = -1
    n_shards = len(set(index.values()))
    state_dict = {}

    saving_path.mkdir(parents=True, exist_ok=True)

    for layer in tqdm(layers, desc="splitting model"):
        # gather the shards that contain tensors for this layer
        layer_shards = sorted({int(v.split('-')[1]) for k, v in index.items()
                               if k.startswith(layer)})

        for target_shard in layer_shards:
            if target_shard != shard:
                if delete_original and shard != -1:
                    fname = (
                        f"model-{shard:05d}-of-{n_shards:05d}.{'safetensors' if safetensors_format else 'bin'}"
                    )
                    remove_real_and_linked_file(checkpoint_path / fname)

                shard = target_shard
                fname = (
                    f"{'model' if safetensors_format else 'pytorch_model'}-{shard:05d}-of-{n_shards:05d}."
                    f"{'safetensors' if safetensors_format else 'bin'}"
                )
                to_load = checkpoint_path / fname
                if not to_load.exists():
                    assert repo_id is not None
                    huggingface_hub.snapshot_download(
                        repo_id, allow_patterns=os.path.basename(to_load), token=hf_token
                    )
                state_dict.update(
                    load_file(to_load, device='cpu')
                    if safetensors_format
                    else torch.load(to_load, map_location='cpu')
                )

        layer_state_dict = {
            k: v for k, v in state_dict.items() if k.startswith(layer)
        }
        layer_state_dict = compress_layer_state_dict(
            layer_state_dict, compression
        )

        if not ModelPersister.get_model_persister().model_persist_exist(layer, saving_path):
            ModelPersister.get_model_persister().persist_model(
                layer_state_dict, layer, saving_path
            )

        for k in list(layer_state_dict.keys()):
            if k in state_dict:
                del state_dict[k]
        del layer_state_dict
        clean_memory()

    return str(saving_path)


# --------------------------------------------------------------------- #
# NEW: locate or create split dir                                       #
# --------------------------------------------------------------------- #
def find_or_create_local_splitted_path(
    model_local_path_or_repo_id: str,
    layer_shards_saving_path: Optional[str] = None,
    compression: Optional[str] = None,
    layer_names: Optional[dict] = None,
    hf_token: Optional[str] = None,
    delete_original: bool = False,
) -> Tuple[str, str]:
    """
    Resolve the model:

    • If `model_local_path_or_repo_id` is an existing directory with an index
      file, we split locally (or reuse existing split dir).

    • If it’s a *repo id* whose *repo already hosts* a `splitted_model*`
      folder, we stream from the Hub → checkpoint path becomes
      ``hf://<repo_id>/splitted_model[.<compression>]`` (never touches disk).

    • Otherwise we download the raw shards to cache and split on disk.
    """
    # ── 0️⃣ LOCAL DIRECTORY ─────────────────────────────────────────────
    if os.path.exists(model_local_path_or_repo_id):
        p = Path(model_local_path_or_repo_id)
        if (p / 'pytorch_model.bin.index.json').exists() or \
           (p / 'model.safetensors.index.json').exists():
            return p, split_and_save_layers(
                p, layer_shards_saving_path, compression=compression,
                layer_names=layer_names, delete_original=delete_original
            )
        print(f"[utils] Directory {p} exists but model index missing – "
              f"treating '{model_local_path_or_repo_id}' as HF repo id")

    # ── 1️⃣ REMOTE PRE-SPLIT SHARDS?  ───────────────────────────────────
    try:
        info = model_info(model_local_path_or_repo_id, token=hf_token)
        files = {s.rfilename for s in info.siblings}

        want_dir = f"splitted_model{'.' + compression if compression else ''}"
        fallbacks = ['splitted_model', 'splitted_model.4bit', 'splitted_model.8bit']

        remote_dir = None
        for d in (want_dir, *fallbacks):
            if any(f.startswith(d + '/') for f in files):
                remote_dir = d
                break

        if remote_dir:
            print(f"[utils] Remote split dir found → streaming from Hub ({remote_dir})")
            return (
                model_local_path_or_repo_id,                           # model_local_path
                f"hf://{model_local_path_or_repo_id}/{remote_dir}",    # checkpoint_path
            )
    except Exception:
        # any error → fall back to local split workflow
        pass

    # ── 2️⃣ FALLBACK: download raw shards and split locally ────────────
    hf_cache_path = huggingface_hub.snapshot_download(
        model_local_path_or_repo_id,
        token=hf_token,
        ignore_patterns=['*.safetensors', '*.bin'],
    )

    return Path(hf_cache_path), split_and_save_layers(
        hf_cache_path, layer_shards_saving_path, compression=compression,
        layer_names=layer_names, delete_original=delete_original,
        repo_id=model_local_path_or_repo_id, hf_token=hf_token
    )
