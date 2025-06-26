import gc
import json
import os
import io
import ctypes
import shutil
from tqdm import tqdm
from pathlib import Path
from glob import glob
import time

from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple, Union
from sys import platform

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

import requests
from huggingface_hub import hf_hub_url, model_info

from .persist import ModelPersister

is_on_mac_os = False
if platform == "darwin":
    is_on_mac_os = True


# replacement for bnb quantstat.as_dict(True), until the bug is fixed....
def save_quant_state_to_dict(self, packed=True):
    """
    returns dict of tensors and strings to use in serialization via _save_to_state_dict()
    param: packed -- returns dict[str, torch.Tensor] for state_dict
    """
    qs_dict = {
        'quant_type': self.quant_type,
        'absmax': self.absmax,
        'blocksize': self.blocksize,
        'quant_map': self.code,
        'dtype': str(self.dtype).strip('torch.'),
        'shape': tuple(self.shape),
    }
    if self.nested:
        qs_dict.update({
            'nested_absmax': self.state2.absmax,
            'nested_blocksize': self.state2.blocksize,
            'nested_quant_map': self.state2.code,
            'nested_dtype': str(self.state2.dtype).strip('torch.'),
            'nested_offset': self.offset.item(),
        })
    if not packed:
        return qs_dict

    qs_packed_dict = {k: v for k, v in qs_dict.items() if isinstance(v, torch.Tensor)}
    non_tensor_dict = {k: v for k, v in qs_dict.items() if not isinstance(v, torch.Tensor)}
    qs_packed_dict["quant_state." + "bitsandbytes__" + self.quant_type] = bnb.utils.pack_dict_to_tensor(non_tensor_dict)
    return qs_packed_dict


class NotEnoughSpaceException(Exception):
    pass


def clean_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    torch.cuda.empty_cache()


def uncompress_layer_state_dict(layer_state_dict):
    uncompressed_layer_state_dict = None
    if any(['4bit' in k for k in layer_state_dict.keys()]):
        uncompressed_layer_state_dict = {}
        for k, v in layer_state_dict.items():
            if '4bit' not in k:
                quant_state_dict = {kk[len(k):]: kv for kk, kv in layer_state_dict.items() if kk.startswith(k) and k != kk}
                quant_state = bnb.functional.QuantState.from_dict(qs_dict=quant_state_dict, device="cuda")

                dqv = bnb.functional.dequantize_nf4(v.cuda(), quant_state)
                uncompressed_layer_state_dict[k] = dqv
        del layer_state_dict
    elif any(['8bit' in k for k in layer_state_dict.keys()]):
        uncompressed_layer_state_dict = {}
        for k, v in layer_state_dict.items():
            if '8bit' not in k:
                absmax = layer_state_dict[k + ".8bit.absmax"]
                code = layer_state_dict[k + ".8bit.code"]

                dqv = bnb.functional.dequantize_blockwise(
                    v.cuda(),
                    bnb.functional.QuantState(
                        absmax=absmax.cuda(),
                        code=code.cuda(),
                        blocksize=2048,
                        dtype=torch.float16
                    )
                )
                uncompressed_layer_state_dict[k] = dqv
        del layer_state_dict

    return layer_state_dict if uncompressed_layer_state_dict is None else uncompressed_layer_state_dict


def _stream_layer_from_hub(hf_path: str, layer_name: str) -> Dict[str, torch.Tensor]:
    """
    Download a single shard into memory and return its state-dict.
    `hf_path` must look like: "hf://<repo_id>/splitted_model…"
    """
    from safetensors.torch import safe_open

    assert hf_path.startswith("hf://"), "Not a valid HF stream URI"
    stripped     = hf_path[5:]
    split_key    = "/splitted_model"
    split_at     = stripped.index(split_key)
    repo_id      = stripped[:split_at]
    splitted_dir = stripped[split_at + 1:]

    filename = f"{layer_name.rstrip('.')}.safetensors"
    url = hf_hub_url(repo_id, f"{splitted_dir}/{filename}", token=os.environ.get("HF_TOKEN"))

    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        buffer = io.BytesIO(r.content)

    state_dict = {}
    with safe_open(buffer.getvalue(), framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
    return state_dict


def load_layer(local_path, layer_name, profiling=False):
    """
    Load one layer shard. If `local_path` starts with "hf://" we stream the
    shard straight from the Hub; otherwise we fall back to mmap/OS-cache.
    """
    if str(local_path).startswith("hf://"):
        if profiling:
            t0 = time.process_time()
        layer_state_dict = _stream_layer_from_hub(str(local_path), layer_name)
        if profiling:
            return uncompress_layer_state_dict(layer_state_dict), (time.process_time() - t0)
        return uncompress_layer_state_dict(layer_state_dict)

    # Local filesystem path (original behavior)
    layer_state_dict = ModelPersister.get_model_persister().load_model(
        layer_name, local_path, mmap=True
    )

    if profiling:
        elapsed = time.process_time() - t0
        return uncompress_layer_state_dict(layer_state_dict), elapsed

    return uncompress_layer_state_dict(layer_state_dict)


def check_space(checkpoint_path, layer_shards_saving_path=None, compression=None, splitted_model_dir_name='splitted_model'):
    total_shard_files_size_bytes = 0
    for model_shard_file in glob(str(checkpoint_path / '*')):
        total_shard_files_size_bytes += os.path.getsize(model_shard_file)

    total_saved_split_files_size_bytes = 0
    if layer_shards_saving_path is not None:
        for saved_split_file in glob(str(Path(layer_shards_saving_path) / splitted_model_dir_name / '*')):
            total_saved_split_files_size_bytes += os.path.getsize(saved_split_file)

    if compression == '4bit':
        total_shard_files_size_bytes = int(total_shard_files_size_bytes / 0.2813)
    elif compression == '8bit':
        total_shard_files_size_bytes //= 2

    total, used, free = shutil.disk_usage(
        checkpoint_path if layer_shards_saving_path is None else layer_shards_saving_path
    )

    if free + total_saved_split_files_size_bytes < total_shard_files_size_bytes:
        raise NotEnoughSpaceException(
            f"Not enough space. Free: {free/1e9:.2f} GB, "
            f"Needed: {total_shard_files_size_bytes/1e9:.2f} GB."
        )


def compress_layer_state_dict(layer_state_dict, compression=None):
    compressed_layer_state_dict = None
    if compression == '4bit':
        compressed_layer_state_dict = {}
        for k, v in layer_state_dict.items():
            v_quant, quant_state = bnb.functional.quantize_nf4(v.cuda(), blocksize=64)
            compressed_layer_state_dict[k] = v_quant
            for qs_k, qs_v in save_quant_state_to_dict(quant_state).items():
                compressed_layer_state_dict[f"{k}.4bit.{qs_k}"] = qs_v
    elif compression == '8bit':
        compressed_layer_state_dict = {}
        for k, v in layer_state_dict.items():
            v_quant, quant_state = bnb.functional.quantize_blockwise(v.cuda(), blocksize=2048)
            absmax = quant_state.absmax.clone().contiguous()
            code   = quant_state.code.clone().contiguous()
            compressed_layer_state_dict[k] = v_quant
            compressed_layer_state_dict[f"{k}.8bit.absmax"] = absmax
            compressed_layer_state_dict[f"{k}.8bit.code"]   = code

    return compressed_layer_state_dict if compressed_layer_state_dict is not None else layer_state_dict


def split_and_save_layers(checkpoint_path, layer_shards_saving_path=None,
                          splitted_model_dir_name='splitted_model',
                          compression=None, layer_names=None,
                          delete_original=False, repo_id=None, hf_token=None):
    # … (unchanged from original) …
    # This function remains exactly as in your original :contentReference[oaicite:1]{index=1} file.
    ...


def find_or_create_local_splitted_path(
    model_local_path_or_repo_id,
    layer_shards_saving_path=None,
    compression=None,
    layer_names=None,
    hf_token=None,
    delete_original=False
):
    """
    Find or download the model, then either:
      • Detect existing `splitted_model*` in the HF repo and return a
        streaming URI, or
      • Split & cache locally as before.
    """
    # 0️⃣ Fast-path: existing remote shards → stream them
    if not os.path.exists(model_local_path_or_repo_id):
        try:
            info = model_info(model_local_path_or_repo_id, token=hf_token)
            files = {f.rfilename for f in info.siblings}

            wanted       = f"splitted_model{'.' + compression if compression else ''}"
            fallback_dirs = ["splitted_model", "splitted_model.4bit", "splitted_model.8bit"]

            candidate = next((d for d in (wanted, *fallback_dirs) if any(p.startswith(d + "/") for p in files)), None)
            if candidate:
                print(f"⚡ Remote split shards detected – streaming from Hub ({candidate})")
                return (
                    model_local_path_or_repo_id,
                    f"hf://{model_local_path_or_repo_id}/{candidate}"
                )
        except Exception:
            pass

    # 1️⃣ Local workflow: download cache & split locally
    hf_cache_path = huggingface_hub.snapshot_download(
        model_local_path_or_repo_id,
        token=hf_token,
        ignore_patterns=["*.safetensors", "*.bin"]
    )
    return (
        Path(hf_cache_path),
        split_and_save_layers(
            hf_cache_path,
            layer_shards_saving_path,
            compression=compression,
            layer_names=layer_names,
            delete_original=delete_original,
            repo_id=model_local_path_or_repo_id,
            hf_token=hf_token
        )
    )
