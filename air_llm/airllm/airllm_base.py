
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel, GenerationMixin, LlamaForCausalLM, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from accelerate import init_empty_weights

from accelerate.utils.modeling import set_module_tensor_to_device
from transformers.quantizers import AutoHfQuantizer, HfQuantizer

from .profiler import LayeredProfiler

from .utils import clean_memory, load_layer, \
    find_or_create_local_splitted_path

try:
    import bitsandbytes as bnbf

    bitsandbytes_installed = True
    print('>>>> bitsandbytes installed')
except ImportError:
    bitsandbytes_installed = False



try:
    from transformers.cache_utils import Cache, DynamicCache

    cache_utils_installed = True
    print('>>>> cache_utils installed')
except ImportError:
    cache_utils_installed = False






class AirLLMBaseModel(GenerationMixin):

    # customize layer names here
    def set_layer_names_dict(self):
        # Added the rotary-embedding entry so the helper that eagerly moves
        # buffers/devices can find it (optional but nice to have)
        self.layer_names_dict = {
            'embed':          'model.embed_tokens',
            'layer_prefix':   'model.layers',
            'norm':           'model.norm',
            'lm_head':        'lm_head',
            'rotary_pos_emb': 'model.rotary_emb',   # NEW
        }



    def __init__(self, model_local_path_or_repo_id, device="cuda:0", dtype=torch.float16, max_seq_len=512,
                 layer_shards_saving_path=None, profiling_mode=False, compression=None,
                 hf_token=None, prefetching=True, delete_original=False):
        """
        Sharded version of LlamaForCausalLM : the model is splitted into layer shards to reduce GPU memory usage.
        During the forward pass, the inputs are processed layer by layer, and the GPU memory is freed after each layer.
        To avoid loading the layers multiple times, we could save all the intermediate activations in RAM.

        Parameters
        ----------
        model_local_path_or_repo_id : str or Path
            path to the local model checkpoint or huggingface repo id
        device : str, optional
            device, by default "cuda:0"
        dtype : torch.dtype, optional
            dtype, by default torch.float16
        max_seq_len : int, optional
            max seq lenght, by default 512
        layer_shards_saving_path : str, optional
            optional path to save layered shards model file, by default just save to the local cache of model, subdir named splitted_model will be saved
        profiling_mode : book, optional
            if to profile the model loading time, default to False
        compression: str, optinal
            setting to '4bit' or '8bit' to enable compression from 16 bits to 4 bits/8 bits which speeed up 4x or 2x inference time with a tiny accuracy loss.
        hf_token: str, optional
            huggingface api token could be provided, by default None
        """


        self.profiling_mode = profiling_mode
        self.profiler = LayeredProfiler()

        self.total_disk_loading_time = None
        self.total_gpu_loading_time = None
        self.total_compression_overhead_time = None
        self._supports_cache_class = False
        self.hf_quantizer = None

        if compression is not None:
            if not bitsandbytes_installed:
                raise ImportError('WARNING: bitsandbytes not found. Compression needs bitsandbytes. To use compression, please install bitsandbytes: `pip install bitsandbytes`')


        self.compression = compression
        self.hf_token = hf_token

        # Save parameters

        self.set_layer_names_dict()


        self.model_local_path, self.checkpoint_path = find_or_create_local_splitted_path(model_local_path_or_repo_id,
                                                                                         layer_shards_saving_path,
                                                                                         compression=compression,
                                                                                         layer_names=self.layer_names_dict,
                                                                                         hf_token=hf_token,
                                                                                         delete_original=delete_original)
        self.running_device = device
        self.device = torch.device(self.running_device)
        self.running_dtype = dtype
        self.dtype = self.running_dtype

        # Create model
        if hf_token is not None:
            self.config = AutoConfig.from_pretrained(self.model_local_path, token=hf_token, trust_remote_code=True)
        else:
            self.config = AutoConfig.from_pretrained(self.model_local_path, trust_remote_code=True, load_in_4bit=True, torch_dtype=torch.bfloat16)

        self.generation_config = self.get_generation_config()
        #print(f"using generation_config: {self.generation_config}")

        self.tokenizer = self.get_tokenizer(hf_token=hf_token)


        self.init_model()

        # get layer count:
        model_attr = self.model
        for attr_name in self.layer_names_dict["layer_prefix"].split("."):
            model_attr = getattr(model_attr, attr_name)

        layers_count = len(model_attr)


        self.layer_names = [self.layer_names_dict['embed']] + [f'{self.layer_names_dict["layer_prefix"]}.{i}' for i in
                                                               range(layers_count)] + \
                           [self.layer_names_dict['norm'], self.layer_names_dict['lm_head']]

        self.max_seq_len = max_seq_len

        self.main_input_name = "input_ids"

        # model weights prefetch cuda stream
        self.prefetching = prefetching

        if self.compression is not None:
            self.prefetching = False
            print(f"not support prefetching for compression for now. loading with no prepetching mode.")

        if prefetching:
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None

    # if derived class needs to create generation config differently, like Mistrial, this function can be overridden
    def get_generation_config(self):
        # protective on generation config

        try:
            return GenerationConfig.from_pretrained(self.model_local_path)
        except Exception as e:
            return GenerationConfig()

    # a chance to customize tokenizer
    def get_tokenizer(self, hf_token=None):
        if hf_token is not None:
            return AutoTokenizer.from_pretrained(self.model_local_path, token=hf_token, trust_remote_code=True)
        else:
            return AutoTokenizer.from_pretrained(self.model_local_path, trust_remote_code=True)

    def get_use_better_transformer(self):
        return True

    def init_model(self):
        
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=True, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)

        quantization_config = getattr(self.config, "quantization_config", None)

        if quantization_config is not None:
            self.hf_quantizer = AutoHfQuantizer.from_config(quantization_config, pre_quantized=True)
            device_map = self.hf_quantizer.update_device_map(None)
            self.hf_quantizer.preprocess_model(model=self.model, device_map=device_map)

        self.model.eval()
        self.model.tie_weights()

        self.set_layers_from_layer_names()

        # ──────────────────────────────────────────────────────────────
        # Handle rotary embeddings *only* if the model provides them
        # ──────────────────────────────────────────────────────────────
        self._rotary_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        if hasattr(self.model, "model") and hasattr(self.model.model, "rotary_emb"):
            # 1⃣  Materialise RoPE once so it never stays on `meta`
            self.model.model.rotary_emb.to(self.running_device)
            has_rope = True
        else:
            has_rope = False

            # Provide a no-op stub so the rest of the code can still call `self._rotary`
            def _rotary_noop(seq_len: int):
                return None

            self._rotary = _rotary_noop.__get__(self, AirLLMBaseModel)  # bind to instance

        # 2⃣  Move the *other* buffers (skip RoPE inv_freq if it was already moved)
        for name, buf in self.model.named_buffers():
            if has_rope and name == "model.rotary_emb.inv_freq":
                continue  # already handled above
            set_module_tensor_to_device(
                self.model,
                name,
                self.running_device,
                value=buf,
                dtype=self.running_dtype,
            )

        # 3⃣  Per-length cache for (cos, sin)
        self._rotary_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    # helper lives here so it has access to self._rotary_cache
    def _rotary(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len not in self._rotary_cache:
            dummy = torch.empty(1, seq_len, self.model.config.hidden_size,
                                device=self.running_device,
                                dtype=self.running_dtype)
            pos   = torch.arange(seq_len, device=self.running_device).unsqueeze(0)
            self._rotary_cache[seq_len] = self.model.model.rotary_emb(dummy, pos)
        return self._rotary_cache[seq_len]

    def set_layers_from_layer_names(self):

        self.layers = []

        model_attr = self.model
        for attr_name in self.layer_names_dict["embed"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

        model_attr = self.model
        for attr_name in self.layer_names_dict["layer_prefix"].split("."):
            model_attr = getattr(model_attr, attr_name)

        self.layers.extend(list(model_attr))

        model_attr = self.model
        for attr_name in self.layer_names_dict["norm"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

        model_attr = self.model
        for attr_name in self.layer_names_dict["lm_head"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

    def load_rotary_pos_emb_to_device(self):
        state_dict = load_layer(self.checkpoint_path, self.layer_names_dict['rotary_pos_emb'])
        self.move_layer_to_device(state_dict)

    def load_layer_to_cpu(self, layer_name):

        t = time.time()

        load_layer_output = load_layer(self.checkpoint_path, layer_name, self.profiling_mode)
        elapsed_time = time.time() - t

        if self.profiling_mode:
            state_dict, compression_time = load_layer_output
            disk_loading_time = elapsed_time - compression_time

            self.profiler.add_profiling_time('load_safe_tensor', disk_loading_time)

            self.profiler.add_profiling_time('compression_time', compression_time)
        else:
            state_dict = load_layer_output

        # pin memory:
        if self.prefetching:
            t = time.time()
            for k in state_dict.keys():
                state_dict[k].pin_memory()

            elapsed_time = time.time() - t
            if self.profiling_mode:
                self.profiler.add_profiling_time('pin_memory_to_trigger_load', elapsed_time)

        return state_dict

    def move_layer_to_device(self, state_dict):
        layers = []
        for param_name, param in state_dict.items():
            if self.hf_quantizer is None:
                layers.append(param_name)
            else:
                if '.weight' in param_name:
                    layer_name = param_name[:param_name.index(".weight") + len(".weight")]
                    if layer_name not in layers:
                        layers.append(layer_name)

        for param_name in layers:
            tensor = state_dict[param_name]

            if (self.hf_quantizer is None or
                not self.hf_quantizer.check_quantized_param(self.model, param_value=tensor, param_name=param_name, state_dict={})
               ):
                set_module_tensor_to_device(self.model, param_name, self.running_device, value=tensor,
                                            dtype=None,
                                            )
            else:
                torch_dtype = self.hf_quantizer.update_torch_dtype(None)
                self.hf_quantizer.create_quantized_param(self.model, state_dict[param_name], param_name, self.running_device, state_dict)
        return layers

    # make GenerationMixin happy
    def can_generate(self):
        return True

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = self.get_past_key_values_cache_seq_len(past_key_values) #[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_past_key_values_cache_seq_len(self, past_key_values):
        return past_key_values[0][0].shape[2]
    def get_sequence_len(self, seq):
        return seq.shape[1]

    def get_pos_emb_args(self, len_p, len_s):
        return {}

    def get_past_key_value_args(self, k_cache, v_cache):
        return {'past_key_value': (k_cache, v_cache)}

    def get_attention_mask_args(self, full_attention_mask, len_p, len_s):
        """
        Ensure the mask is 4-D: (batch, 1, query_len, key_len).
        Users often pass a 2-D mask shaped (B, L); older forks sometimes pass
        (B, 1, L).  Both lead to an IndexError when we slice with four indices.
        """
        if full_attention_mask.dim() == 2:
            # (B, L) ➜ (B, 1, 1, L)
            full_attention_mask = full_attention_mask[:, None, None, :]
        elif full_attention_mask.dim() == 3:
            # (B, 1, L) ➜ (B, 1, L, L) – broadcastable on query axis
            full_attention_mask = full_attention_mask[:, None, :, :]

        return {
            "attention_mask": full_attention_mask[:, :, -len_s:, -len_p - len_s:]
        }

    def get_position_ids_args(self, full_position_ids, len_p, len_s):

        return {'position_ids': full_position_ids[:, len_p:len_p + len_s]}


    def run_lm_head(self, layer, seq, num_samples=32):
        """
        Memory-aware: keep the outer batch loop but remove the inner
        sequence loop using scatter_add on GPU.
        """
        batch_size, seq_len, _ = seq.shape
        results = []

        for i in range(batch_size):                              # per-example processing
            logits = layer(seq[i]).float()                       # [S, V]
            probs  = torch.softmax(logits, dim=-1)
            probs  = torch.nan_to_num(probs,
                                      nan=0.0, posinf=0.0, neginf=0.0)

            samples = torch.multinomial(probs,
                                        num_samples,
                                        replacement=True)        # [S, K]

            counts = torch.zeros(seq_len, probs.size(-1),
                                 device=samples.device,
                                 dtype=torch.int32)
            counts.scatter_add_(1, samples,
                                torch.ones_like(samples,
                                                dtype=counts.dtype))

            counts_topk, ids_topk = counts.float().topk(num_samples, dim=1)
            results.append(torch.stack([ids_topk, counts_topk], dim=1))  # [S, 2, K]

        return torch.stack(results)                              # [B, S, 2, K]

    def run_norm(self, layer, seq):
        return layer(seq)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            num_samples: int = 32,
            minibatch: int = 32,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if cache_utils_installed:
            use_cache = False

        # Move input tensors to the correct device
        input_ids = input_ids.to(self.running_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.running_device)
        if position_ids is not None:
            position_ids = position_ids.to(self.running_device)

        batch_size, seq_len = input_ids.shape

        # Create attention mask and position ids if not provided
        if attention_mask is None:
            attention_mask = torch.ones(self.max_seq_len, self.max_seq_len, device=self.running_device)
            attention_mask = attention_mask.triu(diagonal=1)[None, None, ...] == 0
        if position_ids is None:
            position_ids = torch.arange(self.max_seq_len, dtype=torch.long, device=self.running_device)[None, :]

        hidden_states = None
        all_hidden_states = [] if output_hidden_states else None
        all_self_attns = [] if output_attentions else None
        kv_cache_list = [] if use_cache else None

        with torch.inference_mode(), ThreadPoolExecutor() as executor:
            if self.prefetching:
                future = executor.submit(self.load_layer_to_cpu, self.layer_names[0])
            def assert_no_meta(tensor_module, layer_name):
                metas = [n for n, p in tensor_module.named_parameters() if p.is_meta]
                if metas:
                    print(f"[META] {layer_name}: {len(metas)} parameters are still meta:")
                    for n in metas:
                        print("   •", n)
                    raise RuntimeError("layer has meta parameters")

            for i, (layer_name, layer) in tqdm(enumerate(zip(self.layer_names, self.layers)),
                                            desc=f'running layers({self.running_device})',
                                            total=len(self.layers)):
                if self.prefetching:
                    state_dict   = future.result()
                    moved_layers = self.move_layer_to_device(state_dict)
                    assert_no_meta(layer, layer_name)
                    if (i + 1) < len(self.layer_names):
                        future = executor.submit(self.load_layer_to_cpu, self.layer_names[i+1])
                else:
                    state_dict = self.load_layer_to_cpu(layer_name)
                    moved_layers = self.move_layer_to_device(state_dict)
                    assert_no_meta(layer, layer_name)

                if layer_name == self.layer_names_dict['embed']:
                    batch_hidden_states = [input_ids[j:j+minibatch] for j in range(0, batch_size, minibatch)]
                    new_hidden_states = []

                    for j in range(0, batch_size, minibatch):
                        batch_end = min(j + minibatch, batch_size)
                        batch_input = input_ids[j:batch_end]
                        layer_outputs = layer(batch_input)
                        new_hidden_states.append(layer_outputs)
                        del batch_input
                        torch.cuda.empty_cache()

                    hidden_states = new_hidden_states
                    del batch_hidden_states
                    torch.cuda.empty_cache()

                elif layer_name == self.layer_names_dict['norm']:
                    new_hidden_states = []

                    for j in range(len(hidden_states)):
                        batch_input = hidden_states[j]
                        layer_outputs = layer(batch_input)
                        new_hidden_states.append(layer_outputs)
                        hidden_states[j] = None

                    hidden_states = new_hidden_states

                elif layer_name == self.layer_names_dict['lm_head']:
                    logits = []

                    for j in range(len(hidden_states)):
                        batch_input = hidden_states[j]
                        layer_outputs = self.run_lm_head(layer, batch_input, num_samples).to("cpu")
                        logits.append(layer_outputs)
                        hidden_states[j] = None
                    logits = torch.cat(logits, dim=0)
                else:
                    if not isinstance(hidden_states, list):
                            hidden_states = [hidden_states[j:j+minibatch] for j in range(0, batch_size, minibatch)]
                    new_hidden_states = []

                    for j in range(len(hidden_states)):
                        batch_input = hidden_states[j]
                        batch_past_key_value = past_key_values[i-1][j*minibatch:(j+1)*minibatch] if past_key_values is not None else None

                        # ------------------------------------------------------
                        # NEW: build kwargs dict and inject (cos, sin) tuple for
                        # Qwen-3 layers – other architectures stay untouched.
                        # ------------------------------------------------------
                        layer_kwargs = {
                            "position_ids":    position_ids,
                            "past_key_value":  batch_past_key_value,
                            "use_cache":       use_cache,
                            "output_attentions": output_attentions,
                        }

                        if hasattr(self.model, "model") \
                           and hasattr(self.model.model, "rotary_emb"):
                            layer_kwargs["position_embeddings"] = \
                                self._rotary(batch_input.shape[1])

                        # ------------------------------------------------------
                        # call the decoder layer with the right arguments
                        # ------------------------------------------------------
                        layer_outputs = layer(
                            batch_input,
                            **layer_kwargs,
                        )

                        new_hidden_states.append(layer_outputs[0])

                        if use_cache:
                            kv_cache_list.append(layer_outputs[1])
                        if output_attentions:
                            all_self_attns.append(layer_outputs[1] if use_cache else layer_outputs[2])

                        hidden_states[j] = None
                        #torch.cuda.empty_cache()

                    hidden_states = new_hidden_states
                    torch.cuda.empty_cache()

                if output_hidden_states:
                    all_hidden_states.append(hidden_states)

                # Remove previous layer from memory (including buffers)
                if self.hf_quantizer is not None:
                    for param_name in moved_layers:
                        set_module_tensor_to_device(self.model, param_name, 'meta')
                else:
                    layer.to("meta")

                clean_memory()

        if not return_dict:
            return tuple(v for v in [logits, kv_cache_list, all_hidden_states, all_self_attns] if v is not None)

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=tuple(kv_cache_list) if kv_cache_list is not None else None,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
            attentions=tuple(all_self_attns) if all_hidden_states is not None else None,
        )
