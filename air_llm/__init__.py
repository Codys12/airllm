from sys import platform

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True

if is_on_mac_os:
    from .airllm_llama_mlx import AirLLMLlamaMlx
    from .auto_model import AutoModel
else:
    from .airllm.airllm import AirLLMLlama2
    from .airllm.airllm_chatglm import AirLLMChatGLM
    from .airllm.airllm_qwen import AirLLMQWen
    from .airllm.airllm_baichuan import AirLLMBaichuan
    from .airllm.airllm_internlm import AirLLMInternLM
    from .airllm.airllm_mistral import AirLLMMistral
    from .airllm.airllm_mixtral import AirLLMMixtral
    from .airllm.airllm_base import AirLLMBaseModel
    from .airllm.auto_model import AutoModel
    from .airllm.utils import split_and_save_layers
    from .airllm.utils import NotEnoughSpaceException
