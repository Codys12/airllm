"""
air_llm/__init__.py
-------------------

Public-API shim for the AirLLM implementation package.

 * When AirLLM is installed from PyPI, the build system already
   generates a top-level `airllm` stub that re-exports this package.

 * When you work from a *source checkout* (editable install or putting
   the repo on PYTHONPATH) that stub is absent, so `import airllm`
   resolves to *this* file instead.  We therefore re-export everything
   that end-users expect—most importantly `AutoModel`.

The code is deliberately minimal: we load the real implementation
modules with `import_module`, hoist their symbols into our namespace,
compute `__all__`, and get out of the way.
"""

from importlib import import_module as _imp

# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------

# Auto-detection wrapper (added in v2.6)
AutoModel = _imp('.auto_model', package=__name__).AutoModel        # noqa: N816

# Legacy explicit model wrappers: keep them for back-compat
AirLLMLlama2   = _imp('.llama2',   package=__name__).AirLLMLlama2   # noqa: N816
AirLLMLlama3   = _imp('.llama3',   package=__name__).AirLLMLlama3   # noqa: N816
AirLLMChatGLM  = _imp('.chatglm',  package=__name__).AirLLMChatGLM  # noqa: N816
AirLLMQwen     = _imp('.qwen',     package=__name__).AirLLMQwen     # noqa: N816
AirLLMBaichuan = _imp('.baichuan', package=__name__).AirLLMBaichuan # noqa: N816
AirLLMMistral  = _imp('.mistral',  package=__name__).AirLLMMistral  # noqa: N816
AirLLMInternLM = _imp('.internlm', package=__name__).AirLLMInternLM # noqa: N816

# Tokenizer helper (mirrors transformers.AutoTokenizer)
AutoTokenizer = _imp('.auto_tokenizer', package=__name__).AutoTokenizer

# ---------------------------------------------------------------------------
# House-keeping
# ---------------------------------------------------------------------------

def _pub(obj):
    """helper: public names in *obj* that don’t start with '_'."""
    return [n for n in obj.__dict__ if not n.startswith('_')]

__all__ = sorted(
    {'AutoModel', 'AutoTokenizer',
     'AirLLMLlama2', 'AirLLMLlama3', 'AirLLMChatGLM',
     'AirLLMQwen', 'AirLLMBaichuan', 'AirLLMMistral', 'AirLLMInternLM',
     *_pub(_imp('.auto_model', package=__name__)),
     *_pub(_imp('.auto_tokenizer', package=__name__)),
    }
)

# Surface version string if build machinery generated it
try:
    from .version import __version__   # created at sdist/wheel time
except ModuleNotFoundError:
    __version__ = 'dev'

del _imp, _pub      # keep module namespace clean
