from importlib import import_module as _imp

_module = _imp("air_llm")          # pull in the real library
globals().update(_module.__dict__) # re-export everything

# Optional niceties
__all__  = getattr(_module, "__all__", [k for k in globals() if not k.startswith("_")])
__doc__  = _module.__doc__
__version__ = getattr(_module, "__version__", "dev")
del _imp, _module
