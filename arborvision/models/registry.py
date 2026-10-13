"""Model registry — maps string names to adapter factories.

New adapters should be registered here (or via :func:`register` at import
time) so they show up in the API, CLI, and Streamlit UI without further
plumbing.
"""

from __future__ import annotations

from typing import Callable

from .base import ModelAdapter

# Factory: zero-arg callable that returns a fresh ModelAdapter instance.
# Factories are used (not singletons) so each benchmark run gets a clean
# adapter — important for fair timing comparisons.
_FACTORIES: dict[str, Callable[[], ModelAdapter]] = {}


def register(name: str, factory: Callable[[], ModelAdapter]) -> None:
    """Register a factory under ``name``."""
    if name in _FACTORIES:
        raise ValueError(f"Model {name!r} is already registered.")
    _FACTORIES[name] = factory


def unregister(name: str) -> None:
    """Remove a previously-registered factory (mainly for tests)."""
    _FACTORIES.pop(name, None)


def list_models() -> list[str]:
    """Return the sorted list of available model names."""
    return sorted(_FACTORIES.keys())


def get_model(name: str) -> ModelAdapter:
    """Instantiate a fresh adapter by name.

    Raises
    ------
    KeyError
        If ``name`` is not registered. The error message lists valid names
        to make typos obvious.
    """
    if name not in _FACTORIES:
        raise KeyError(
            f"Unknown model {name!r}. Available: {list_models()}"
        )
    return _FACTORIES[name]()


# --- Built-in registrations ---------------------------------------------------
# Importing the adapter modules here (rather than at the top of the file)
# avoids a circular import and keeps torch/torchvision loading deferred
# until the user actually asks for a model.

def _register_builtins() -> None:
    """Register the models that ship with ArborVision."""
    if "maskrcnn" in _FACTORIES:
        return  # idempotent
    from .maskrcnn import MaskRCNNAdapter
    from .classifier import ResNet50Adapter

    register("maskrcnn", MaskRCNNAdapter)
    register("resnet50", ResNet50Adapter)


_register_builtins()