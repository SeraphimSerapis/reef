"""Reef: tiny typed-decision models that run on a Coral Edge TPU.

    from reef import Reef

    reef = Reef("artifacts/reef-tiny")
    reef.decide(state, questions)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"
__all__ = ["Reef", "__version__"]

if TYPE_CHECKING:
    from reef.runtime import Reef


def __getattr__(name: str) -> Any:
    # Lazy, so `import reef` stays cheap and never touches the TFLite runtime.
    if name == "Reef":
        from reef.runtime import Reef

        return Reef
    raise AttributeError(name)
