"""Built-in homelab packs."""

from __future__ import annotations

from reef.packs.base import Pack, QuestionSpec

__all__ = ["Pack", "QuestionSpec", "get", "names"]


def _registry() -> dict[str, Pack]:
    from reef.packs import alerts, frigate, notify, protect, unifi

    return {p.name: p for p in (alerts.PACK, frigate.PACK, notify.PACK, protect.PACK, unifi.PACK)}


def names() -> list[str]:
    return sorted(_registry())


def get(name: str) -> Pack:
    reg = _registry()
    if name not in reg:
        raise KeyError(f"unknown pack {name!r}; available: {', '.join(sorted(reg))}")
    return reg[name]
