"""dots.mocr package exports.

Keep heavy parser dependencies lazy so lightweight tools such as the MLX CLI can
import ``dots_mocr`` without requiring optional vLLM/SVG runtime libraries.
"""

__all__ = ["DotsMOCRParser"]


def __getattr__(name):
    if name == "DotsMOCRParser":
        from .parser import DotsMOCRParser

        return DotsMOCRParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
