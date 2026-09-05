"""Extensible schema value-handler API."""

from src.decoder_core import (
    UnsupportedTypeError,
    ValueHandler,
    ValueHandlerRegistry,
)

__all__ = ["UnsupportedTypeError", "ValueHandler", "ValueHandlerRegistry"]
