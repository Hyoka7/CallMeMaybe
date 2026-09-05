"""Errors raised by constrained decoding."""


class DecoderError(RuntimeError):
    """Base error for an invalid constrained-generation transition."""


class NoValidTokenError(DecoderError):
    """Raised when no vocabulary token can continue the current state."""


class UnsupportedTypeError(DecoderError):
    """Raised when a schema type has no registered value handler."""
