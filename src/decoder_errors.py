"""Errors raised by constrained decoding."""


class DecoderError(RuntimeError):
    """Base error for an invalid constrained-generation transition."""

    def __init__(self, msg: str = "Decoder Error") -> None:
        self.msg = msg

    def __str__(self) -> str:
        return self.msg


class NoValidTokenError(DecoderError):
    """Raised when no vocabulary token can continue the current state."""

    def __init__(self, msg: str = "No Valid Token Error") -> None:
        self.msg = msg

    def __str__(self) -> str:
        return self.msg


class UnsupportedTypeError(DecoderError):
    """Raised when a schema type has no registered value handler."""

    def __init__(self, msg: str = "Unsupported Type Error") -> None:
        self.msg = msg

    def __str__(self) -> str:
        return self.msg
