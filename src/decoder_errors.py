"""Errors raised by constrained decoding."""


class DecoderError(RuntimeError):
    """Base error for an invalid constrained-generation transition."""

    def __init__(self, msg: str = "Decoder Error") -> None:
        """Initialize the decoding failure with a readable message."""
        self.msg = msg

    def __str__(self) -> str:
        """Return the error message for command-line reporting."""
        return self.msg


class NoValidTokenError(DecoderError):
    """Raised when no vocabulary token can continue the current state."""

    def __init__(self, msg: str = "No Valid Token Error") -> None:
        """Initialize a failure caused by an empty token candidate set."""
        self.msg = msg

    def __str__(self) -> str:
        """Return the decoding error message."""
        return self.msg


class UnsupportedTypeError(DecoderError):
    """Raised when a schema type has no registered value handler."""

    def __init__(self, msg: str = "Unsupported Type Error") -> None:
        """Initialize a failure caused by an unknown schema type."""
        self.msg = msg

    def __str__(self) -> str:
        """Return the decoding error message."""
        return self.msg
