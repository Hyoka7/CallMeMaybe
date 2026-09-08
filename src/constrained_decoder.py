"""Public compatibility facade for the constrained decoder.

The implementation lives in :mod:`src.decoder_core`; these re-exports keep
the original import path stable for the CLI and downstream users.
"""

from src.decoder_core import (
    END,
    NUMBER_COMPLETE,
    NUMBER_PREFIX,
    ConstrainedDecoder,
    DecoderError,
    FunctionNameState,
    LiteralResult,
    LiteralState,
    NoValidTokenError,
    ParameterKeyState,
    ParameterSeparatorState,
    ParameterValueState,
    TrieNode,
    UnsupportedTypeError,
    ValueHandler,
    ValueHandlerRegistry,
    Vocabulary,
)

__all__ = [
    "END", "NUMBER_COMPLETE", "NUMBER_PREFIX", "ConstrainedDecoder",
    "DecoderError", "FunctionNameState", "LiteralResult", "LiteralState",
    "NoValidTokenError", "ParameterKeyState", "ParameterSeparatorState",
    "ParameterValueState", "TrieNode", "UnsupportedTypeError",
    "ValueHandler", "ValueHandlerRegistry", "Vocabulary",
]
