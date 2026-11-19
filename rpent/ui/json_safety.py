"""Small guards applied before decoding retained JSON artifacts."""

from __future__ import annotations


def require_bounded_json_nesting(
    raw: bytes, *, source: object, max_depth: int = 128
) -> None:
    """Reject pathologically nested JSON without depending on parser recursion."""
    depth = 0
    in_string = False
    escaped = False
    for byte in raw:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:  # backslash
                escaped = True
            elif byte == 0x22:  # quote
                in_string = False
            continue
        if byte == 0x22:
            in_string = True
        elif byte in (0x5B, 0x7B):  # [ {
            depth += 1
            if depth > max_depth:
                raise ValueError(f"JSON nesting limit exceeded in {source}")
        elif byte in (0x5D, 0x7D):  # ] }
            depth -= 1


__all__ = ["require_bounded_json_nesting"]
