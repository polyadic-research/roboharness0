"""Model-facing context projection contracts.

Context policies decide what an LLM sees after a tool call.  They do not
execute tools, mutate the environment, or decide whether an action is safe.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ProjectedImage:
    """One image selected for model-facing multimodal context."""

    data: bytes
    media_type: str = "image/png"
    role: str = "observation"
    camera_id: str | None = None

    def as_anthropic_block(self) -> dict[str, Any]:
        """Return the Anthropic-style content-block shape used by RPent."""
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": self.media_type,
                "data": base64.b64encode(self.data).decode("utf-8"),
            },
        }


@dataclass(frozen=True)
class ContextProjection:
    """Text and images produced from one raw tool result."""

    text: str
    images: tuple[ProjectedImage, ...] = ()

    def content_blocks(self) -> list[dict[str, Any]]:
        """Build the content-block shape consumed by all current runtimes."""
        blocks = [{"type": "text", "text": self.text}]
        for index, image in enumerate(self.images, start=1):
            if image.camera_id is not None:
                # Numbering survives runtimes that separate text and images.
                blocks.append({
                    "type": "text",
                    "text": f"Image {index} camera: {image.camera_id}",
                })
            blocks.append(image.as_anthropic_block())
        return blocks


class ContextPolicy(Protocol):
    """Project raw tool output into bounded model-facing context."""

    id: str

    def project_tool_result(
        self,
        *,
        tool_name: str,
        result: dict[str, Any],
        max_text_chars: int,
    ) -> ContextProjection:
        """Return the context sent back to the model after a tool call."""
        ...
