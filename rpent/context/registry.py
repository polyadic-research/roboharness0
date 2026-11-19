"""Compatibility imports for the former context-policy registry module."""

from rpent.context.projectors import (
    ContextPolicyFactory,
    list_context_policy_ids,
    resolve_context_policy,
)

__all__ = [
    "ContextPolicyFactory",
    "list_context_policy_ids",
    "resolve_context_policy",
]
