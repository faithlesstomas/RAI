"""Compatibility facade for callers migrating to the AgentBackend port."""

from rai.backends.antigravity import AntigravityBackend


class ChatService(AntigravityBackend):
    """Deprecated name preserving the pre-Stage-1 chat API."""
