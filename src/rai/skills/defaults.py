"""Built-in default skill recipes for Rich AI."""

from __future__ import annotations

from .models import SkillDefinition, SkillRegistry


def create_default_skill_registry() -> SkillRegistry:
    """Create a registry pre-loaded with Rich AI's core desktop skills."""
    registry = SkillRegistry()

    registry.register(
        SkillDefinition(
            name="system-inspection",
            description=(
                "Discover and count installed desktop applications "
                "or inspect running processes."
            ),
            capabilities=("application.list", "process.inspect"),
            when_to_use=(
                "When the user asks to count, list, or inspect applications, "
                "software, or processes on GNU/Linux."
            ),
            instructions=(
                "1. If asked about applications or installed software, emit tool call "
                "`<tool_call>call: application.list{}</tool_call>`.\n"
                "2. When observing the applications list, calculate the exact count "
                "or filter entries based on user query.\n"
                "3. Synthesize the final answer reporting verified results honestly."
            ),
        )
    )

    registry.register(
        SkillDefinition(
            name="web-research",
            description=(
                "Perform bounded search and retrieval of public information from the web."
            ),
            capabilities=("browser.search", "browser.read_page", "browser.open_result"),
            when_to_use=(
                "When answering questions about current events, online documentation, "
                "or public websites."
            ),
            instructions=(
                "1. Deliberate search keywords and emit "
                '`<tool_call>call: browser.search{"query": ...}</tool_call>`.\n'
                "2. If deeper information from a specific search result is required, "
                "read the page with `browser.read_page`.\n"
                "3. Ground the final synthesis strictly in observed page content."
            ),
        )
    )

    registry.register(
        SkillDefinition(
            name="file-inspection",
            description="Search, list, and open files within configured allowed directories.",
            capabilities=("file.list", "file.search", "document.open"),
            when_to_use=(
                "When the user asks to find, locate, list or open documents and local files."
            ),
            instructions=(
                "1. Search allowed directories with `file.search` or list contents "
                "with `file.list`.\n"
                "2. If requested to open a document, use `document.open` "
                "with verified target path."
            ),
        )
    )

    return registry
