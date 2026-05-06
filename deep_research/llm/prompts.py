from __future__ import annotations


def format_system_prompt(role_description: str, tools_description: str = "") -> str:
    """Build a system prompt that includes role context and available tools."""
    parts = [role_description.strip()]
    if tools_description:
        parts += [
            "",
            "## Available Tools",
            "",
            tools_description,
            "",
            "## How to Call a Tool",
            "",
            'To call a tool, output a JSON block inside triple backticks tagged `json`:',
            "",
            "```json",
            '{"tool": "tool_name", "args": {"param1": "value1"}}',
            "```",
            "",
            "You may call multiple tools in sequence. After each tool call, wait for the",
            "result before deciding whether to call another tool or produce a final answer.",
            "When you have enough information, output your final answer as plain text.",
        ]
    return "\n".join(parts)


def build_messages(
    system: str,
    history: list[dict],
    user_message: str,
) -> list[dict]:
    """Assemble the messages list for an Ollama chat call."""
    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4
