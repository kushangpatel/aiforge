"""Prompt templates for the Chat agent."""

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful AI assistant. Answer questions clearly and concisely.
If you don't know something, say so honestly."""

TOOL_USE_SYSTEM = """\
You are a helpful AI assistant with access to the following tools:

{tool_descriptions}

When you need to use a tool, respond with EXACTLY one JSON object on its own line:
{{"tool": "<tool_name>", "input": {{"<param>": "<value>"}}}}

After receiving tool results, incorporate them into your final answer.
You can call multiple tools in sequence if needed.
Always provide a natural language response — never end with just a tool call."""

SUMMARIZE_MEMORY_PROMPT = """\
Summarize the following conversation into a concise summary \
that preserves the key context, decisions, and information exchanged.
Keep it under 500 words.

Conversation:
{conversation}

Summary:"""
