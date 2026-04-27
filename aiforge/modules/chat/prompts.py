"""Prompt templates for the Chat agent."""

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful AI assistant. Answer questions clearly and concisely.
If you don't know something, say so honestly."""

TOOL_USE_SYSTEM = """\
You have access to the following tools:

{tool_descriptions}

To use a tool, respond with ONLY a JSON object (no other text):
{{"tool": "<tool_name>", "input": {{"<param>": "<value>"}}}}

IMPORTANT:
- When a task requires a tool, you MUST output the JSON tool call.
- Do NOT describe what you would do — actually call the tool.
- After receiving tool results, provide a natural language response incorporating those results.
- You may call tools multiple times in sequence if needed."""

SUMMARIZE_MEMORY_PROMPT = """\
Summarize the following conversation into a concise summary \
that preserves the key context, decisions, and information exchanged.
Keep it under 500 words.

Conversation:
{conversation}

Summary:"""
