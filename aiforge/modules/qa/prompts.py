"""Prompt templates for the DocQA RAG agent."""

CLASSIFY_PROMPT = """\
Classify the following question into one of these categories:
- factual: Asking for specific facts from the codebase
- conceptual: Asking how something works at a high level
- debugging: Asking about a bug or error
- comparison: Comparing two approaches or implementations
- exploratory: Open-ended exploration of the codebase

Question: {query}

Return a single word: the category."""

REASON_PROMPT = """\
Given the question and retrieved context, determine if you have \
enough information to answer confidently.

Question: {query}
Intent: {intent}
Iteration: {iteration}/{max_iterations}

Retrieved Context:
{context}

Evaluate:
1. Do the retrieved chunks directly address the question?
2. Is there enough detail to give a complete answer?
3. Would a refined search likely find better information?

Respond with exactly one of:
- SUFFICIENT: [confidence score 0.0-1.0]
- INSUFFICIENT: [suggested refined query]"""

RESPOND_PROMPT = """\
Answer the following question using ONLY the provided context.
If the context doesn't contain enough information, say so clearly.
Include specific source citations in your answer.

Question: {query}

Context:
{context}

Rules:
- Cite sources as [filename:line_range] inline
- Be specific and detailed
- If information is incomplete, acknowledge what's missing
- Do not fabricate information not present in the context

Answer:"""
