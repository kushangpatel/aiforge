"""Reusable sample texts for all module tests."""

SHORT_ARTICLE = """
Artificial intelligence is transforming how businesses operate. Companies are
now using AI to automate repetitive tasks, improve customer service, and gain
insights from large datasets. The technology continues to advance rapidly.
"""

LONG_ARTICLE = """
Artificial intelligence (AI) has become one of the most transformative
technologies of the 21st century. From healthcare to finance, education to
manufacturing, AI systems are reshaping industries and changing how we work.

Machine learning, a subset of AI, enables computers to learn from data
without being explicitly programmed. Deep learning, in turn, uses neural
networks with many layers to tackle complex problems like image recognition,
natural language processing, and game playing.

Large language models (LLMs) like GPT-4 and Claude represent the current
pinnacle of NLP research. These models are trained on vast corpora of text
and can generate human-quality prose, answer questions, summarise documents,
write code, and engage in multi-turn conversations.

Despite their impressive capabilities, LLMs face challenges including
hallucinations (generating plausible but incorrect information), bias in
training data, high computational costs, and questions about data privacy
and intellectual property.

The responsible deployment of AI requires careful attention to ethics,
fairness, transparency, and accountability. Organisations must balance the
considerable benefits of AI adoption against potential harms and risks.
"""

INVOICE_TEXT = """
Invoice #4521
From: Acme Corporation
Date: January 15, 2026

Item 1: Widget A x10 @ $5.00 each = $50.00
Item 2: Widget B x3 @ $12.00 each = $36.00

Subtotal: $86.00
Tax (0%): $0.00
Total Due: $86.00 USD

Payment terms: Net 30
"""

BUG_REPORT = "The login button is broken on mobile chrome. Clicking it does nothing."
FEATURE_REQUEST = "It would be great to have dark mode support in the dashboard."
QUESTION_TEXT = "How do I reset my password if I forgot my email address?"

TOXIC_TEXT = "You are completely worthless and should be fired immediately."
PII_TEXT = "Please contact John Smith at john.smith@example.com or call 555-867-5309."
SAFE_TEXT = "The weather today is sunny with a high of 72 degrees Fahrenheit."

CODE_PYTHON = """
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
"""

MULTI_TURN_CONVERSATION = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "What is its population?"},
]

PRODUCTS = [
    {"id": "1", "text": "Wireless noise-cancelling headphones", "metadata": {"price": 299}},
    {"id": "2", "text": "Portable Bluetooth speaker", "metadata": {"price": 79}},
    {"id": "3", "text": "Running shoes for marathon training", "metadata": {"price": 129}},
    {"id": "4", "text": "Ergonomic office chair", "metadata": {"price": 499}},
    {"id": "5", "text": "Coffee maker with built-in grinder", "metadata": {"price": 189}},
]
