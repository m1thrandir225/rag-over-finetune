import re

_MULTI_QUERY_PROMPT = (
    "You are an expert at rephrasing questions for information retrieval.\n"
    "Given the following question, generate {n} alternative phrasings that "
    "a user might use to search for the same information.\n"
    "Write each alternative on a new line. Do not include numbering or bullets.\n"
    "Respond only with the alternative questions, one per line.\n\n"
    "Original question: {question}"
)

_HYDE_PROMPT = (
    "Based on the following question, write a short, factual paragraph that "
    "could appear in a document answering this question.\n"
    "Write as if you are excerpting from a relevant Wikipedia article or "
    "textbook. Use the same language as the question.\n"
    "Do not say you don't know; write a plausible answer.\n\n"
    "Question: {question}\n\n"
    "Hypothetical answer:"
)

_STEP_BACK_PROMPT = (
    "You are an expert at breaking down complex questions into simpler, "
    "more general ones.\n"
    "Given the following question, write a single more general 'step-back' "
    "question that would help gather broader context.\n"
    "The step-back question should be conceptual and abstract, not specific "
    "to the original question's details.\n\n"
    "Original question: {question}\n\n"
    "Step-back question:"
)

# Patterns that suggest a complex question where transforms can help
_COMPLEX_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(зошто|како|споредба|објасни|анализирај|разлика)\b", re.IGNORECASE),
    re.compile(r"\b(why|how|compare|explain|analyze|difference)\b", re.IGNORECASE),
]

# Very short queries where a transform is likely not needed
_MIN_WORDS_FOR_TRANSFORM = 3
