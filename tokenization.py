from functools import lru_cache

import tiktoken


KNOWN_ENCODINGS = {
    "cl100k_base",
    "o200k_base",
    "p50k_base",
    "r50k_base",
    "p50k_edit",
    "gpt2",
}

MODEL_ALIASES = {
    "gpt4o": "gpt-4o",
    "gpt4omini": "gpt-4o-mini",
    "gpt-4omini": "gpt-4o-mini",
    "gpt4o-mini": "gpt-4o-mini",
    "claude-3.5-sonnet": "claude-3-5-sonnet",
}

EXPLICIT_MODEL_ENCODINGS = {
    "claude-3-5-sonnet": "cl100k_base",
    "deepseek-chat": "cl100k_base",
    "gemini-1.5-pro": "cl100k_base",
    "gemini-2.0-pro": "cl100k_base",
    "o1": "o200k_base",
    "o1-mini": "o200k_base",
    "o1-preview": "o200k_base",
    "o3-mini": "o200k_base",
}


def normalize_model_name(model_name):
    return MODEL_ALIASES.get(model_name, model_name)


def encoding_name_for_model(model_name):
    normalized_model = normalize_model_name(model_name)

    if normalized_model in KNOWN_ENCODINGS:
        return normalized_model

    explicit_encoding = EXPLICIT_MODEL_ENCODINGS.get(normalized_model)
    if explicit_encoding is not None:
        return explicit_encoding

    if normalized_model.startswith(("gpt-4o", "o1", "o3", "o4")):
        return "o200k_base"
    if normalized_model.startswith(("gpt-4", "gpt-3.5", "claude", "deepseek", "gemini")):
        return "cl100k_base"

    return normalized_model


@lru_cache(maxsize=None)
def get_encoding(model_name):
    encoding_name = encoding_name_for_model(model_name)

    if encoding_name in KNOWN_ENCODINGS:
        return tiktoken.get_encoding(encoding_name)

    return tiktoken.encoding_for_model(encoding_name)


def count_text_tokens(text, model_name):
    return len(get_encoding(model_name).encode(text))


def count_message_tokens(messages, model_name):
    encoding = get_encoding(model_name)
    return sum(len(encoding.encode(message["content"])) for message in messages)
