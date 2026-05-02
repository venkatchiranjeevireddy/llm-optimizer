import re
from typing import Dict, List, Set

FILLER_PHRASES: List[str] = [
    "i hope you are doing well",
    "if you don't mind",
    "thank you so much",
    "i wanted to ask you",
    "basically",
    "so basically",
    "i was wondering",
    "could you please",
    "would you be able to",
    "i would like to know",
]

_WORD_RE = re.compile(r"[a-zA-Z0-9']+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_REDUNDANCY_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "for",
    "to",
    "of",
    "in",
    "on",
    "this",
    "that",
    "you",
    "your",
    "can",
    "could",
    "would",
    "please",
    "me",
}


def _token_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _unique_words(text: str) -> Set[str]:
    return {word.lower() for word in _WORD_RE.findall(text)}


def _remove_filler_phrases(text: str) -> str:
    updated = text
    for phrase in sorted(FILLER_PHRASES, key=len, reverse=True):
        pattern = re.compile(rf"\b{re.escape(phrase)}\b[,\s]*", re.IGNORECASE)
        updated = pattern.sub("", updated)
    return updated


def _sentence_signature(sentence: str) -> Set[str]:
    tokens = {w.lower() for w in _WORD_RE.findall(sentence)}
    # Ignore tiny function words to focus on meaning overlap.
    return {t for t in tokens if len(t) > 2 and t not in _REDUNDANCY_STOPWORDS}


def _jaccard_similarity(left: Set[str], right: Set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _remove_redundant_sentences(text: str) -> str:
    raw_sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    if not raw_sentences:
        return text

    kept_sentences: List[str] = []
    kept_signatures: List[Set[str]] = []

    for sentence in raw_sentences:
        signature = _sentence_signature(sentence)
        duplicate_index = None
        for idx, kept_signature in enumerate(kept_signatures):
            if _jaccard_similarity(signature, kept_signature) >= 0.85:
                duplicate_index = idx
                break

        if duplicate_index is None:
            kept_sentences.append(sentence)
            kept_signatures.append(signature)
            continue

        # Keep the clearest duplicate: fewer words usually means less fluff.
        if _token_count(sentence) < _token_count(kept_sentences[duplicate_index]):
            kept_sentences[duplicate_index] = sentence
            kept_signatures[duplicate_index] = signature

    return " ".join(kept_sentences)


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"^[\.,;:!?]+\s*", "", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text.strip()


def compress_prompt(text: str) -> Dict[str, object]:
    original_text = text if isinstance(text, str) else str(text)
    original_tokens = _token_count(original_text)
    original_unique_words = _unique_words(original_text)

    no_filler_baseline = _remove_filler_phrases(original_text)
    baseline_unique_words = _unique_words(no_filler_baseline) or original_unique_words

    compressed_text = no_filler_baseline
    compressed_text = _remove_redundant_sentences(compressed_text)
    compressed_text = _normalize_whitespace(compressed_text)

    if not compressed_text:
        compressed_text = _normalize_whitespace(original_text)

    compressed_unique_words = _unique_words(compressed_text)
    retained_ratio = (
        len(compressed_unique_words & baseline_unique_words) / len(baseline_unique_words)
        if baseline_unique_words
        else 1.0
    )

    # If more than 30% unique words are lost, keep original text to preserve meaning.
    if retained_ratio < 0.70:
        compressed_text = _normalize_whitespace(original_text)

    compressed_tokens = _token_count(compressed_text)
    tokens_saved = max(original_tokens - compressed_tokens, 0)
    compression_ratio = (compressed_tokens / original_tokens) if original_tokens else 1.0
    compression_applied = compressed_text != _normalize_whitespace(original_text)

    orig_chars = len(original_text)
    comp_chars = len(compressed_text)

    return {
        "original_text": original_text,
        "compressed_text": compressed_text,
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "tokens_saved": tokens_saved,
        "compression_ratio": round(compression_ratio, 2),
        "compression_applied": compression_applied,
        # Unicode code units — meaningful for CJK where Latin-token count can be 0.
        "original_chars": orig_chars,
        "compressed_chars": comp_chars,
        "chars_saved": max(orig_chars - comp_chars, 0),
    }
