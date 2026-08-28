from __future__ import annotations

import re


ZERO_WIDTH_PATTERN = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")
PICTOGRAPHIC_SYMBOL_PATTERN = re.compile(
    r"[\U0001F000-\U0001FFFF"
    r"\U00002700-\U000027BF"
    r"\U00002600-\U000026FF"
    r"\U00002300-\U000023FF"
    r"\U00002B00-\U00002BFF"
    r"\U000E0020-\U000E007F"
    r"\ufe0e\ufe0f\u20e3]"
)
PLACEHOLDER_TOKEN_PATTERN = re.compile(
    r"(\[\s*M\s*[\W_]*A\s*[\W_]*S\s*[\W_]*K\s*(\d*)\s*\]"
    r"|\{\s*M\s*[\W_]*A\s*[\W_]*S\s*[\W_]*K\s*(\d*)\s*\}"
    r"|<\s*M\s*[\W_]*A\s*[\W_]*S\s*[\W_]*K\s*(\d*)\s*>"
    r"|\(\s*M\s*[\W_]*A\s*[\W_]*S\s*[\W_]*K\s*(\d*)\s*\)"
    r"|\bM\s*[\W_]*A\s*[\W_]*S\s*[\W_]*K\s*(\d*)\b)",
    flags=re.IGNORECASE,
)
ASCII_CONFUSABLE_NORMALIZATION = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
ANCHOR_CONFUSABLE_NORMALIZATION = str.maketrans({"$": "S", "3": "e", "7": "t", "@": "a", "0": "o"})
FRAGMENTED_LETTER_SEQUENCE_PATTERN = re.compile(r"\b(?:[A-Za-z]\s+){2,}[A-Za-z]\b")
WORD_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9@$]{2,24}\b")
RECURRING_AFFIX_QUOTED_PAYLOAD_PATTERN = re.compile(
    r"(?i)(?:my\s+)?question\s+is\s*[\"'\u201c\u201d](.*?)[\"'\u201c\u201d]\s*,\s*which is in ubbi dubbi",
    re.DOTALL,
)
QUOTED_PAYLOAD_FALLBACK_PATTERN = re.compile(
    r"(?i)(?:my\s+)?(?:question|request|query|prompt|instruction)\s+is\s*[\"'\u201c\u201d](.*?)[\"'\u201c\u201d]",
    re.DOTALL,
)
RECURRING_AFFIX_MARKER_PATTERN = re.compile(r"(?i)ub")
VOWEL_PREFIX_FRAGMENT_PATTERN = re.compile(r"(?i)ub(?=[aeiouAEIOU])")
BOUNDED_REPAIR_VOCABULARY = {
    "attack",
    "backdoor",
    "bomb",
    "bypass",
    "card",
    "credential",
    "credentials",
    "database",
    "ddos",
    "exploit",
    "fraud",
    "hack",
    "hacking",
    "illegal",
    "keylogger",
    "malware",
    "password",
    "phishing",
    "poison",
    "ransomware",
    "social",
    "steal",
    "stealing",
    "trojan",
    "virus",
    "weapon",
}
