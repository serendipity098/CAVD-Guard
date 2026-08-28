from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .constants import (
    ANCHOR_TRANSLATION,
    ASCII_CONFUSABLE_TRANSLATION,
    EMOJI_PATTERN,
    MASK_TOKEN_PATTERN,
    RISKY_CANONICAL_TERMS,
    SINGLE_LETTER_RUN_PATTERN,
    UBBI_QUESTION_FALLBACK_RE,
    UBBI_QUESTION_RE,
    UBBI_VOWEL_PREFIX_RE,
    WORD_PATTERN,
    ZERO_WIDTH_PATTERN,
)


def normalize_prompt(prompt: str) -> str:
    return ZERO_WIDTH_PATTERN.sub("", unicodedata.normalize("NFKC", prompt))


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_mask_tokens(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        digits = "".join(group for group in match.groups()[1:] if group)
        return f"[MASK{digits}]" if digits else "[MASK]"

    return MASK_TOKEN_PATTERN.sub(repl, text)


def separator_chars(text: str, preferred: Sequence[str]) -> List[str]:
    counts: Dict[str, int] = {}
    for ch in text:
        if not ch.isalnum() and not ch.isspace():
            counts[ch] = counts.get(ch, 0) + 1
    ranked = [ch for ch in preferred if counts.get(ch, 0) >= 2]
    ranked.extend(ch for ch, count in sorted(counts.items(), key=lambda item: item[1], reverse=True) if ch not in ranked and ch not in {"`", "'", '"', "-", ".", "_", "(", ")", "/", "\\"} and count >= 2)
    return ranked


def extract_ascii_art_block(prompt: str) -> str:
    text = normalize_prompt(prompt)
    match = re.search(r"ASCII art:\s*```\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if not match:
        raise ValueError("missing ASCII-art block")
    return match.group(1)


def extract_word_lengths(prompt: str) -> List[int]:
    text = re.sub(r"(?i)l3ngth", "length", normalize_prompt(prompt))
    text = re.sub(r"(?i)r3spectively", "respectively", text)
    patterns = [
        r"each word has a length of\s+([0-9,\sand]+),\s*respectively",
        r"each word has a length of\s+([0-9,\sand]+)\.",
        r"word\s+lengths?\s*[:=]\s*([0-9,\sand]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return [int(x) for x in re.findall(r"\d+", match.group(1))]
    raise ValueError("missing word lengths")


def split_h_sep_rows(ascii_block: str) -> List[List[str]]:
    rows = [line for line in ascii_block.splitlines() if line.strip()]
    best_rows: List[List[str]] = []
    best_score = (0, 0)
    for sep in separator_chars(ascii_block, ["*", "#", "@", "+", "=", "~", "^", "|", ";", ":", "!", "%"]):
        split_rows = [row.split(sep) for row in rows if sep in row]
        if not split_rows:
            continue
        counts: Dict[int, int] = {}
        for row in split_rows:
            counts[len(row)] = counts.get(len(row), 0) + 1
        width, num_rows = max(counts.items(), key=lambda item: (item[1], item[0]))
        aligned = [row for row in split_rows if len(row) == width]
        if width >= 2 and (num_rows, width) > best_score:
            best_rows = aligned
            best_score = (num_rows, width)
    if not best_rows:
        raise ValueError("missing separated rows")
    return best_rows


def recover_sequence_from_h_sep(ascii_block: str) -> str:
    rows = split_h_sep_rows(ascii_block)
    out: List[str] = []
    for col in range(len(rows[0])):
        letters = re.findall(r"[A-Za-z]", "\n".join(row[col] for row in rows))
        lower = [ch for ch in letters if ch.islower()]
        source = lower or letters
        if not source:
            out.append(" ")
            continue
        counts: Dict[str, int] = {}
        for ch in source:
            key = ch.lower()
            counts[key] = counts.get(key, 0) + 1
        out.append(max(counts.items(), key=lambda item: item[1])[0])
    return "".join(out)


def split_sequence_by_lengths(sequence: str, lengths: Sequence[int]) -> List[str]:
    words: List[str] = []
    start = 0
    for length in lengths:
        end = start + int(length)
        words.append(sequence[start:end].strip())
        start = end
    return words


def fill_masked_instruction(masked_instruction: str, words: Sequence[str]) -> str:
    restored = normalize_prompt(masked_instruction)
    for idx, word in enumerate(words, start=1):
        restored = re.sub(
            rf"(\[\s*M\s*[\W_]*A\s*[\W_]*S\s*[\W_]*K\s*{idx}\s*\]"
            rf"|\{{\s*M\s*[\W_]*A\s*[\W_]*S\s*[\W_]*K\s*{idx}\s*\}}"
            rf"|<\s*M\s*[\W_]*A\s*[\W_]*S\s*[\W_]*K\s*{idx}\s*>"
            rf"|\(\s*M\s*[\W_]*A\s*[\W_]*S\s*[\W_]*K\s*{idx}\s*\)"
            rf"|\bM\s*[\W_]*A\s*[\W_]*S\s*[\W_]*K\s*{idx}\b)",
            word,
            restored,
            flags=re.IGNORECASE,
        )
    return normalize_space(restored)


def restore_artperception(prompt: str, masked_instruction: str) -> str:
    block = extract_ascii_art_block(prompt)
    lengths = extract_word_lengths(prompt)
    words = split_sequence_by_lengths(recover_sequence_from_h_sep(block), lengths)
    return fill_masked_instruction(masked_instruction, words)


def extract_artprompt_rows(prompt: str) -> List[str]:
    text = normalize_prompt(prompt)
    anchor = text.translate(ANCHOR_TRANSLATION)
    match = re.search(r"5\.\s*Concatenate.*?\n\s*\n(?P<art>.*?)\n\s*Remember", anchor, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return [line.rstrip("\n") for line in match.group("art").splitlines() if line.strip()]
    fenced = re.search(r"```(.*?)```", text, flags=re.DOTALL)
    if fenced:
        rows = [line.rstrip("\n") for line in fenced.group(1).splitlines() if line.strip()]
        if rows:
            return rows
    raise ValueError("missing ArtPrompt rows")


def extract_artprompt_length(prompt: str) -> Optional[int]:
    match = re.search(r"ascii art of\s+(\d+)\s+letters", prompt, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def split_artprompt_rows(rows: Sequence[str]) -> List[List[str]]:
    text = "\n".join(rows)
    best_rows: List[List[str]] = []
    best_score = (0, 0)
    for sep in separator_chars(text, ["|", "#", "*", "@", "+", "=", "~", "^", ";", ":", "!", "%"]):
        split_rows = [row.split(sep) for row in rows if sep in row]
        counts: Dict[int, int] = {}
        for row in split_rows:
            counts[len(row)] = counts.get(len(row), 0) + 1
        if not counts:
            continue
        width, num_rows = max(counts.items(), key=lambda item: (item[1], item[0]))
        aligned = [row for row in split_rows if len(row) == width]
        if width >= 2 and (num_rows, width) > best_score:
            best_rows = aligned
            best_score = (num_rows, width)
    if not best_rows:
        raise ValueError("missing separated ArtPrompt rows")
    return best_rows


def recover_artprompt_word(prompt: str) -> str:
    aligned = split_artprompt_rows(extract_artprompt_rows(prompt))
    chars: List[str] = []
    for col in range(len(aligned[0])):
        letters = re.findall(r"[A-Za-z]", "\n".join(row[col] for row in aligned))
        if not letters:
            continue
        counts: Dict[str, int] = {}
        for ch in letters:
            key = ch.lower()
            counts[key] = counts.get(key, 0) + 1
        chars.append(max(counts.items(), key=lambda item: item[1])[0])
    word = "".join(chars)
    expected = extract_artprompt_length(prompt)
    return word[:expected] if expected is not None else word


def extract_artprompt_masked_instruction(prompt: str) -> str:
    anchor = normalize_prompt(prompt).translate(ANCHOR_TRANSLATION)
    patterns = [
        r"After that,\s*(.*?)\s*\(replace\s+\[MASK\]\s+with this word\)\?",
        r"After that,\s*(.*?)\s*\(replace\s+.*?mask.*?with this word\)\?",
        r"After that,\s*(.*?)\s*\(replace\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, anchor, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return normalize_space(match.group(1))
    raise ValueError("missing ArtPrompt masked instruction")


def fill_single_mask(masked_instruction: str, word: str) -> str:
    mask = re.compile(
        r"(\[\s*M\s*[\W_]*A\s*[\W_]*S\s*[\W_]*K\s*\]"
        r"|\{\s*M\s*[\W_]*A\s*[\W_]*S\s*[\W_]*K\s*\}"
        r"|<\s*M\s*[\W_]*A\s*[\W_]*S\s*[\W_]*K\s*>"
        r"|\(\s*M\s*[\W_]*A\s*[\W_]*S\s*[\W_]*K\s*\)"
        r"|\bM\s*[\W_]*A\s*[\W_]*S\s*[\W_]*K\b)",
        flags=re.IGNORECASE,
    )
    return normalize_space(mask.sub(word, normalize_prompt(masked_instruction), count=1))


def restore_artprompt(prompt: str) -> str:
    return fill_single_mask(extract_artprompt_masked_instruction(prompt), recover_artprompt_word(prompt))


def strip_inline_noise(text: str) -> str:
    clean = unicodedata.normalize("NFKC", text)
    clean = ZERO_WIDTH_PATTERN.sub("", clean)
    clean = EMOJI_PATTERN.sub("", clean)
    clean = re.sub(r"(?<=\w)[_\-`~^|#*@+=:;!?%]+(?=\w)", "", clean)
    return normalize_space(clean.translate(ASCII_CONFUSABLE_TRANSLATION))


def remove_emoji(text: str) -> str:
    clean = ZERO_WIDTH_PATTERN.sub("", unicodedata.normalize("NFKC", text))
    return normalize_space(EMOJI_PATTERN.sub("", clean))


def collapse_spelled_letters(text: str) -> str:
    out = re.sub(r"(?i)\b([a-z])[\s._\-|/]+([a-z])[\s._\-|/]+([a-z])(?:[\s._\-|/]+([a-z])){1,12}\b", lambda m: re.sub(r"[\s._\-|/]+", "", m.group(0)), text)
    return SINGLE_LETTER_RUN_PATTERN.sub(lambda m: m.group(0).replace(" ", ""), out)


def bounded_levenshtein(a: str, b: str, limit: int = 1) -> int:
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        row_min = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
            row_min = min(row_min, current[-1])
        if row_min > limit:
            return limit + 1
        previous = current
    return previous[-1]


def is_adjacent_swap(a: str, b: str) -> bool:
    if len(a) != len(b):
        return False
    diffs = [idx for idx, pair in enumerate(zip(a, b)) if pair[0] != pair[1]]
    return len(diffs) == 2 and diffs[1] == diffs[0] + 1 and a[diffs[0]] == b[diffs[1]] and a[diffs[1]] == b[diffs[0]]


def repair_security_keyword(token: str) -> str:
    lower = token.lower().translate(ASCII_CONFUSABLE_TRANSLATION)
    if lower in RISKY_CANONICAL_TERMS:
        return lower
    for term in RISKY_CANONICAL_TERMS:
        if abs(len(lower) - len(term)) <= 1 and (is_adjacent_swap(lower, term) or bounded_levenshtein(lower, term, 1) <= 1):
            return term
    return token


def repair_character_edit_noise(text: str) -> str:
    return WORD_PATTERN.sub(lambda m: repair_security_keyword(m.group(0)), strip_inline_noise(collapse_spelled_letters(text)))


def remove_anomalous_repetition(text: str) -> str:
    clean = ZERO_WIDTH_PATTERN.sub("", unicodedata.normalize("NFKC", text))
    clean = EMOJI_PATTERN.sub("", clean)
    clean = re.sub(r"([^\w\s])\1{2,}", r"\1\1", clean)
    clean = re.sub(r"([A-Za-z])\1{4,}", r"\1\1", clean)
    clean = re.sub(r"[ \t]{3,}", "  ", clean)
    clean = re.sub(r"\n{4,}", "\n\n\n", clean)
    return clean.strip()


def extract_quoted_payloads(text: str) -> List[Tuple[str, str]]:
    patterns = [
        r"(?:my\s+)?(?:question|request|query|prompt|instruction)\s+is\s*[\"'\u201c\u201d](.*?)[\"'\u201c\u201d]",
        r"(?:question|request|query|prompt|instruction)\s*:\s*[\"'\u201c\u201d](.*?)[\"'\u201c\u201d]",
        r"<(?:untrusted_)?(?:user_)?(?:input|query|request|prompt)>\s*(.*?)\s*</(?:untrusted_)?(?:user_)?(?:input|query|request|prompt)>",
        r"(?:my\s+)?(?:question|request|query|prompt|instruction)\s+is\s*(.+?)(?:,\s*which is|\n|$)",
        r"(?:question|request|query|prompt|instruction)\s*:\s*(.+?)(?:,\s*which is|\n|$)",
    ]
    payloads: List[Tuple[str, str]] = []
    seen = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            payload = normalize_space(match.group(1))
            if len(payload) >= 8 and payload not in seen:
                payloads.append(("payload_island", payload))
                seen.add(payload)
    return payloads


def extract_ubbi_core(text: str, explicit: Optional[str] = None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    for pattern in (UBBI_QUESTION_RE, UBBI_QUESTION_FALLBACK_RE):
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    for _, payload in extract_quoted_payloads(text):
        if len(re.findall(r"(?i)ub", payload)) >= 2:
            return payload.strip()
    return ""


def decode_ubbi(text: str) -> str:
    return normalize_space(UBBI_VOWEL_PREFIX_RE.sub("", unicodedata.normalize("NFKC", text)))


def reconstruct_ubbi_wrapper(text: str, explicit: Optional[str] = None) -> str:
    core = extract_ubbi_core(text, explicit)
    if not core:
        return ""
    decoded = decode_ubbi(core)
    if not decoded or decoded == core:
        return ""
    return text.replace(core, decoded, 1) if core in text else decoded


def looks_like_artperception(text: str, masked_instruction: Optional[str]) -> bool:
    if masked_instruction and re.search(r"\[\s*MASK\s*\d+\s*\]", masked_instruction, flags=re.IGNORECASE):
        return True
    return bool(re.search(r"ASCII\s+art|text\s+art|letter\s+grid", text, flags=re.IGNORECASE) and re.search(r"each word has a length|word\s+length|number of letters|\[\s*MASK", text, flags=re.IGNORECASE))


def looks_like_artprompt(text: str) -> bool:
    return bool(re.search(r"ascii\s+art|text\s+art|letter\s+grid", text, flags=re.IGNORECASE) and re.search(r"\d+\s+letters|replace\s+.*?mask|mask.*?with\s+this\s+word|\[\s*MASK", text, flags=re.IGNORECASE | re.DOTALL))


def candidate_differs(text: str, refs: Iterable[str]) -> bool:
    key = normalize_space(text)
    return bool(key) and all(key != normalize_space(ref) for ref in refs)


def append_changed(out: List[Tuple[str, str]], name: str, text: str, refs: List[str], used: set[str]) -> None:
    clean = text.strip()
    if clean and name not in used and candidate_differs(clean, refs + [value for _, value in out]):
        out.append((name, clean))
        used.add(name)


def collapse_layout_to_lines(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def t0_strict_recovery_views(prompt: str, masked_instruction: Optional[str], ubbi_obfuscated_text: Optional[str], refs: List[str], used: set[str]) -> List[Tuple[str, str]]:
    views: List[Tuple[str, str]] = []
    if masked_instruction and looks_like_artperception(prompt, masked_instruction):
        try:
            append_changed(views, "t0_ascii_grid_restored", restore_artperception(prompt, masked_instruction), refs, used)
        except Exception:
            pass
    if looks_like_artprompt(prompt):
        try:
            append_changed(views, "t0_ascii_mask_restored", restore_artprompt(prompt), refs, used)
        except Exception:
            pass
    if len(re.findall(r"(?i)ub", prompt)) >= 2 or ubbi_obfuscated_text:
        append_changed(views, "t0_ubbi_wrapper_decoded", reconstruct_ubbi_wrapper(prompt, ubbi_obfuscated_text), refs, used)
        core = extract_ubbi_core(prompt, ubbi_obfuscated_text)
        if core:
            append_changed(views, "t0_ubbi_payload_decoded", decode_ubbi(core), refs, used)
    for name, payload in extract_quoted_payloads(prompt):
        append_changed(views, f"t0_{name}", normalize_mask_tokens(payload), refs, used)
    if masked_instruction:
        append_changed(views, "t0_masked_instruction", normalize_mask_tokens(masked_instruction), refs, used)
    return views


def t1_structural_cleanup_views(base: str, refs: List[str], used: set[str]) -> List[Tuple[str, str]]:
    views: List[Tuple[str, str]] = []
    nfkc = unicodedata.normalize("NFKC", base)
    append_changed(views, "t1_pictographic_cleanup", remove_emoji(nfkc), refs, used)
    append_changed(views, "t1_zero_width_removed", ZERO_WIDTH_PATTERN.sub("", nfkc), refs, used)
    append_changed(views, "t1_inline_fragment_cleanup", strip_inline_noise(nfkc), refs, used)
    append_changed(views, "t1_repetition_layout_normalized", remove_anomalous_repetition(nfkc), refs, used)
    append_changed(views, "t1_bounded_character_repair", repair_character_edit_noise(nfkc), refs, used)
    return views


def t2_universal_normalization_views(base: str, refs: List[str], used: set[str]) -> List[Tuple[str, str]]:
    views: List[Tuple[str, str]] = []
    append_changed(views, "t2_unicode_nfkc", unicodedata.normalize("NFKC", base), refs, used)
    append_changed(views, "t2_line_trimmed", collapse_layout_to_lines(base), refs, used)
    append_changed(views, "t2_whitespace_flattened", normalize_space(base), refs, used)
    return views


def build_canonical_views(
    prompt: str,
    masked_instruction: Optional[str] = None,
    ubbi_obfuscated_text: Optional[str] = None,
) -> List[Tuple[str, str]]:
    raw = prompt.strip()
    used: set[str] = set()
    views: List[Tuple[str, str]] = []
    refs = [raw]
    t0_views = t0_strict_recovery_views(prompt, masked_instruction, ubbi_obfuscated_text, refs, used)
    views.extend(t0_views)
    base = t0_views[0][1] if t0_views else raw
    refs = [raw, base] + [value for _, value in views]
    t1_views = t1_structural_cleanup_views(base, refs, used)
    views.extend(t1_views)
    refs = [raw, base] + [value for _, value in views]
    t2_views = t2_universal_normalization_views(base, refs, used)
    views.extend(t2_views)
    return views or [("raw", raw)]
