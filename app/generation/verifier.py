"""Citation verification: check each cited claim against the chunk it cites.

For every sentence that carries ``[n]`` markers, the claim text (markers removed)
is compared with chunk ``n``. A claim is *verified* when either the fuzzy
partial-ratio against the chunk reaches ``VERIFY_PARTIAL_THRESHOLD`` or the share
of the claim's content words found in the chunk reaches
``VERIFY_RECALL_THRESHOLD`` (the latter tolerates light paraphrase), AND every
number in the claim also appears in the chunk (``VERIFY_NUMBER_GUARD``).
A claim citing several sources such as ``[1][2]`` that fails individually may still
pass against the concatenation of those sources (it then verifies all of them).
A source is verified only if every claim citing it is verified.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Mapping

from app import config

CITATION_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_TRAILING_CITATION_RE = re.compile(r"([.!?])\s+((?:\[\d+(?:\s*[,;]\s*\d+)*\]\s*)+)")
_NON_WORD_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
_LIST_MARKER_RE = re.compile(r"^\s*\d+[.)]\s+")
_STOPWORDS = frozenset(
    "the a an and or but of in on at to for with by from as is are was were be been being this that these "
    "those it its their there which who whom what when where while than then also such can could may might "
    "will would should has have had not into over under between within about using used use based".split()
)
_PERCENT = 100.0
_FALLBACK_STEP_DIVISOR = 4


@dataclass
class SourceCheck:
    """Verification outcome for one cited source."""

    verified: bool
    match_score: float
    claims_checked: int


@dataclass
class VerificationResult:
    """Aggregate verification outcome for a generated answer."""

    per_source: dict[int, SourceCheck] = field(default_factory=dict)
    unverified: list[int] = field(default_factory=list)
    invalid: list[int] = field(default_factory=list)
    claims_total: int = 0
    claims_verified: int = 0

    @property
    def cited(self) -> list[int]:
        """Valid source numbers that the answer cites."""
        return sorted(self.per_source)

    @property
    def verified_ratio(self) -> float:
        """Fraction of cited sources that verified (0.0 when nothing is cited)."""
        if not self.per_source:
            return 0.0
        return sum(1 for c in self.per_source.values() if c.verified) / len(self.per_source)

    @property
    def coverage(self) -> float:
        """Share of the answer's substantive sentences that carry a verified citation."""
        return self.claims_verified / self.claims_total if self.claims_total else 0.0


def normalize(text: str) -> str:
    """Lower-case, strip punctuation and collapse whitespace."""
    return _WHITESPACE_RE.sub(" ", _NON_WORD_RE.sub(" ", text.lower())).strip()


def _fallback_partial_ratio(short: str, long: str) -> float:
    """Sliding-window difflib approximation of rapidfuzz's partial_ratio."""
    if not short or not long:
        return 0.0
    if len(short) > len(long):
        short, long = long, short
    if short in long:
        return _PERCENT
    window = len(short)
    step = max(1, window // _FALLBACK_STEP_DIVISOR)
    best = 0.0
    for start in range(0, max(1, len(long) - window + 1), step):
        best = max(best, SequenceMatcher(None, short, long[start : start + window]).ratio())
    return best * _PERCENT


def partial_ratio(claim: str, chunk: str) -> float:
    """Fuzzy substring similarity in [0, 100] (rapidfuzz when installed)."""
    try:
        from rapidfuzz import fuzz

        return float(fuzz.partial_ratio(claim, chunk))
    except ImportError:
        return _fallback_partial_ratio(claim, chunk)


def _content_words(text: str) -> list[str]:
    """Words that carry meaning: no stopwords, minimum length."""
    return [
        w for w in normalize(text).split() if len(w) >= config.VERIFY_MIN_WORD_LENGTH and w not in _STOPWORDS
    ]


def _stem(word: str) -> str:
    """Crude prefix stem so 'retrieval'/'retrieves' count as the same word."""
    return word[: config.VERIFY_STEM_LENGTH]


def content_recall(claim: str, chunk: str) -> float:
    """Share of the claim's content words (stemmed) that appear in the chunk."""
    words = _content_words(claim)
    if not words:
        return 0.0
    chunk_stems = {_stem(w) for w in normalize(chunk).split()}
    return sum(1 for w in words if _stem(w) in chunk_stems) / len(words)


def _numbers(text: str) -> set[str]:
    """Numeric tokens in raw text (thousands separators removed, decimals kept)."""
    return set(_NUMBER_RE.findall(_THOUSANDS_RE.sub("", text)))


def numbers_supported(claim: str, chunk_text: str) -> bool:
    """True when every number in the claim also appears in the chunk."""
    claim_numbers = _numbers(_LIST_MARKER_RE.sub("", claim))
    return claim_numbers <= _numbers(chunk_text)


def extract_cited_numbers(text: str) -> list[int]:
    """All source numbers cited anywhere in ``text`` (duplicates removed, ordered)."""
    seen: dict[int, None] = {}
    for match in CITATION_RE.finditer(text):
        for number in re.findall(r"\d+", match.group(1)):
            seen.setdefault(int(number), None)
    return list(seen)


def _segments(answer: str) -> list[str]:
    """Split an answer into sentence-like segments, citations attached to their sentence."""
    normalised = _TRAILING_CITATION_RE.sub(lambda m: f" {m.group(2).strip()}{m.group(1)}", answer)
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(normalised) if s.strip()]


def extract_claims(answer: str) -> list[tuple[str, list[int]]]:
    """Split an answer into ``(claim_text, [cited numbers])`` for cited sentences."""
    claims: list[tuple[str, list[int]]] = []
    for segment in _segments(answer):
        numbers = extract_cited_numbers(segment)
        if not numbers:
            continue
        text = CITATION_RE.sub("", segment).strip(" -*•|#>\t")
        claims.append((text, numbers))
    return claims


def count_uncited_sentences(answer: str) -> int:
    """Substantive sentences that carry no citation (they lower coverage)."""
    return sum(
        1
        for segment in _segments(answer)
        if not extract_cited_numbers(segment) and len(_content_words(segment)) >= config.VERIFY_COVERAGE_MIN_WORDS
    )


def score_claim(claim: str, chunk_text: str) -> tuple[bool, float]:
    """Return ``(verified, score_0_to_100)`` for one claim against one chunk."""
    claim_norm, chunk_norm = normalize(claim), normalize(chunk_text)
    partial = partial_ratio(claim_norm, chunk_norm)
    recall = content_recall(claim, chunk_text)
    score = max(partial, recall * _PERCENT)
    verified = partial >= config.VERIFY_PARTIAL_THRESHOLD or recall >= config.VERIFY_RECALL_THRESHOLD
    if verified and config.VERIFY_NUMBER_GUARD and not numbers_supported(claim, chunk_text):
        verified = False
        score = min(score, config.VERIFY_NUMBER_MISMATCH_SCORE)
    return verified, score


def verify_citations(answer: str, chunk_texts: Mapping[int, str]) -> VerificationResult:
    """Verify every cited claim in ``answer`` against ``chunk_texts`` (1-based keys)."""
    result = VerificationResult()
    invalid: set[int] = set()
    outcomes: dict[int, list[tuple[bool, float]]] = {}
    for claim, numbers in extract_claims(answer):
        valid = [n for n in numbers if n in chunk_texts]
        invalid.update(n for n in numbers if n not in chunk_texts)
        if not valid:
            result.claims_total += 1  # cites only non-existent sources: counts against coverage
            continue
        if len(_content_words(claim)) < config.VERIFY_MIN_CONTENT_WORDS:
            for number in valid:
                outcomes.setdefault(number, [])  # too short to judge; cited but not penalised
            continue
        checks = {number: score_claim(claim, chunk_texts[number]) for number in valid}
        if len(valid) > 1 and not all(ok for ok, _ in checks.values()):
            joint_ok, joint_score = score_claim(claim, " ".join(chunk_texts[n] for n in valid))
            if joint_ok:  # supported by the cited passages taken together
                checks = {n: (True, max(score, joint_score)) for n, (_, score) in checks.items()}
        for number, check in checks.items():
            outcomes.setdefault(number, []).append(check)
        result.claims_total += 1
        if all(ok for ok, _ in checks.values()):
            result.claims_verified += 1
    for number, checks_list in outcomes.items():
        verified = all(ok for ok, _ in checks_list) if checks_list else True
        score = min((s for _, s in checks_list), default=_PERCENT)
        result.per_source[number] = SourceCheck(verified, round(score, 1), len(checks_list))
        if not verified:
            result.unverified.append(number)
    result.unverified.sort()
    result.invalid = sorted(invalid)
    result.claims_total += count_uncited_sentences(answer)
    return result