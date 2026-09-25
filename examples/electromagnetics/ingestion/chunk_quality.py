"""Deterministic pre-ingest chunk quality checks.

This checks structural loss before RAGFlow sees a document. It is not a
semantic retrieval accuracy claim; that second layer needs a labelled query
set and the actual RAGFlow chunks.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChunkQualityReport:
    chunk_count: int
    coverage: float
    boundary_integrity: float
    structure_integrity: float
    score: float
    violations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_count": self.chunk_count,
            "coverage": round(self.coverage, 4),
            "boundary_integrity": round(self.boundary_integrity, 4),
            "structure_integrity": round(self.structure_integrity, 4),
            "score": round(self.score, 4),
            "violations": self.violations,
        }


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _quality_tokens(value: str) -> list[str]:
    """Extract stable evidence tokens without requiring identical markup.

    PDF layout parsers legitimately differ in whitespace, equation markup,
    HTML tables, and image descriptions.  Exact-string equality therefore
    measures parser formatting drift rather than information loss.
    """
    return re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?|\d+(?:\.\d+)?|[\u4e00-\u9fff]", value.lower())


def _split_long_prose_unit(unit: str, max_chars: int) -> list[str]:
    """Split oversized prose without treating MinerU hard wraps as boundaries."""
    compact = _normalise(unit)
    if len(compact) <= max_chars:
        return [compact]
    # A hard-wrapped MinerU paragraph is still one sentence. Prefer complete
    # sentence boundaries even when that produces an oversized chunk; a
    # nominal character target must not create a false parameter statement by
    # starting the next chunk halfway through a sentence.
    fragments = [
        fragment.strip()
        for fragment in re.split(r"(?<=[.!?。！？])\s+", compact)
        if fragment.strip()
    ]
    if len(fragments) == 1:
        return [compact]
    parts: list[str] = []
    current = ""
    for fragment in fragments:
        if current and len(current) + len(fragment) + 1 > max_chars:
            parts.append(current.strip())
            current = ""
        # Keep a single oversized sentence intact instead of falling back to
        # word-level splitting, which would manufacture mid-sentence starts.
        if len(fragment) > max_chars:
            if current:
                parts.append(current.strip())
                current = ""
            parts.append(fragment)
        else:
            current = f"{current} {fragment}".strip()
    if current:
        parts.append(current.strip())
    return parts or [compact]


def split_markdown(markdown: str, max_chars: int = 6000) -> list[str]:
    """Split at Markdown block boundaries and never split protected blocks."""
    if max_chars < 500:
        raise ValueError("max_chars must be at least 500")
    lines = markdown.splitlines()
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    in_math = False
    in_table = False
    for line in lines:
        stripped = line.strip()
        is_fence = stripped.startswith("```") or stripped.startswith("~~~")
        is_heading = bool(re.match(r"^#{1,6}\s+", stripped))
        is_table = stripped.startswith("|") or bool(re.match(r"(?i)^<(?:table|tr|td|th)\b", stripped))
        if current and in_table and not is_table and not in_fence and not in_math:
            blocks.append("\n".join(current).strip())
            current = []
            in_table = False
        if current and is_heading and not in_fence and not in_math and not in_table:
            blocks.append("\n".join(current).strip())
            current = []
        current.append(line)
        if stripped.lower().startswith("<table") and "</table>" in stripped.lower():
            blocks.append("\n".join(current).strip())
            current = []
            in_table = False
            continue
        if is_fence:
            in_fence = not in_fence
        if not in_fence:
            # MinerU emits display equations as a delimiter line, followed
            # by a blank line and the formula body. Blank lines must not
            # terminate such a protected block.
            dollar_count = stripped.count("$$")
            if dollar_count % 2:
                in_math = not in_math
            if stripped == r"\[":
                in_math = True
            elif stripped == r"\]":
                in_math = False
        in_table = is_table and not in_fence and not in_math
        if not stripped and current and not in_fence and not in_math and not in_table:
            blocks.append("\n".join(current).strip())
            current = []
    if current:
        blocks.append("\n".join(current).strip())
    blocks = [block for block in blocks if block]

    # MinerU sometimes places prose and a complete HTML table on the same
    # physical line. Split those inline tables into protected blocks before
    # applying the chunk-size logic, otherwise a long parameter table can be
    # cut in the middle of a row.
    separated_blocks: list[str] = []
    table_pattern = re.compile(r"<table\b.*?</table>", flags=re.I | re.S)
    for block in blocks:
        matches = list(table_pattern.finditer(block))
        if not matches:
            separated_blocks.append(block)
            continue
        cursor = 0
        for match in matches:
            prefix = block[cursor:match.start()].strip()
            if prefix:
                separated_blocks.append(prefix)
            separated_blocks.append(match.group(0).strip())
            cursor = match.end()
        suffix = block[cursor:].strip()
        if suffix:
            separated_blocks.append(suffix)
    blocks = separated_blocks

    # MinerU occasionally inserts a blank line at a PDF line wrap. If the
    # previous block ends in a grammatical connector, joining the next block
    # restores the sentence before chunk sizing. This is deliberately narrow
    # so real headings, lists, tables, and figure captions remain separate.
    merged_blocks: list[str] = []
    dangling = re.compile(
        r"(?i)(?:\b(?:the|a|an|of|and|or|in|on|to|for|with|is|are|was|were|would|that|this|as|by|from|than|into|at|its)\s*)$"
    )
    structural_start = re.compile(r"(?i)^(?:#{1,6}\s|[-*+>]\s|\|\s*|(?:figure|fig\.|table|图)\s*\d+\b|\d+[.)]?\s+)")
    for block in blocks:
        if merged_blocks and dangling.search(merged_blocks[-1]) and not structural_start.match(block.lstrip()):
            merged_blocks[-1] = f"{merged_blocks[-1]}\n\n{block}".strip()
        else:
            merged_blocks.append(block)
    blocks = merged_blocks

    chunks: list[str] = []
    current = ""
    for block in blocks:
        if current and len(current) + len(block) + 2 > max_chars:
            chunks.append(current.strip())
            current = ""
        if len(block) <= max_chars:
            current = f"{current}\n\n{block}".strip()
            continue
        # A pathological paragraph is split only at line boundaries. Fenced
        # blocks, tables, and display-math blocks stay whole even when they
        # exceed max_chars: structural fidelity is more important than a
        # nominal chunk-size target.
        if current:
            chunks.append(current.strip())
            current = ""
        protected_block = block.lstrip().startswith(("|", "<table", "<tr", "```", "~~~", "$$", r"\["))
        if protected_block:
            chunks.append(block.strip())
            continue
        # For prose, prefer sentence boundaries over arbitrary line breaks.
        # Tables/fenced blocks remain line-based because punctuation is not a
        # reliable structural delimiter there.
        units = [block]
        if not (block.lstrip().startswith("|") or "```" in block or "~~~" in block):
            units = [unit for unit in re.split(r"(?<=[.!?。！？；;])\s+", block) if unit]
        part = ""
        for unit in units:
            expanded = [unit]
            if len(unit) > max_chars and not (unit.lstrip().startswith("|") or "```" in unit or "~~~" in unit):
                expanded = _split_long_prose_unit(unit, max_chars)
            for piece in expanded:
                if part and len(part) + len(piece) + 2 > max_chars:
                    chunks.append(part.strip())
                    part = ""
                part = f"{part}\n\n{piece}".strip()
        if part:
            current = part
    if current:
        chunks.append(current.strip())
    return chunks


def evaluate_chunks(markdown: str, chunks: list[str]) -> ChunkQualityReport:
    violations: list[str] = []
    source = _normalise(markdown)
    combined = _normalise("\n".join(chunks))
    if not source:
        coverage = 0.0
    elif combined == source or source in combined:
        coverage = 1.0
    else:
        source_tokens = Counter(_quality_tokens(source))
        combined_tokens = Counter(_quality_tokens(combined))
        matched = sum((source_tokens & combined_tokens).values())
        coverage = matched / max(1, sum(source_tokens.values()))
    if not chunks:
        violations.append("no_chunks")
    boundary_issues = 0
    structure_issues = 0
    for index, chunk in enumerate(chunks):
        if not chunk.strip():
            boundary_issues += 1
            violations.append(f"empty_chunk:{index}")
        if index > 0:
            previous = chunks[index - 1].rstrip()
            current_start = chunk.lstrip()
            code_like_start = bool(
                re.search(r"[=;]", current_start[:500])
                and ("_" in current_start[:500] or ";" in current_start[:500] or "<sub>" in current_start[:500])
            )
            protected_start = current_start.startswith(("#", "-", "*", "+", ">", "|", "```", "~~~")) or code_like_start or bool(re.match(r"(?i)^(abstract|keywords?|introduction|conclusion|references?)\b", current_start)) or bool(re.match(r"(?i)^(?:figure|fig\.|table|图)\s*\d+\b", current_start)) or bool(re.match(r"^\d+[.)]?\s+", current_start))
            last_line = next((line.strip() for line in reversed(previous.splitlines()) if line.strip()), "")
            ends_sentence = bool(re.search(r"[.!?。！？；;:：）)\]】]$", previous))
            ends_structure = (
                last_line.startswith("#")
                or previous.endswith(("$$", "$", "</table>", "</div>"))
                or bool(re.match(r"(?i)^(?:figure|fig\.|table|图)\s*\d+\b", last_line))
                or bool(re.search(r"\b\d+\s*$", last_line))
                # MinerU may emit a section label such as ``A``/``B`` on its
                # own line. A chunk ending at that label is a structural
                # boundary, not a prose sentence split.
                or bool(re.match(r"^[A-Z]$", last_line))
                # PDF extraction may preserve subsection headings as
                # ``A. Title``/``B) Title`` at the end of a chunk.
                or bool(re.match(r"^[A-Z][.)]\s+\S", last_line))
                # Algorithm listings commonly end with these control-flow
                # markers, which are also safe structural boundaries.
                or bool(re.match(r"(?i)^(?:\d+\s*:\s*)?end\s+(?:for|if|while)$", last_line))
                or (len(last_line.split()) >= 6 and all(len(token) == 1 for token in last_line.split()))
            )
            if not protected_start and not ends_sentence and not ends_structure and re.match(r"^[a-zA-Z0-9]", current_start):
                boundary_issues += 1
                violations.append(f"possible_mid_sentence_start:{index}")
        if chunk.count("```") % 2:
            structure_issues += 1
            violations.append(f"unclosed_code_fence:{index}")
        if chunk.count("$$") % 2:
            structure_issues += 1
            violations.append(f"unclosed_formula:{index}")
        if any(line.strip().startswith("|") for line in chunk.splitlines()):
            table_lines = [line.strip() for line in chunk.splitlines() if line.strip().startswith("|")]
            if len(table_lines) < 2 or any(not line.endswith("|") for line in table_lines):
                structure_issues += 1
                violations.append(f"possible_truncated_table:{index}")
    count = max(1, len(chunks))
    boundary_integrity = max(0.0, 1.0 - boundary_issues / count)
    structure_integrity = max(0.0, 1.0 - structure_issues / count)
    score = 0.45 * coverage + 0.3 * boundary_integrity + 0.25 * structure_integrity
    return ChunkQualityReport(len(chunks), coverage, boundary_integrity, structure_integrity, score, violations)


def evaluate_positioned_chunks(markdown: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate service-side PDF chunks using their page/box evidence.

    RAGFlow's PDF chunks may contain harmless MinerU formatting differences
    and may be split at layout boxes, so the generic sentence-boundary score
    is intentionally not the primary gate here.  Page/coordinate positions
    are a stronger invariant for visual fallback and citation traceability.
    """
    texts = [str(item.get("content") or "") for item in chunks]
    base = evaluate_chunks(markdown, texts)
    positioned = 0
    for item in chunks:
        positions = item.get("positions")
        if isinstance(positions, list) and any(isinstance(pos, list) and pos for pos in positions):
            positioned += 1
    position_coverage = positioned / max(1, len(chunks))
    # The score emphasizes source-token recall and page traceability, while
    # retaining a structural penalty for broken formulas/tables.
    score = 0.60 * base.coverage + 0.25 * position_coverage + 0.15 * base.structure_integrity
    violations = list(base.violations)
    if positioned < len(chunks):
        violations.append(f"missing_page_positions:{len(chunks) - positioned}")
    return {
        "chunk_count": base.chunk_count,
        "coverage": round(base.coverage, 4),
        "boundary_integrity": round(base.boundary_integrity, 4),
        "structure_integrity": round(base.structure_integrity, 4),
        "position_coverage": round(position_coverage, 4),
        "score": round(score, 4),
        "violations": violations,
    }
