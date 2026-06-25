"""Local knowledge-base loading with parent-child chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class KnowledgeDocument:
    """A parent knowledge-base file.

    `evidence_id` is retained for the first keyword-RAG tests and callers; new
    retrieval code treats `parent_id` as the stable parent identifier.
    """

    evidence_id: str
    source: str
    kind: str
    content: str
    parent_id: str = ""
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.parent_id:
            object.__setattr__(self, "parent_id", self.evidence_id)
        if not self.title:
            object.__setattr__(self, "title", _title_from_content(self.content, self.source))


@dataclass(frozen=True)
class KnowledgeChunk:
    """A child chunk used for retrieval."""

    chunk_id: str
    parent_id: str
    source: str
    kind: str
    title: str
    heading: str
    content: str
    chunk_type: str
    start_line: int
    end_line: int
    metadata: dict[str, Any] = field(default_factory=dict)


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TABLE_RE = re.compile(r"^\s*\|.+\|\s*$")
_FAILURE_MODE_RE = re.compile(r"`?([a-z][a-z0-9_]+)`?\s*:")
_SLUG_RE = re.compile(r"[^a-zA-Z0-9_]+")


def _kind_from_path(path: Path) -> str:
    for part in path.parts:
        if part.endswith("_kb"):
            return part.removesuffix("_kb")
    return "generic"


def _parent_id_from_path(path: Path) -> str:
    return path.as_posix().replace("/", "__").rsplit(".", 1)[0]


def _title_from_content(content: str, source: str) -> str:
    for line in content.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            return match.group(2).strip()
    return Path(source).stem.replace("_", " ").title()


def _slug(text: str) -> str:
    slug = _SLUG_RE.sub("_", text.lower()).strip("_")
    return slug[:48] or "chunk"


def _chunk_type_for(content: str, kind: str) -> str:
    stripped = content.strip()
    if _TABLE_RE.match(stripped):
        return "table"
    if kind == "failure" and _FAILURE_MODE_RE.search(stripped.splitlines()[0] if stripped else ""):
        return "failure_case"
    if kind == "physics_rule":
        return "rule"
    return "paragraph"


def _bounded_parent_context(content: str, start_line: int, end_line: int, radius: int = 2) -> str:
    lines = content.splitlines()
    start = max(1, start_line - radius)
    end = min(len(lines), end_line + radius)
    return "\n".join(lines[start - 1:end]).strip()


def _make_chunk(
    doc: KnowledgeDocument,
    *,
    index: int,
    heading: str,
    content: str,
    chunk_type: str,
    start_line: int,
    end_line: int,
) -> KnowledgeChunk:
    chunk_id = f"{doc.parent_id}::{_slug(heading)}::{index:03d}"
    return KnowledgeChunk(
        chunk_id=chunk_id,
        parent_id=doc.parent_id,
        source=doc.source,
        kind=doc.kind,
        title=doc.title,
        heading=heading,
        content=content.strip(),
        chunk_type=chunk_type,
        start_line=start_line,
        end_line=end_line,
        metadata={
            **doc.metadata,
            "parent_title": doc.title,
            "parent_context": _bounded_parent_context(doc.content, start_line, end_line),
        },
    )


def chunk_document(doc: KnowledgeDocument, *, max_chars: int = 1600) -> list[KnowledgeChunk]:
    """Split one parent document into deterministic retrieval chunks."""

    lines = doc.content.splitlines()
    chunks: list[KnowledgeChunk] = []
    heading = doc.title
    paragraph: list[str] = []
    paragraph_start = 1
    code: list[str] = []
    code_start = 1
    in_code = False

    def append_chunk(content: str, chunk_type: str, start_line: int, end_line: int) -> None:
        if not content.strip():
            return
        if chunk_type != "code_block" and len(content) > max_chars:
            sub_start = start_line
            buffer: list[str] = []
            for part in re.split(r"(\n\s*\n)", content):
                buffer.append(part)
                joined = "".join(buffer).strip()
                if len(joined) >= max_chars:
                    chunks.append(
                        _make_chunk(
                            doc,
                            index=len(chunks) + 1,
                            heading=heading,
                            content=joined,
                            chunk_type=chunk_type,
                            start_line=sub_start,
                            end_line=end_line,
                        )
                    )
                    buffer = []
                    sub_start = end_line
            tail = "".join(buffer).strip()
            if tail:
                chunks.append(
                    _make_chunk(
                        doc,
                        index=len(chunks) + 1,
                        heading=heading,
                        content=tail,
                        chunk_type=chunk_type,
                        start_line=sub_start,
                        end_line=end_line,
                    )
                )
            return
        chunks.append(
            _make_chunk(
                doc,
                index=len(chunks) + 1,
                heading=heading,
                content=content,
                chunk_type=chunk_type,
                start_line=start_line,
                end_line=end_line,
            )
        )

    def flush_paragraph(end_line: int) -> None:
        nonlocal paragraph
        if not paragraph:
            return
        content = "\n".join(paragraph).strip()
        append_chunk(content, _chunk_type_for(content, doc.kind), paragraph_start, end_line)
        paragraph = []

    for line_no, line in enumerate(lines, start=1):
        if line.strip().startswith("```"):
            if not in_code:
                flush_paragraph(line_no - 1)
                in_code = True
                code_start = line_no
                code = [line]
            else:
                code.append(line)
                append_chunk("\n".join(code), "code_block", code_start, line_no)
                in_code = False
                code = []
            continue

        if in_code:
            code.append(line)
            continue

        match = _HEADING_RE.match(line)
        if match:
            flush_paragraph(line_no - 1)
            heading = match.group(2).strip()
            continue

        if not line.strip():
            flush_paragraph(line_no - 1)
            continue

        if not paragraph:
            paragraph_start = line_no
        paragraph.append(line)

    if in_code and code:
        append_chunk("\n".join(code), "code_block", code_start, len(lines))
    flush_paragraph(len(lines))

    if not chunks and doc.content.strip():
        chunks.append(
            _make_chunk(
                doc,
                index=1,
                heading=heading,
                content=doc.content,
                chunk_type=_chunk_type_for(doc.content, doc.kind),
                start_line=1,
                end_line=max(1, len(lines)),
            )
        )
    return chunks


def load_knowledge_base(root: str | Path = "knowledge_base") -> list[KnowledgeDocument]:
    """Recursively load Markdown/TXT knowledge-base files as parent documents."""

    root_path = Path(root)
    if not root_path.exists():
        return []

    docs: list[KnowledgeDocument] = []
    for path in sorted(root_path.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        rel = path.relative_to(root_path)
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        parent_id = _parent_id_from_path(rel)
        docs.append(
            KnowledgeDocument(
                evidence_id=parent_id,
                parent_id=parent_id,
                source=rel.as_posix(),
                kind=_kind_from_path(rel),
                title=_title_from_content(content, rel.as_posix()),
                content=content,
                metadata={"path": rel.as_posix()},
            )
        )
    return docs


def load_knowledge_chunks(root: str | Path = "knowledge_base") -> list[KnowledgeChunk]:
    """Load all knowledge-base child chunks."""

    chunks: list[KnowledgeChunk] = []
    for doc in load_knowledge_base(root):
        chunks.extend(chunk_document(doc))
    return chunks
