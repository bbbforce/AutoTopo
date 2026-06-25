"""Parent-child knowledge-base chunking tests."""

from __future__ import annotations

from autotopo.rag.kb_loader import load_knowledge_chunks


def test_markdown_headings_split_into_multiple_chunks(tmp_path):
    root = tmp_path / "knowledge_base"
    kb = root / "case_template_kb"
    kb.mkdir(parents=True)
    (kb / "templates.md").write_text(
        "# Templates\n\n"
        "## MBB\n\n"
        "Benchmark type: `mbb`.\n\n"
        "## Cantilever\n\n"
        "Benchmark type: `cantilever`.\n",
        encoding="utf-8",
    )

    chunks = load_knowledge_chunks(root)

    assert {chunk.heading for chunk in chunks} >= {"MBB", "Cantilever"}
    assert len(chunks) >= 2


def test_code_block_is_separate_chunk(tmp_path):
    root = tmp_path / "knowledge_base"
    kb = root / "solver_api_kb"
    kb.mkdir(parents=True)
    (kb / "api.md").write_text(
        "# API\n\n"
        "Use this call:\n\n"
        "```python\n"
        "solve(problem, optimizer='MMA')\n"
        "```\n\n"
        "Then inspect density.\n",
        encoding="utf-8",
    )

    chunks = load_knowledge_chunks(root)

    assert any(chunk.chunk_type == "code_block" and "solve(problem" in chunk.content for chunk in chunks)


def test_failure_mode_gets_failure_case_chunk(tmp_path):
    root = tmp_path / "knowledge_base"
    kb = root / "failure_kb"
    kb.mkdir(parents=True)
    (kb / "execution.md").write_text(
        "# Failures\n\n"
        "`singular_stiffness_matrix`: missing supports cause rigid body motion.\n"
        "Repair: add supports.\n\n"
        "`shape_mismatch`: arrays disagree.\n",
        encoding="utf-8",
    )

    chunks = load_knowledge_chunks(root)

    assert any(chunk.chunk_type == "failure_case" and "singular_stiffness_matrix" in chunk.content for chunk in chunks)
