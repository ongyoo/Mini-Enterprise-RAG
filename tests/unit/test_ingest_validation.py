import json
from pathlib import Path

from app.config import Settings
from app.rag.embeddings import HashingEmbeddings
from app.rag.ingest import SourceDocument, build_chunks, load_documents
from app.rag.splitter import split_content

DATA = Path(__file__).parents[2] / "data"


def _doc(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "document_id": "HR-LEAVE-009",
        "title": "ทดสอบ",
        "type": "SOP",
        "category": "leave",
        "content": "## หัวข้อ\nเนื้อหาทดสอบยาวพอ",
        "version": 1,
        "status": "active",
        "access_level": "internal",
        "source_authority": "authoritative",
        "synthetic": True,
        "updated_at": "2026-11-01",
    }
    return base | overrides


def _write(tmp_path: Path, *rows: object) -> Path:
    path = tmp_path / "docs.jsonl"
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows))
    return path


def test_shipped_corpus_is_valid_and_synthetic() -> None:
    docs, errors = load_documents(DATA / "documents.jsonl")
    assert errors == []
    assert docs
    assert all(d.synthetic for d in docs)
    codes = {d.document_id for d in docs}
    for d in docs:
        assert set(d.related_document_ids) <= codes


def test_malformed_lines_are_reported(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "{not json",
        _doc(synthetic=False),
        _doc(document_id="bad id"),
        _doc(type="Chat"),
        _doc(document_id="HR-LEAVE-010"),
    )
    docs, errors = load_documents(path)
    assert [d.document_id for d in docs] == ["HR-LEAVE-010"]
    assert len(errors) == 4


def test_conflicting_duplicates_are_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, _doc(), _doc(content="## อื่น\nเนื้อหาไม่เหมือนกัน"))
    docs, errors = load_documents(path)
    assert docs == []
    assert "conflicting" in errors[0]


def test_identical_duplicates_are_collapsed(tmp_path: Path) -> None:
    docs, errors = load_documents(_write(tmp_path, _doc(), _doc()))
    assert len(docs) == 1
    assert errors == []


def test_sop_splits_by_heading_and_prefixes_title() -> None:
    chunks = split_content(
        title="T",
        content="intro\n## A\naaa\n## B\nbbb",
        doc_type="SOP",
        chunk_size=800,
        chunk_overlap=80,
    )
    assert chunks == ["T\n## A\naaa", "T\n## B\nbbb"]


def test_chat_stays_one_chunk() -> None:
    assert (
        len(
            split_content(
                title="T", content="a\n## b", doc_type="Chat", chunk_size=800, chunk_overlap=80
            )
        )
        == 1
    )


def test_chunk_keys_are_deterministic() -> None:
    doc = SourceDocument.model_validate(_doc())
    settings = Settings()
    assert build_chunks(doc, settings) == build_chunks(doc, settings)
    assert build_chunks(doc, settings)[0][0].startswith("HR-LEAVE-009:1:0:")


def test_hashing_embeddings_are_normalized_and_lexically_similar() -> None:
    emb = HashingEmbeddings(256)
    a, b, c = emb.embed_documents(["ลาป่วยยื่นที่ไหน", "ยื่นลาป่วยที่ไหน", "ราคาน้ำมัน"])
    dot = lambda x, y: sum(i * j for i, j in zip(x, y, strict=True))  # noqa: E731
    assert abs(dot(a, a) - 1) < 1e-9
    assert dot(a, b) > dot(a, c)


def test_handbook_export_format_is_normalized(tmp_path: Path) -> None:
    record = {
        "document_id": "HR-001",
        "title": "นโยบายการลาป่วย",
        "department": "Human Resources",
        "classification": "INTERNAL",
        "language": "th",
        "effective_date": "2026-11-22",
        "content": "หลักเกณฑ์: ยื่นลาป่วยในระบบ HR Portal",
        "source_type": "policy",
        "version": "1.0",
    }
    path = tmp_path / "docs.jsonl"
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    docs, errors = load_documents(path)

    assert errors == []
    doc = docs[0]
    assert (doc.type, doc.source_authority, doc.access_level, doc.version) == (
        "SOP",
        "authoritative",
        "internal",
        1,
    )
    assert doc.category == "Human Resources"


def test_fractional_version_is_rejected(tmp_path: Path) -> None:
    record = {
        "document_id": "HR-001",
        "title": "t",
        "department": "HR",
        "classification": "INTERNAL",
        "effective_date": "2026-11-22",
        "content": "เนื้อหาตัวอย่างยาวพอ",
        "source_type": "policy",
        "version": "1.1",
    }
    path = tmp_path / "docs.jsonl"
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    docs, errors = load_documents(path)

    assert docs == [] and len(errors) == 1
