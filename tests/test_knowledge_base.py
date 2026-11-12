
import json
from pathlib import Path

import faiss
import pytest


ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
INDEX_PATH = ROOT / "data" / "index" / "faiss.index"


@pytest.fixture
def knowledge_base():
    if not CHUNKS_PATH.exists() or not INDEX_PATH.exists():
        pytest.skip(
            "Knowledge base not built. Run the extraction, "
            "chunking and indexing pipeline first."
        )

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        chunks = [json.loads(line) for line in file if line.strip()]

    index = faiss.read_index(str(INDEX_PATH))

    return chunks, index


def test_chunks_file_is_not_empty(knowledge_base):
    chunks, _ = knowledge_base

    assert len(chunks) > 0


def test_chunks_have_required_metadata(knowledge_base):
    chunks, _ = knowledge_base

    required_fields = {
        "chunk_id",
        "book_id",
        "book_title",
        "pdf_page",
        "text",
    }

    for chunk in chunks:
        assert required_fields.issubset(chunk.keys())
        assert isinstance(chunk["text"], str)
        assert chunk["text"].strip()


def test_faiss_index_matches_number_of_chunks(knowledge_base):
    chunks, index = knowledge_base

    assert index.ntotal == len(chunks)


def test_embedding_dimension(knowledge_base):
    _, index = knowledge_base

    assert index.d == 384
