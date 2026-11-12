
import json
import logging
from pathlib import Path


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT_DIR / "data" / "processed" / "pages.jsonl"
OUTPUT_FILE = ROOT_DIR / "data" / "processed" / "chunks.jsonl"

CHUNK_SIZE = 1800
CHUNK_OVERLAP = 250


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)


# --------------------------------------------------
# Text splitting
# --------------------------------------------------

def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[str]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size.")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(start + chunk_size, text_length)

        # Try to find a natural boundary near the end
        if end < text_length:
            search_start = start + int(chunk_size * 0.7)

            newline_position = text.rfind(
                "\n", search_start, end
            )
            space_position = text.rfind(
                " ", search_start, end
            )

            boundary = max(newline_position, space_position)

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        # Move forward while keeping some context
        start = max(end - overlap, start + 1)

    return chunks


# --------------------------------------------------
# Corpus chunking
# --------------------------------------------------

def build_chunks():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    total_pages = 0
    total_chunks = 0

    with INPUT_FILE.open("r", encoding="utf-8") as source, \
         OUTPUT_FILE.open("w", encoding="utf-8") as output:

        for line in source:

            page = json.loads(line)
            page_chunks = split_text(page["text"])

            for index, chunk_text in enumerate(page_chunks):

                chunk_record = {
                    "chunk_id": f'{page["book_id"]}_p{page["pdf_page"]}_c{index}',
                    "book_id": page["book_id"],
                    "book_title": page["book_title"],
                    "source_file": page["source_file"],
                    "pdf_page": page["pdf_page"],
                    "chunk_index": index,
                    "text": chunk_text,
                }

                output.write(
                    json.dumps(
                        chunk_record,
                        ensure_ascii=False
                    ) + "\n"
                )

                total_chunks += 1

            total_pages += 1

    logging.info("Chunking completed.")
    logging.info("Pages processed: %d", total_pages)
    logging.info("Total chunks: %d", total_chunks)
    logging.info("Output saved to: %s", OUTPUT_FILE)


if __name__ == "__main__":
    build_chunks()
