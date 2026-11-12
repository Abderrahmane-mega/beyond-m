
import json
import re
import logging
from pathlib import Path

import pymupdf


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
BOOKS_DIR = ROOT_DIR / "books"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "pages.jsonl"

BOOK_TITLES = {
    "bishop": "Pattern Recognition and Machine Learning",
    "deep learning": "Deep Learning",
    "geron": "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow",
    "murphy": "Machine Learning: A Probabilistic Perspective",
}

LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
}


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)


# --------------------------------------------------
# Text preprocessing
# --------------------------------------------------

def clean_text(text: str) -> str:
    """Apply conservative text normalization."""

    # Replace common ligatures
    for ligature, replacement in LIGATURES.items():
        text = text.replace(ligature, replacement)

    # Remove soft hyphens
    text = text.replace("\u00ad", "")

    # Normalize Unicode without converting mathematical
    # superscripts and subscripts into ordinary characters
    import unicodedata
    text = unicodedata.normalize("NFC", text)

    # Remove excessive horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# --------------------------------------------------
# Book metadata
# --------------------------------------------------

def get_book_title(pdf_path: Path, metadata: dict) -> str:
    """Get a readable book title."""

    stem = pdf_path.stem.lower()

    for key, title in BOOK_TITLES.items():
        if key in stem:
            return title

    metadata_title = metadata.get("title", "").strip()

    if metadata_title:
        return metadata_title

    return pdf_path.stem.replace("_", " ").replace("-", " ").title()


# --------------------------------------------------
# PDF extraction
# --------------------------------------------------

def extract_pdf(pdf_path: Path) -> list[dict]:
    """Extract one record per PDF page."""

    logging.info("Extracting: %s", pdf_path.name)

    records = []

    with pymupdf.open(pdf_path) as document:

        metadata = document.metadata or {}
        book_title = get_book_title(pdf_path, metadata)

        for page_index, page in enumerate(document):

            # Extract text in reading order
            raw_text = page.get_text("text", sort=True)

            if not raw_text.strip():
                continue

            cleaned = clean_text(raw_text)

            if not cleaned:
                continue

            record = {
                "book_id": pdf_path.stem,
                "book_title": book_title,
                "source_file": pdf_path.name,
                "pdf_page": page_index + 1,
                "text": cleaned,
                "raw_text": raw_text,
            }

            records.append(record)

    logging.info(
        "Extracted %d pages from %s",
        len(records),
        pdf_path.name
    )

    return records


# --------------------------------------------------
# Corpus creation
# --------------------------------------------------

def build_corpus() -> None:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(BOOKS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {BOOKS_DIR}"
        )

    logging.info("Found %d PDF files.", len(pdf_files))

    total_pages = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as output:

        for pdf_path in pdf_files:

            try:
                records = extract_pdf(pdf_path)

                for record in records:
                    output.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )

                total_pages += len(records)

            except Exception:
                logging.exception(
                    "Failed to process %s",
                    pdf_path.name
                )

    logging.info("Extraction completed.")
    logging.info("Total extracted pages: %d", total_pages)
    logging.info("Output saved to: %s", OUTPUT_FILE)


if __name__ == "__main__":
    build_corpus()
