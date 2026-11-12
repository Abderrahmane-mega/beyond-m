
import json
from pathlib import Path

CORPUS_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "processed"
    / "pages.jsonl"
)

with CORPUS_PATH.open("r", encoding="utf-8") as file:
    first_record = json.loads(next(file))

print("Book:", first_record["book_title"])
print("Source:", first_record["source_file"])
print("PDF page:", first_record["pdf_page"])
print("\nExtracted text:\n")
print(first_record["text"][:2000])
