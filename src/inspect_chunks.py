
import json
from pathlib import Path
from statistics import mean, median

ROOT_DIR = Path(__file__).resolve().parent.parent
CHUNKS_FILE = ROOT_DIR / "data" / "processed" / "chunks.jsonl"

chunks = []

with CHUNKS_FILE.open("r", encoding="utf-8") as file:
    for line in file:
        chunks.append(json.loads(line))

lengths = [len(chunk["text"]) for chunk in chunks]

print("=" * 55)
print("CORPUS STATISTICS")
print("=" * 55)

print("Total chunks:", len(chunks))
print("Average length:", round(mean(lengths), 2))
print("Median length:", median(lengths))
print("Shortest chunk:", min(lengths))
print("Longest chunk:", max(lengths))

print("\nChunks by length:")

ranges = [
    (0, 300),
    (300, 800),
    (800, 1400),
    (1400, 1800),
    (1800, float("inf")),
]

for lower, upper in ranges:
    count = sum(lower <= length < upper for length in lengths)
    print(f"{lower:4} - {upper:>7}: {count}")

# Select a representative chunk
representative = max(
    (chunk for chunk in chunks if len(chunk["text"]) >= 1000),
    key=lambda chunk: len(chunk["text"])
)

print("\n" + "=" * 55)
print("REPRESENTATIVE CHUNK")
print("=" * 55)

print("ID:", representative["chunk_id"])
print("Book:", representative["book_title"])
print("PDF page:", representative["pdf_page"])
print("Length:", len(representative["text"]))

print("\nContent:\n")
print(representative["text"])
