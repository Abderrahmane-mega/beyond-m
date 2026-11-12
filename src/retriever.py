
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

CHUNKS_FILE = ROOT_DIR / "data" / "processed" / "chunks.jsonl"
INDEX_FILE = ROOT_DIR / "data" / "index" / "faiss.index"

MODEL_NAME = "intfloat/multilingual-e5-small"


# --------------------------------------------------
# Retriever
# --------------------------------------------------

class Retriever:

    def __init__(self):

        print("Loading chunks...")

        with CHUNKS_FILE.open("r", encoding="utf-8") as file:
            self.chunks = [
                json.loads(line)
                for line in file
            ]

        print("Loading FAISS index...")

        self.index = faiss.read_index(str(INDEX_FILE))

        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                f"Index contains {self.index.ntotal} vectors "
                f"but corpus contains {len(self.chunks)} chunks."
            )

        print("Loading embedding model...")

        self.model = SentenceTransformer(MODEL_NAME)

        print("Retriever ready.")

    def search(self, query: str, top_k: int = 5):

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be positive.")

        # E5 requires the query prefix
        query_text = "query: " + query

        query_embedding = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, self.index.ntotal)
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx < 0:
                continue

            chunk = self.chunks[int(idx)]

            results.append({
                "score": float(score),
                "chunk_id": chunk["chunk_id"],
                "book_title": chunk["book_title"],
                "source_file": chunk["source_file"],
                "pdf_page": chunk["pdf_page"],
                "text": chunk["text"]
            })

        return results


# --------------------------------------------------
# Interactive test
# --------------------------------------------------

if __name__ == "__main__":

    retriever = Retriever()

    print("\nEnter your question (type 'exit' to quit).")

    while True:

        query = input("\nQuestion: ").strip()

        if query.lower() == "exit":
            break

        results = retriever.search(query, top_k=5)

        for i, result in enumerate(results, start=1):

            print("\n" + "=" * 70)
            print(f"RESULT {i}")
            print("=" * 70)

            print("Similarity:", round(result["score"], 4))
            print("Book:", result["book_title"])
            print("PDF page:", result["pdf_page"])
            print("Chunk:", result["chunk_id"])

            print("\nContent:\n")
            print(result["text"])
