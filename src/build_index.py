
import json
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

CHUNKS_FILE = ROOT_DIR / "data" / "processed" / "chunks.jsonl"
INDEX_DIR = ROOT_DIR / "data" / "index"

INDEX_FILE = INDEX_DIR / "faiss.index"
MODEL_NAME = "intfloat/multilingual-e5-small"

BATCH_SIZE = 32


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)


# --------------------------------------------------
# Load chunks
# --------------------------------------------------

def load_chunks():

    chunks = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            chunks.append(json.loads(line))

    logging.info("Loaded %d chunks.", len(chunks))

    return chunks


# --------------------------------------------------
# Generate embeddings
# --------------------------------------------------

def generate_embeddings(chunks):

    logging.info("Loading embedding model: %s", MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME)

    # The model supports up to 512 tokens.
    model.max_seq_length = 512

    texts = [
        "passage: " + chunk["text"]
        for chunk in chunks
    ]

    logging.info("Generating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    logging.info("Embedding shape: %s", embeddings.shape)

    return embeddings


# --------------------------------------------------
# Build FAISS index
# --------------------------------------------------

def build_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    # Inner product on normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    logging.info("FAISS index created.")
    logging.info("Number of vectors: %d", index.ntotal)
    logging.info("Vector dimension: %d", dimension)

    return index


# --------------------------------------------------
# Main pipeline
# --------------------------------------------------

def main():

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks()

    embeddings = generate_embeddings(chunks)

    index = build_faiss_index(embeddings)

    faiss.write_index(index, str(INDEX_FILE))

    logging.info("Index saved to: %s", INDEX_FILE)
    logging.info("Embedding pipeline completed.")


if __name__ == "__main__":
    main()
