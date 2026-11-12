
import requests


class OllamaGenerator:
    def __init__(
        self,
        model="qwen2.5:1.5b",
        base_url="http://localhost:11434",
    ):
        self.model = model
        self.url = f"{base_url}/api/chat"

    def generate(self, query, retrieved_chunks):

        if not retrieved_chunks:
            return "No relevant passages were retrieved."

        context_parts = []

        for i, chunk in enumerate(retrieved_chunks, start=1):
            book = chunk.get("book_title", "Unknown book")
            page = chunk.get("pdf_page", "Unknown")
            passage = chunk.get("text", "")

            context_parts.append(
                f"[S{i}] {book}, PDF page {page}\n{passage}"
            )

        context = "\n\n".join(context_parts)

        system_prompt = """
You are a rigorous scientific assistant specializing in
mathematics, statistics, machine learning and deep learning.

Your task is to answer questions using the supplied book excerpts.

STRICT RULES:

1. Use only information supported by the excerpts.
2. Never invent mathematical formulas or claim that a formula
   appears in a book unless it is present in the excerpts.
3. Check mathematical expressions carefully before presenting them.
4. If information is insufficient, explicitly acknowledge it.
5. Distinguish mathematical facts from interpretations.
6. Cite relevant claims using [S1], [S2], etc.
7. Only use source identifiers that actually exist in the context.
8. Do not invent references.
9. Avoid repetition and unnecessary introductions.
10. Give a focused answer, normally between 200 and 350 words.
11. Use the same language as the user's question.
12. If uncertain about a derivation, explain the limitation
    rather than presenting an unsupported derivation.

For comparison questions, use a concise structured comparison.
For mathematical questions, define the variables and explain
the equations.
"""

        user_prompt = f"""
BOOK EXCERPTS:

{context}

QUESTION:

{query}

Write a rigorous and concise answer with source citations.
"""

        try:
            response = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_ctx": 4096,
                        "num_predict": 900,
                    },
                },
                timeout=240,
            )

            response.raise_for_status()

            result = response.json()

            return result["message"]["content"].strip()

        except requests.exceptions.ConnectionError:
            return "Error: Ollama is not running."

        except requests.exceptions.Timeout:
            return "Error: Generation timed out."

        except requests.exceptions.RequestException as error:
            return f"Ollama API error: {error}"
