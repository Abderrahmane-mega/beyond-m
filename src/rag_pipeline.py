
from src.retriever import Retriever
from src.generator import OllamaGenerator


class RAGPipeline:
    def __init__(self, top_k=5):
        self.retriever = Retriever()
        self.generator = OllamaGenerator()
        self.top_k = top_k

    def answer(self, question):
        # 1. Retrieve relevant passages
        retrieved_chunks = self.retriever.search(
            query=question,
            top_k=self.top_k
        )

        # 2. Generate an answer from the retrieved context
        answer = self.generator.generate(
            query=question,
            retrieved_chunks=retrieved_chunks
        )

        # 3. Build a deterministic source list
        sources = []
        seen = set()

        for chunk in retrieved_chunks:
            key = (
                chunk["book_title"],
                chunk["pdf_page"]
            )

            if key not in seen:
                seen.add(key)
                sources.append(
                    f"- {chunk['book_title']}, "
                    f"PDF page {chunk['pdf_page']}"
                )

        # 4. Return answer and retrieved references
        source_section = "\n".join(sources)

        return (
            f"{answer}\n\n"
            f"---\n"
            f"PASSAGES RETRIEVED FROM THE BOOKS\n"
            f"{source_section}"
        )


if __name__ == "__main__":
    pipeline = RAGPipeline(top_k=5)

    print("Machine Learning Research Assistant")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Your question: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        print("\nGenerating answer...\n")

        response = pipeline.answer(question)

        print(response)
        print("\n" + "=" * 70 + "\n")
