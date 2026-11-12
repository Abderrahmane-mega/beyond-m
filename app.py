
import streamlit as st

from src.retriever import Retriever
from src.generator import OllamaGenerator


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Beyond ML | Theory & Practice",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# Load RAG components once
# --------------------------------------------------

@st.cache_resource
def load_components():
    retriever = Retriever()
    generator = OllamaGenerator()
    return retriever, generator


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("📚 Beyond ML: Theory & Practice")

st.markdown(
    """
    **Understand the theory. Explore the mathematics. Master the practice.**

    An interactive learning environment exploring the mathematical,
    statistical and computational foundations of Machine Learning
    and Deep Learning.

    Ask questions, retrieve relevant passages from scientific textbooks,
    and investigate the ideas behind machine learning algorithms.
    """
)

st.divider()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    top_k = st.slider(
        "Retrieved passages",
        min_value=1,
        max_value=10,
        value=5
    )

    st.divider()

    st.subheader("Knowledge base")

    st.markdown(
        """
        - **Bishop** — Pattern Recognition and Machine Learning
        - **Goodfellow et al.** — Deep Learning
        - **Géron** — Hands-On Machine Learning
        - **Murphy** — Machine Learning: A Probabilistic Perspective
        """
    )

    st.divider()

    st.subheader("Technology")

    st.caption("Embeddings: multilingual-e5-small")
    st.caption("Vector search: FAISS")
    st.caption("Local LLM: Qwen 2.5 1.5B")
    st.caption("LLM runtime: Ollama")
    
    st.divider()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------
# Initialize components
# --------------------------------------------------

try:
    retriever, generator = load_components()
    st.sidebar.success("Knowledge base loaded")
except Exception as error:
    st.error(f"Could not initialize the RAG system: {error}")
    st.info(
        "Check that the FAISS index and processed chunks exist, "
        "and that Ollama is running."
    )
    st.stop()


# --------------------------------------------------
# Conversation state
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# Display conversation history
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            sources = message.get("sources", [])

            if sources:
                with st.expander(
                    f"Inspect retrieved passages ({len(sources)})"
                ):
                    for i, chunk in enumerate(sources, start=1):

                        score = chunk.get("score")

                        st.markdown(
                            f"**[S{i}] {chunk['book_title']} "
                            f"— PDF page {chunk['pdf_page']}**"
                        )

                        if score is not None:
                            st.caption(
                                f"Cosine similarity: {score:.4f}"
                            )

                        st.text(chunk["text"])
                        st.divider()


# --------------------------------------------------
# User input and RAG pipeline
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about machine learning or deep learning..."
)

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        try:
            with st.spinner(
                "Searching the textbooks and generating an answer..."
            ):

                # Step 1: Retrieve relevant passages
                chunks = retriever.search(
                    query=question,
                    top_k=top_k
                )

                if not chunks:
                    answer = (
                        "I could not retrieve relevant passages "
                        "from the knowledge base."
                    )
                else:
                    # Step 2: Generate an answer from retrieved context
                    answer = generator.generate(
                        query=question,
                        retrieved_chunks=chunks
                    )

            st.markdown(answer)

            # Step 3: Display retrieved sources
            if chunks:
                with st.expander(
                    f"Inspect retrieved passages ({len(chunks)})"
                ):
                    for i, chunk in enumerate(chunks, start=1):

                        score = chunk.get("score")

                        st.markdown(
                            f"**[S{i}] {chunk['book_title']} "
                            f"— PDF page {chunk['pdf_page']}**"
                        )

                        if score is not None:
                            st.caption(
                                f"Cosine similarity: {score:.4f}"
                            )

                        st.text(chunk["text"])
                        st.divider()

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": chunks
            })

        except Exception as error:
            st.error(f"An error occurred: {error}")
