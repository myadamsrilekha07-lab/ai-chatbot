import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import ollama


st.set_page_config(
    page_title="Mini RAG",
    page_icon="📒"
)

st.title("📒 Mini RAG: Document Store + Retrieval")
st.caption("PDF → Chunks → Embeddings → ChromaDB → Retrieval → Ollama")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_embedding_model()


client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="documents"
)

st.sidebar.header("Settings")

ollama_model = st.sidebar.text_input(
    "Ollama model",
    "llama3.2"
)

chunk_size = st.sidebar.slider(
    "Chunk size",
    min_value=200,
    max_value=1500,
    value=500,
    step=100
)

top_k = st.sidebar.slider(
    "Chunks to retrieve",
    min_value=1,
    max_value=5,
    value=3
)


st.header("📄 Build Document Store")

uploaded_file = st.file_uploader(
    "Upload a text-based PDF",
    type=["pdf"]
)


if uploaded_file and st.button("Process & Store PDF"):

    # Read PDF
    reader = PdfReader(uploaded_file)

    text = ""

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    # Check if text was extracted
    if not text.strip():
        st.error(
            "No readable text was found. "
            "Please upload a text-based PDF."
        )
        st.stop()

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunk = text[i:i + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

    st.info(f"Created {len(chunks)} chunks.")
    with st.spinner("Generating embeddings..."):

        embeddings = model.encode(chunks)
    existing = collection.get()

    if existing["ids"]:
        collection.delete(
            ids=existing["ids"]
        )

    collection.add(
        ids=[
            f"chunk_{i}"
            for i in range(len(chunks))
        ],
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "source": uploaded_file.name,
                "chunk": i
            }
            for i in range(len(chunks))
        ]
    )

    st.success(
        f"Stored {len(chunks)} chunks in ChromaDB."
    )

    with st.expander("Preview stored chunks"):

        for i, chunk in enumerate(chunks[:5]):

            st.write(f"**Chunk {i + 1}:**")
            st.write(chunk)
            st.divider()


st.header("💬 Ask Questions")

question = st.text_input(
    "Ask a question about your uploaded document"
)


if st.button("🔍 Retrieve & Answer"):

    # Check question
    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()

    # Check document store
    count = collection.count()

    if count == 0:

        st.warning(
            "Please upload and process a PDF first."
        )

        st.stop()

    with st.spinner("Searching document..."):

        question_embedding = model.encode(
            [question]
        )

        # Convert numpy array to list
        question_embedding = question_embedding[0].tolist()

        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=min(top_k, count)
        )

        retrieved_chunks = results["documents"][0]

    st.subheader("🔍 Retrieved Chunks")

    for i, chunk in enumerate(retrieved_chunks):

        with st.expander(
            f"Retrieved Chunk {i + 1}"
        ):
            st.write(chunk)

    context = "\n\n".join(
        retrieved_chunks
    )

    prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question using ONLY the context provided below.

If the answer is not present in the context, say exactly:

"I could not find the answer in the uploaded document."

Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
"""

    with st.spinner(
        f"Generating answer with {ollama_model}..."
    ):

        try:

            response = ollama.chat(
                model=ollama_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            answer = response["message"]["content"]

            st.subheader("🤖 Answer")
            st.write(answer)

        except Exception as e:

            st.error(
                f"Could not connect to Ollama.\n\n"
                f"Make sure Ollama is running and the model "
                f"'{ollama_model}' is installed.\n\n"
                f"Error: {e}"
            )