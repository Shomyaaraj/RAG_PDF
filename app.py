# app.py
import os
from io import BytesIO

import faiss
import google.generativeai as genai
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()
genai.configure(
    api_key=st.secrets["GEMINI_API_KEY"]
)
llm = genai.GenerativeModel("gemini-2.5-flash")

st.set_page_config(page_title="RAG PDF Chatbot", page_icon="📄")
st.title("📄 RAG PDF Chatbot")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    pdf = BytesIO(uploaded_file.getvalue())
    reader = PdfReader(pdf)

    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"

    if not text.strip():
        st.error("No extractable text found.")
        st.stop()

    chunk_size = 1000
    overlap = 200
    chunks = []

    for i in range(0, len(text), chunk_size-overlap):
        chunks.append(text[i:i+chunk_size])

    st.success(f"Chunks: {len(chunks)}")

    with st.spinner("Loading embedding model..."):
        embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    embeddings = embedder.encode(
        chunks,
        convert_to_numpy=True
    ).astype(np.float32)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    query = st.text_input("Ask a question")

    if query:
        q = embedder.encode(
            [query],
            convert_to_numpy=True
        ).astype(np.float32)

        distances, ids = index.search(q, 5)

        context = "\n\n".join(
            [chunks[i] for i in ids[0]]
        )

        prompt = f"""
Answer ONLY from the given context.

If answer is unavailable, reply:
'I could not find this information in the PDF.'

Context:
{context}

Question:
{query}

Answer:
"""

        with st.spinner("Generating answer..."):
            response = llm.generate_content(prompt)

        st.subheader("Answer")
        st.write(response.text)

        with st.expander("Retrieved Context"):
            st.write(context)
