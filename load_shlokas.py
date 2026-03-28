import json
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# ---------------- CONFIG ----------------
JSON_FILE = "shlokas.json"   # your file name
INDEX_NAME = "iks-shlokas"   # separate index for shlokas

# ---------------- LOAD JSON ----------------
def load_shlokas():
    docs = []

    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:

        kanda = entry.get("kanda_name", "")
        sarga = entry.get("sarga_title", "")

        shlokas = entry.get("shlokas_dv", [])
        translations = entry.get("translations_en", [])

        for i, shloka in enumerate(shlokas):

            translation = translations[i] if i < len(translations) else ""

            # 🔥 Combine for better semantic search
            text = f"""
Shloka: {shloka}

Meaning: {translation}
"""

            metadata = {
                "type": "shloka",
                "kanda": kanda,
                "sarga": sarga,
                "index": i
            }

            docs.append(Document(
                page_content=text.strip(),
                metadata=metadata
            ))

    return docs


# ---------------- MAIN ----------------
if __name__ == "__main__":

    print("📖 Loading shlokas...")
    documents = load_shlokas()

    print(f"✅ Total shlokas: {len(documents)}")

    # ---------------- CHUNKING ----------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

    print(f"✂️ Total chunks: {len(docs)}")

    # ---------------- EMBEDDINGS ----------------
    print("🔄 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ---------------- PINECONE ----------------
    load_dotenv()
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    print("✅ Pinecone index ready")

    vectorstore = PineconeVectorStore(
        index=pc.Index(INDEX_NAME),
        embedding=embeddings
    )

    print("⬆️ Uploading shlokas...")
    vectorstore.add_documents(docs)

    print("🎉 Shloka embedding complete!")