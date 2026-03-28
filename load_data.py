import json
import os
from collections import defaultdict
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# 📁 Base directory
DATA_DIR = "data"
TRACK_FILE = "uploaded_files.json"


# ---------------- TRACKING ---------------- #
def load_uploaded_files():
    if os.path.exists(TRACK_FILE):
        with open(TRACK_FILE, "r") as f:
            return set(json.load(f))
    return set()


def save_uploaded_files(files_set):
    with open(TRACK_FILE, "w") as f:
        json.dump(list(files_set), f, indent=2)


# ---------------- LOADERS ---------------- #
def load_json(path, metadata):
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        text = f"Question: {item.get('instruction','')}\nAnswer: {item.get('output','')}"
        docs.append(Document(page_content=text, metadata=metadata))

    return docs


def load_jsonl(path, metadata):
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            text = f"Question: {item.get('instruction','')}\nAnswer: {item.get('output','')}"
            docs.append(Document(page_content=text, metadata=metadata))

    return docs


def load_txt(path, metadata):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    return [Document(page_content=text, metadata=metadata)]


# ---------------- MAIN LOADER ---------------- #
def load_new_documents(uploaded_files):
    all_docs = []
    new_files = []

    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:

            file_path = os.path.join(root, file)

            # Skip already uploaded files
            if file_path in uploaded_files:
                continue

            metadata = {
                "source": file,
                "folder": os.path.basename(root).lower()  # 🔥 normalize folder
            }

            try:
                if file.endswith(".json"):
                    docs = load_json(file_path, metadata)

                elif file.endswith(".jsonl"):
                    docs = load_jsonl(file_path, metadata)

                elif file.endswith(".txt"):
                    docs = load_txt(file_path, metadata)

                else:
                    print(f"Skipping unsupported file: {file}")
                    continue

                all_docs.extend(docs)
                new_files.append(file_path)

            except Exception as e:
                print(f"Error loading {file}: {e}")

    return all_docs, new_files


# ---------------- MAIN ---------------- #
if __name__ == "__main__":

    print("🔄 Checking for new files...\n")

    uploaded_files = load_uploaded_files()
    new_docs, new_files = load_new_documents(uploaded_files)

    if not new_docs:
        print("✅ No new files to upload.")
        exit()

    print(f"📄 New documents: {len(new_docs)}")

    # 🔹 Embeddings
    print("\n🔄 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("✅ Embedding model ready.")

    # 🔹 Pinecone
    load_dotenv()
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    index_name = "iks-rag-v2"

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    print("✅ Pinecone index ready.")

    # 🔥 GROUP DOCUMENTS BY FOLDER (KEY CHANGE)
    folder_docs = defaultdict(list)

    for doc in new_docs:
        folder = doc.metadata["folder"]
        folder_docs[folder].append(doc)

    print("\n⬆️ Uploading data by namespace...\n")

    for folder, docs in folder_docs.items():

        print(f"📂 Processing folder: {folder}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100
        )

        chunked_docs = splitter.split_documents(docs)

        vectorstore = PineconeVectorStore(
            index=pc.Index(index_name),
            embedding=embeddings,
            namespace=folder   # 🔥 CORE FEATURE
        )

        vectorstore.add_documents(chunked_docs)

        print(f"✅ Uploaded {len(chunked_docs)} chunks to namespace: {folder}")

    # ✅ Update tracking
    uploaded_files.update(new_files)
    save_uploaded_files(uploaded_files)

    print("\n🎉 New data upload complete!")