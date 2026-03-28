import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


# 🔹 Load environment variables
load_dotenv()

# 🔹 Embedding model (same one used for indexing)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 🔹 Connect to Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "iks-rag-v2"   # IMPORTANT: match your new index

vectorstore = PineconeVectorStore(
    index=pc.Index(index_name),
    embedding=embeddings
)

# 🔹 Retriever (reduced context)
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}   # smaller context = fewer tokens
)

# 🔹 Load Mistral (controlled output)
llm = ChatMistralAI(
    mistral_api_key=os.getenv("MISTRAL_API_KEY"),
    model="mistral-large-latest",
    max_tokens=300,        # limit response size
    temperature=0.3        # more focused answers
)

# 🔹 Custom concise prompt
prompt_template = """
Use the provided context to answer the question.
Be concise (max 5-7 sentences).
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# 🔹 RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT}
)

# 🔹 Interactive loop
while True:
    query = input("\nAsk something (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    response = qa_chain.run(query)

    print("\nAnswer:\n")
    print(response)
