import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import re
import asyncio
import streamlit as st
import edge_tts
import numpy as np
import sounddevice as sd
import unicodedata
import uuid
import threading
import time
import json
import base64
import json
import os
import re
from serpapi import GoogleSearch
import markdown
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="VedaMitra",
    page_icon="🕉️",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=EB+Garamond:ital,wght@0,400;0,500;1,400&display=swap');

/* ── Root palette ── */
:root {
    --saffron:       #E8710A;
    --saffron-glow:  #F9A825;
    --deep-indigo:   #0D0A1E;
    --indigo-mid:    #150F2E;
    --indigo-card:   #1A1338;
    --indigo-border: #2D2060;
    --gold:          #C9963A;
    --gold-light:    #F0C060;
    --cream:         #F5EDD6;
    --muted:         #A89BC2;
    --lotus-pink:    #D4608A;
}

/* ── Global background & text ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--deep-indigo) !important;
    color: var(--cream) !important;
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 17px;
}

[data-testid="stHeader"] { background: transparent !important; }

/* Sidebar & main block */
[data-testid="block-container"] {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* ── Mandala / decorative top strip ── */
.vedamitra-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    position: relative;
}

.vedamitra-header::before {
    content: '';
    display: block;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold), var(--saffron), var(--gold), transparent);
    margin-bottom: 1.5rem;
    border-radius: 1px;
}

.vedamitra-header::after {
    content: '';
    display: block;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--indigo-border), transparent);
    margin-top: 1.5rem;
}

/* ── Main title ── */
.main-title {
    font-family: 'Cinzel', serif;
    font-size: 3rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: var(--gold-light);
    text-shadow:
        0 0 30px rgba(201, 150, 58, 0.5),
        0 2px 4px rgba(0,0,0,0.6);
    margin: 0;
    line-height: 1.1;
}

.main-title .om {
    color: var(--saffron);
    margin-right: 0.2em;
    text-shadow: 0 0 20px rgba(232, 113, 10, 0.7);
}

/* ── Subtitle ── */
.sub-title {
    font-family: 'EB Garamond', serif;
    font-style: italic;
    font-size: 1.05rem;
    color: var(--muted);
    letter-spacing: 0.04em;
    margin-top: 0.5rem;
}

/* ── Section heading ── */
.domain-heading {
    font-family: 'Cinzel', serif;
    font-size: 0.8rem;
    letter-spacing: 0.25em;
    color: var(--saffron);
    text-transform: uppercase;
    margin: 2rem 0 0.8rem;
}

/* ── Category buttons ── */
div[data-testid="column"] button {
    background: linear-gradient(135deg, var(--indigo-card), #120D28) !important;
    border: 1px solid var(--indigo-border) !important;
    border-radius: 10px !important;
    color: var(--cream) !important;
    font-family: 'Cinzel', serif !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    padding: 0.7rem 0.4rem !important;
    transition: all 0.25s ease !important;
    position: relative;
    overflow: hidden;
}

div[data-testid="column"] button::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(201,150,58,0.07), transparent);
    opacity: 0;
    transition: opacity 0.25s;
}

div[data-testid="column"] button:hover {
    border-color: var(--gold) !important;
    color: var(--gold-light) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(201,150,58,0.18) !important;
}

div[data-testid="column"] button:hover::before { opacity: 1; }

/* ── Success / Warning banners ── */
[data-testid="stAlert"] {
    background: rgba(201, 150, 58, 0.08) !important;
    border: 1px solid rgba(201, 150, 58, 0.3) !important;
    border-radius: 10px !important;
    color: var(--gold-light) !important;
    font-family: 'EB Garamond', serif !important;
    font-size: 1rem !important;
}

/* ── Text input ── */
[data-testid="stTextInput"] input {
    background: var(--indigo-card) !important;
    border: 1px solid var(--indigo-border) !important;
    border-radius: 12px !important;
    color: var(--cream) !important;
    font-family: 'EB Garamond', serif !important;
    font-size: 1rem !important;
    padding: 0.65rem 1rem !important;
    transition: border-color 0.2s;
}

[data-testid="stTextInput"] input:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px rgba(201, 150, 58, 0.12) !important;
}

[data-testid="stTextInput"] input::placeholder {
    color: var(--muted) !important;
}

/* ── Mic button ── */
div[data-testid="column"]:last-child button {
    background: linear-gradient(135deg, #2A1F0A, #1A1300) !important;
    border: 1px solid var(--gold) !important;
    border-radius: 10px !important;
    color: var(--gold-light) !important;
    font-size: 1.1rem !important;
    padding: 0.6rem !important;
    transition: all 0.2s !important;
}

div[data-testid="column"]:last-child button:hover {
    background: linear-gradient(135deg, #3A2910, #251A00) !important;
    box-shadow: 0 4px 16px rgba(201, 150, 58, 0.3) !important;
    transform: scale(1.05) !important;
}

/* ── Chat bubbles ── */
.user-bubble {
    background: linear-gradient(135deg, #1F1740, #18103A);
    border: 1px solid var(--indigo-border);
    border-radius: 16px 16px 4px 16px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0 1rem;
    color: var(--cream);
    font-family: 'EB Garamond', serif;
    font-size: 1rem;
    position: relative;
}

.user-bubble .bubble-label {
    font-family: 'Cinzel', serif;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: var(--muted);
    text-transform: uppercase;
    display: block;
    margin-bottom: 0.3rem;
}

.bot-bubble {
    background: linear-gradient(135deg, #0F0B22, #130E28);
    border: 1px solid var(--indigo-border);
    border-left: 3px solid var(--saffron);
    border-radius: 4px 16px 16px 16px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1.5rem;
    color: var(--cream);
    font-family: 'EB Garamond', serif;
    font-size: 1.05rem;
    line-height: 1.75;
    position: relative;
}

.bot-bubble .bubble-label {
    font-family: 'Cinzel', serif;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: var(--saffron);
    text-transform: uppercase;
    display: block;
    margin-bottom: 0.5rem;
}

/* Markdown within bot bubble */
.bot-bubble h1, .bot-bubble h2, .bot-bubble h3 {
    font-family: 'Cinzel', serif;
    color: var(--gold-light);
    margin-top: 1rem;
}

.bot-bubble strong { color: var(--gold-light); }
.bot-bubble em { color: var(--lotus-pink); }

.bot-bubble ul, .bot-bubble ol {
    padding-left: 1.4rem;
}

.bot-bubble li { margin-bottom: 0.3rem; }

/* ── Shloka box ── */
.shloka-box {
    background: rgba(201, 150, 58, 0.06);
    border: 1px solid rgba(201, 150, 58, 0.25);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-top: 1rem;
    font-style: italic;
    color: var(--gold-light);
    font-size: 1rem;
    line-height: 1.8;
}

.shloka-box .shloka-title {
    font-family: 'Cinzel', serif;
    font-style: normal;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: var(--saffron);
    text-transform: uppercase;
    display: block;
    margin-bottom: 0.5rem;
}

/* ── Divider ── */
hr {
    border: none;
    border-top: 1px solid var(--indigo-border);
    margin: 1.5rem 0;
}

/* ── Image display ── */
[data-testid="stImage"] img {
    border-radius: 14px;
    border: 1px solid var(--indigo-border);
    box-shadow: 0 8px 30px rgba(0,0,0,0.5);
    margin: 0.5rem 0 1rem;
}

/* ── Audio player ── */
[data-testid="stAudio"] audio {
    filter: invert(0.85) hue-rotate(180deg) saturate(0.6);
    border-radius: 30px;
    width: 100%;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--deep-indigo); }
::-webkit-scrollbar-thumb {
    background: var(--indigo-border);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: var(--gold); }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--saffron) !important; }

/* ── Info boxes ── */
.stInfo {
    background: rgba(201, 150, 58, 0.06) !important;
    border: 1px solid rgba(201, 150, 58, 0.2) !important;
    color: var(--muted) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div class="vedamitra-header">
    <h1 class="main-title"><span class="om">ॐ</span> VedaMitra</h1>
    <p class="sub-title">Voice &amp; RAG-powered Indian Knowledge System</p>
</div>
""", unsafe_allow_html=True)

# ---------------- STARTUP AUDIO ----------------
if "startup_audio_played" not in st.session_state:
    st.session_state.startup_audio_played = True
    with open("data/intro.mp3", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <audio autoplay>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """, unsafe_allow_html=True)

# ---------------- ENV ----------------
load_dotenv()

# ---------------- CATEGORY ----------------
CATEGORY_NAMESPACE_MAP = {
    "🧘 Yoga":        "yoga",
    "🌿 Ayurveda":    "ayurveda",
    "🏹 Ramayana":    "ramayana",
    "⚔️ Mahabharata": "mahabharata",
    "📜 General":     "general",
}

if "selected_category" not in st.session_state:
    st.session_state.selected_category = None

st.markdown('<p class="domain-heading">✦ Select Knowledge Domain ✦</p>', unsafe_allow_html=True)

cols = st.columns(5)
for col, (label, namespace) in zip(cols, CATEGORY_NAMESPACE_MAP.items()):
    with col:
        if st.button(label, use_container_width=True):
            st.session_state.selected_category = namespace

if st.session_state.selected_category:
    display_name = {v: k for k, v in CATEGORY_NAMESPACE_MAP.items()}[st.session_state.selected_category]
    st.success(f"Domain: {display_name}")
else:
    st.warning("Please select a knowledge domain to continue.")
    st.stop()

# ---------------- IMAGE MAP ----------------
@st.cache_resource


# ---------------- LOAD LOCAL IMAGE MAP ----------------
def load_image_map():
    with open("data/image_map.json", "r") as f:
        return json.load(f)

image_map = load_image_map()


# ---------------- SERPAPI IMAGE SEARCH ----------------
def fetch_images(query, num_images=3):
    try:
        params = {
            "engine": "google_images",
            "q": query,
            "api_key": os.getenv("SERPAPI_KEY"),
            "num": num_images
        }

        results = GoogleSearch(params).get_dict()

        return [img["original"] for img in results.get("images_results", [])[:num_images]]

    except Exception:
        return []


# ---------------- QUERY OPTIMIZATION ----------------
def extract_search_query(query):
    query = query.lower()

    if "yoga" in query:
        return query + " yoga pose"
    elif "who" in query or "sita" in query or "rama" in query:
        return query + " indian mythology character"
    else:
        return query + " india"


# ---------------- FINAL HYBRID FUNCTION ----------------
def fetch_image(query):
    cleaned_query = re.sub(r'[^\w\s]', '', query.lower())

    # 🔍 1. Check local dataset first
    for key, path in image_map.items():
        if key in cleaned_query:
            return [os.path.join("data", path)]  # return as list

    # 🌐 2. Fallback to SerpAPI
    search_query = extract_search_query(cleaned_query)
    return fetch_images(search_query, num_images=3)

# ---------------- TEXT CLEAN ----------------
def remove_diacritics(text):
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))

def clean_text(text):
    text = remove_diacritics(text)
    text = re.sub(r"- ", "", text)
    text = re.sub(r"\n", " ", text)
    return text.strip()

# ---------------- FORMAT ----------------
def format_response(text):
    text = re.sub(r"(###)", r"\n\n\1", text)
    text = text.replace("\n", "\n\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return markdown.markdown(text)

# ---------------- TTS ----------------
async def generate_audio(text):
    filename = f"temp_{uuid.uuid4().hex}.mp3"
    communicate = edge_tts.Communicate(
        clean_text(text),
        voice="hi-IN-SwaraNeural",
        rate="+20%"
    )
    await communicate.save(filename)
    return filename

# ---------------- WHISPER ----------------
@st.cache_resource
def load_whisper():
    return WhisperModel("tiny", device="cpu", compute_type="int8")

whisper_model = load_whisper()

def record_audio():
    st.info("🎙️ Listening for 6 seconds…")
    audio = sd.rec(int(6 * 16000), samplerate=16000, channels=1)
    sd.wait()
    return audio.flatten()

def transcribe(audio):
    segments, _ = whisper_model.transcribe(audio, beam_size=3)
    return " ".join([seg.text for seg in segments]).strip()

# ---------------- RAG ----------------
@st.cache_resource
def load_rag(namespace):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    vectorstore = PineconeVectorStore(
        index=pc.Index("iks-rag-v2"),
        embedding=embeddings,
        namespace=namespace
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    llm = ChatMistralAI(
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        model="mistral-large-latest",
        temperature=0.3
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )

qa_chain = load_rag(st.session_state.selected_category)

# ---------------- SHLOKA ----------------
@st.cache_resource
def load_shloka_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    vectorstore = PineconeVectorStore(
        index=pc.Index("iks-shlokas"),
        embedding=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 1})

shloka_retriever = load_shloka_retriever()

def clean_shloka(text):
    text = re.sub(r"।।.*?।।", "", text)
    text = text.replace("\\n", "\n")
    return text.strip()

def get_relevant_shloka(query):
    docs = shloka_retriever.invoke(query)
    if not docs:
        return ""
    shloka = clean_shloka(docs[0].page_content)
    return f"""
<div class="shloka-box">
    <span class="shloka-title">✦ Shloka ✦</span>
    {shloka}
</div>
"""

# ---------------- DIVIDER ----------------
st.markdown("<hr>", unsafe_allow_html=True)

# ---------------- INPUT ----------------
col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input(
        label="question",
        placeholder="Ask about yoga, dharma, healing, or ancient wisdom…",
        label_visibility="collapsed"
    )

with col2:
    voice_clicked = st.button("🎙️", use_container_width=True, help="Speak your question")
    if voice_clicked:
        audio = record_audio()
        user_input = transcribe(audio)

# ---------------- RESPONSE ----------------
if user_input:

    st.markdown(
        f'<div class="user-bubble">'
        f'<span class="bubble-label">You</span>'
        f'{user_input}'
        f'</div>',
        unsafe_allow_html=True
    )

    with st.spinner("Consulting the ancient texts…"):
        result = qa_chain.invoke({"question": user_input})
        answer = result["answer"]

    images = fetch_image(user_input)

    if images:

        cols = st.columns(len(images))
        for i, img in enumerate(images):
            with cols[i]:
                if isinstance(img, str) and os.path.exists(img):
                    st.image(img, use_container_width=True)
                else:
                    st.image(img, use_container_width=True)

    placeholder = st.empty()
    temp = ""

    for word in answer.split():
        temp += word + " "
        placeholder.markdown(
            f'<div class="bot-bubble">'
            f'<span class="bubble-label">VedaMitra</span>'
            f'{format_response(temp)}'
            f'</div>',
            unsafe_allow_html=True
        )
        time.sleep(0.04)

    audio_file = asyncio.run(generate_audio(answer))
    st.audio(audio_file)