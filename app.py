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
import time
import json
import base64
import re
from serpapi import GoogleSearch
import markdown
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from age_detection import detect_age, get_age_group

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="VedaMitra",
    page_icon="🕉️",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=EB+Garamond:ital,wght@0,400;0,500;1,400&display=swap');

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

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--deep-indigo) !important;
    color: var(--cream) !important;
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 17px;
}

[data-testid="stHeader"] { background: transparent !important; }

[data-testid="block-container"] {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

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

.main-title {
    font-family: 'Cinzel', serif;
    font-size: 3rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: var(--gold-light);
    text-shadow: 0 0 30px rgba(201, 150, 58, 0.5), 0 2px 4px rgba(0,0,0,0.6);
    margin: 0;
    line-height: 1.1;
}

.main-title .om {
    color: var(--saffron);
    margin-right: 0.2em;
    text-shadow: 0 0 20px rgba(232, 113, 10, 0.7);
}

.sub-title {
    font-family: 'EB Garamond', serif;
    font-style: italic;
    font-size: 1.05rem;
    color: var(--muted);
    letter-spacing: 0.04em;
    margin-top: 0.5rem;
}

.age-badge {
    display: inline-block;
    background: rgba(201, 150, 58, 0.1);
    border: 1px solid rgba(201, 150, 58, 0.3);
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-family: 'Cinzel', serif;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    color: var(--gold-light);
    margin-top: 0.6rem;
}

.domain-heading {
    font-family: 'Cinzel', serif;
    font-size: 0.8rem;
    letter-spacing: 0.25em;
    color: var(--saffron);
    text-transform: uppercase;
    margin: 2rem 0 0.8rem;
}

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

[data-testid="stAlert"] {
    background: rgba(201, 150, 58, 0.08) !important;
    border: 1px solid rgba(201, 150, 58, 0.3) !important;
    border-radius: 10px !important;
    color: var(--gold-light) !important;
    font-family: 'EB Garamond', serif !important;
    font-size: 1rem !important;
}

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

[data-testid="stTextInput"] input::placeholder { color: var(--muted) !important; }

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

.user-bubble {
    background: linear-gradient(135deg, #1F1740, #18103A);
    border: 1px solid var(--indigo-border);
    border-radius: 16px 16px 4px 16px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0 1rem;
    color: var(--cream);
    font-family: 'EB Garamond', serif;
    font-size: 1rem;
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

.bot-bubble h1, .bot-bubble h2, .bot-bubble h3 {
    font-family: 'Cinzel', serif;
    color: var(--gold-light);
    margin-top: 1rem;
}

.bot-bubble strong { color: var(--gold-light); }
.bot-bubble em { color: var(--lotus-pink); }
.bot-bubble ul, .bot-bubble ol { padding-left: 1.4rem; }
.bot-bubble li { margin-bottom: 0.3rem; }

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

hr { border: none; border-top: 1px solid var(--indigo-border); margin: 1.5rem 0; }

[data-testid="stImage"] img {
    border-radius: 14px;
    border: 1px solid var(--indigo-border);
    box-shadow: 0 8px 30px rgba(0,0,0,0.5);
    margin: 0.5rem 0 1rem;
}

[data-testid="stAudio"] audio {
    filter: invert(0.85) hue-rotate(180deg) saturate(0.6);
    border-radius: 30px;
    width: 100%;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--deep-indigo); }
::-webkit-scrollbar-thumb { background: var(--indigo-border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--gold); }

[data-testid="stSpinner"] { color: var(--saffron) !important; }

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

# ---------------- AGE DETECTION (once per session) ----------------
if "age_group" not in st.session_state:
    with st.spinner("📷 Detecting viewer profile…"):
        _age = detect_age()
        st.session_state.age_group    = get_age_group(_age)
        st.session_state.detected_age = _age

# ── Age badge below header ──
_AGE_BADGE_ICONS = {"child": "🧒", "teen": "🧑", "adult": "👤", "senior": "🧓", "unknown": "👤"}
_age_icon  = _AGE_BADGE_ICONS.get(st.session_state.age_group, "👤")
_age_label = (
    f"~{st.session_state.detected_age} yrs · {st.session_state.age_group.capitalize()}"
    if st.session_state.detected_age else "Profile: Unknown"
)
st.markdown(
    f'<div style="text-align:center"><span class="age-badge">{_age_icon} {_age_label}</span></div>',
    unsafe_allow_html=True
)

# ---------------- AGE-AWARE SYSTEM PROMPTS ----------------
AGE_SYSTEM_PROMPTS = {
    "child": """You are VedaMitra, a warm and friendly guide explaining ancient Indian wisdom to a young child under 13 years old.
- Use very simple words that a child can easily understand. Avoid any difficult or complex terms.
- Use fun comparisons, short stories and relatable examples. For example: "Think of Dharma like the rules of a game that help everyone be fair and kind."
- Keep your sentences very short. Be warm, playful and encouraging.
- If you must use a Sanskrit word, immediately explain it in the simplest possible way right after.
- Keep your answer to 3 to 5 short sentences maximum. Do not overwhelm the child with too much information at once.""",

    "teen": """You are VedaMitra, explaining ancient Indian wisdom to a teenager between 13 and 19 years old.
- Use clear, modern and friendly language. Avoid being preachy, overly formal or boring.
- Make the answer relatable to everyday teenage life wherever naturally possible.
- You can use some Sanskrit terms but always explain them simply right after using them.
- Use bullet points when it makes things clearer and easier to read.
- Give a focused, medium-length answer. Detailed enough to be genuinely useful but not so long it loses their attention.""",

    "adult": """You are VedaMitra, a knowledgeable guide on ancient Indian wisdom speaking to an adult.
- Use proper vocabulary. Sanskrit terms are welcome with brief contextual explanations where needed.
- Provide depth, nuance and real context. Reference specific scriptures, texts or philosophical schools where relevant.
- Be precise, informative and respectful in tone.
- Structure the answer clearly. You may use headings or bullet points for complex multi-part topics.
- Give a thorough, well-rounded and intellectually satisfying answer.""",

    "senior": """You are VedaMitra, respectfully sharing ancient Indian wisdom with a senior person above 50 years old.
- Use clear, warm and deeply respectful language. Speak with reverence for their life experience and wisdom.
- Avoid unnecessary jargon. When using Sanskrit terms, explain them gently and naturally within the flow.
- Draw meaningful connections to traditional values, lived experience and timeless wisdom where appropriate.
- Keep the tone calm, thoughtful and unhurried. Do not make the response excessively long.
- Be compassionate, patient and dignified throughout your response.""",

    "unknown": """You are VedaMitra, a knowledgeable and friendly guide on ancient Indian wisdom.
- Use clear and accessible language suitable for a general adult audience.
- Provide balanced depth that is neither too simple nor too complex.
- Explain Sanskrit terms naturally when used. Be informative, accurate and respectful.""",
}

# ---------------- BUILD AGE-AWARE RAG CHAIN ----------------
def build_chain(retriever, llm, memory, age_group):
    """
    Builds a ConversationalRetrievalChain with the age-appropriate system prompt
    embedded directly into the LLM prompt template, so every answer Mistral
    generates is written in the right vocabulary, depth and tone for the user.
    """
    system_prompt = AGE_SYSTEM_PROMPTS.get(age_group, AGE_SYSTEM_PROMPTS["unknown"])

    prompt_template = f"""{system_prompt}

Use the following context retrieved from ancient Indian texts to answer the question.
If the context does not contain enough information, draw on your own knowledge but stay true to the spirit and accuracy of Indian philosophy and tradition.

Context:
{{context}}

Chat History:
{{chat_history}}

Question: {{question}}

Answer:"""

    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=prompt_template
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

# ---------------- LOAD RAG COMPONENTS (cached) ----------------
@st.cache_resource
def load_rag_components(namespace):
    """
    Loads retriever, LLM and memory separately and caches them.
    The chain is built outside this function so the prompt can be
    customised per age group without breaking the cache.
    """
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
    return retriever, llm, memory

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

# ── Both age and category are now known — build the age-aware chain ──
retriever, llm, memory = load_rag_components(st.session_state.selected_category)
qa_chain = build_chain(retriever, llm, memory, st.session_state.age_group)

# ---------------- LOAD LOCAL IMAGE MAP ----------------
@st.cache_resource
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

# ---------------- HYBRID IMAGE FETCH ----------------
def fetch_image(query):
    cleaned_query = re.sub(r'[^\w\s]', '', query.lower())
    for key, path in image_map.items():
        if key in cleaned_query:
            return [os.path.join("data", path)]
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
        # answer is natively age-appropriate — Mistral wrote it that way
        # because the age-aware system prompt was embedded in the chain

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

    # TTS reads the answer directly — it is already clean, no post-processing needed
    audio_file = asyncio.run(generate_audio(answer))
    st.audio(audio_file)