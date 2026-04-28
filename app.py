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
st.set_page_config(page_title="VedaMitra", page_icon="🕉️", layout="centered")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=EB+Garamond:ital,wght@0,400;0,500;1,400&display=swap');

:root {
    --saffron:       #E8710A;
    --deep-indigo:   #0D0A1E;
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
[data-testid="block-container"] { padding-top: 2rem; padding-bottom: 3rem; }

/* ── header ── */
.vedamitra-header { text-align: center; padding: 2.5rem 1rem 1.5rem; }
.vedamitra-header::before {
    content: '';
    display: block;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold), var(--saffron), var(--gold), transparent);
    margin-bottom: 1.5rem;
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
    text-shadow: 0 0 30px rgba(201,150,58,0.5), 0 2px 4px rgba(0,0,0,0.6);
    margin: 0;
}
.main-title .om { color: var(--saffron); margin-right: 0.2em; }
.sub-title {
    font-family: 'EB Garamond', serif;
    font-style: italic;
    font-size: 1.05rem;
    color: var(--muted);
    letter-spacing: 0.04em;
    margin-top: 0.5rem;
}

/* ── badges ── */
.badge-row { text-align:center; margin-top:0.7rem; display:flex; justify-content:center; gap:0.6rem; flex-wrap:wrap; }
.age-badge, .lang-badge {
    display: inline-block;
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-family: 'Cinzel', serif;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
}
.age-badge  { background:rgba(201,150,58,0.1);  border:1px solid rgba(201,150,58,0.3); color:var(--gold-light); }
.lang-badge { background:rgba(232,113,10,0.12); border:1px solid rgba(232,113,10,0.35); color:var(--saffron);    }

/* ── section headings ── */
.section-heading {
    font-family: 'Cinzel', serif;
    font-size: 0.78rem;
    letter-spacing: 0.25em;
    color: var(--saffron);
    text-transform: uppercase;
    margin: 1.8rem 0 0.8rem;
    text-align: center;
}

/* ── all column buttons share one base style ── */
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
}
div[data-testid="column"] button:hover {
    border-color: var(--gold) !important;
    color: var(--gold-light) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(201,150,58,0.18) !important;
}

/* active language button */
.lang-active button {
    background: linear-gradient(135deg, #2A1F0A, #1A1300) !important;
    border-color: var(--gold) !important;
    color: var(--gold-light) !important;
}

/* voice button (last column in input row) */
div[data-testid="column"]:last-child button {
    background: linear-gradient(135deg, #2A1F0A, #1A1300) !important;
    border: 1px solid var(--gold) !important;
    color: var(--gold-light) !important;
    font-size: 1.1rem !important;
    padding: 0.6rem !important;
}
div[data-testid="column"]:last-child button:hover {
    box-shadow: 0 4px 16px rgba(201,150,58,0.3) !important;
    transform: scale(1.05) !important;
}

/* ── alerts ── */
[data-testid="stAlert"] {
    background: rgba(201,150,58,0.08) !important;
    border: 1px solid rgba(201,150,58,0.3) !important;
    border-radius: 10px !important;
    color: var(--gold-light) !important;
    font-family: 'EB Garamond', serif !important;
}

/* ── text input ── */
[data-testid="stTextInput"] input {
    background: var(--indigo-card) !important;
    border: 1px solid var(--indigo-border) !important;
    border-radius: 12px !important;
    color: var(--cream) !important;
    font-family: 'EB Garamond', serif !important;
    font-size: 1rem !important;
    padding: 0.65rem 1rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px rgba(201,150,58,0.12) !important;
}
[data-testid="stTextInput"] input::placeholder { color: var(--muted) !important; }

/* ── chat bubbles ── */
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
.bot-bubble h1,.bot-bubble h2,.bot-bubble h3 { font-family:'Cinzel',serif; color:var(--gold-light); margin-top:1rem; }
.bot-bubble strong { color:var(--gold-light); }
.bot-bubble em     { color:var(--lotus-pink);  }
.bot-bubble ul,.bot-bubble ol { padding-left:1.4rem; }
.bot-bubble li { margin-bottom:0.3rem; }

/* ── shloka ── */
.shloka-box {
    background: rgba(201,150,58,0.06);
    border: 1px solid rgba(201,150,58,0.25);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-top: 1rem;
    font-style: italic;
    color: var(--gold-light);
    line-height: 1.8;
}
.shloka-box .shloka-title {
    font-family:'Cinzel',serif;
    font-style:normal;
    font-size:0.7rem;
    letter-spacing:0.2em;
    color:var(--saffron);
    text-transform:uppercase;
    display:block;
    margin-bottom:0.5rem;
}

hr { border:none; border-top:1px solid var(--indigo-border); margin:1.5rem 0; }

[data-testid="stImage"] img {
    border-radius:14px;
    border:1px solid var(--indigo-border);
    box-shadow:0 8px 30px rgba(0,0,0,0.5);
    margin:0.5rem 0 1rem;
}
[data-testid="stAudio"] audio {
    filter:invert(0.85) hue-rotate(180deg) saturate(0.6);
    border-radius:30px;
    width:100%;
}
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:var(--deep-indigo); }
::-webkit-scrollbar-thumb { background:var(--indigo-border); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:var(--gold); }
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
    st.markdown(f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>',
                unsafe_allow_html=True)

# ---------------- ENV ----------------
load_dotenv()

# ---------------- AGE DETECTION ----------------
if "age_group" not in st.session_state:
    with st.spinner("📷 Detecting viewer profile…"):
        _age = detect_age()
        st.session_state.age_group    = get_age_group(_age)
        st.session_state.detected_age = _age

_AGE_ICONS = {"child": "🧒", "teen": "🧑", "adult": "👤", "senior": "🧓", "unknown": "👤"}
_age_icon  = _AGE_ICONS.get(st.session_state.age_group, "👤")
_age_label = (
    f"~{st.session_state.detected_age} yrs · {st.session_state.age_group.capitalize()}"
    if st.session_state.detected_age else "Profile: Unknown"
)

# ============================================================
# RESPONSE LANGUAGE CONFIG
# ============================================================
RESPONSE_LANG_CONFIG = {
    "English": {
        "flag":        "🇬🇧",
        "tts_voice":   "en-IN-NeerjaNeural",
        "lang_code":   "en",
        # Clear, firm instruction — questions will always arrive in English
        "instruction": (
            "Respond entirely in English. "
            "Every heading, bullet point and explanation must be in English."
        ),
    },
    "हिन्दी": {
        "flag":        "🇮🇳",
        "tts_voice":   "hi-IN-SwaraNeural",
        "lang_code":   "hi",
        "instruction": (
            "उत्तर पूरी तरह हिन्दी में दें। "
            "सभी शीर्षक, सूचियाँ और व्याख्याएँ हिन्दी में लिखें। "
            "संस्कृत शब्दों को हिन्दी में सरल रूप से समझाएं। "
            "प्रश्न अंग्रेज़ी में होगा — फिर भी उत्तर केवल हिन्दी में दें।"
        ),
    },
    "मराठी": {
        "flag":        "🟠",
        "tts_voice":   "mr-IN-AarohiNeural",
        "lang_code":   "mr",
        "instruction": (
            "उत्तर पूर्णपणे मराठीत द्या. "
            "सर्व शीर्षके, यादी आणि स्पष्टीकरणे मराठीत लिहा. "
            "संस्कृत शब्द मराठीत सोप्या भाषेत समजावून सांगा. "
            "प्रश्न इंग्रजीत असेल — तरीही उत्तर फक्त मराठीत द्या."
        ),
    },
}

# Initialise default language
if "response_language" not in st.session_state:
    st.session_state.response_language = "English"

# ── Badges ──
_rl  = st.session_state.response_language
_rlc = RESPONSE_LANG_CONFIG[_rl]
st.markdown(
    f'<div class="badge-row">'
    f'<span class="age-badge">{_age_icon} {_age_label}</span>'
    f'<span class="lang-badge">{_rlc["flag"]} Response: {_rl}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Response language selector ──
st.markdown('<p class="section-heading">✦ Response Language ✦</p>', unsafe_allow_html=True)

lang_cols = st.columns(len(RESPONSE_LANG_CONFIG))
for col, (lang_name, cfg) in zip(lang_cols, RESPONSE_LANG_CONFIG.items()):
    with col:
        btn_label = f"{cfg['flag']} {lang_name}"
        if st.button(btn_label, use_container_width=True, key=f"lang_{lang_name}"):
            if st.session_state.response_language != lang_name:
                st.session_state.response_language = lang_name
                st.session_state.pop("qa_chain", None)  # rebuild with new language prompt
                st.rerun()

# ============================================================
# AGE-AWARE SYSTEM PROMPTS
# ============================================================
AGE_SYSTEM_PROMPTS = {
    "child": """You are VedaMitra, a warm and friendly guide explaining ancient Indian wisdom to a young child under 13.
- Use very simple words a child can easily understand.
- Use fun comparisons, short stories and relatable examples.
- Keep sentences very short. Be warm, playful and encouraging.
- Immediately explain any Sanskrit word in the simplest possible way.
- Limit your answer to 3–5 short sentences.""",

    "teen": """You are VedaMitra, explaining ancient Indian wisdom to a teenager (13–19 years old).
- Use clear, modern and friendly language. Avoid being preachy or boring.
- Relate the answer to everyday teenage life where naturally possible.
- Explain Sanskrit terms simply right after using them.
- Use bullet points when they make things clearer.
- Keep answers short and to the point (4–5 lines).
- Give a focused, medium-length answer.""",

    "adult": """You are VedaMitra, a knowledgeable guide on ancient Indian wisdom speaking to an adult.
- Keep the answer concise (5–6 lines maximum).
- Avoid long paragraphs.
- Focus only on key points.
- Use proper vocabulary. Sanskrit terms welcome with brief contextual explanations.
- Provide depth, nuance and context. Reference specific scriptures where relevant.
- Be precise, informative and respectful.
- Structure clearly with headings or bullet points for complex topics.""",

    "senior": """You are VedaMitra, respectfully sharing ancient Indian wisdom with a senior person (50+).
- Use clear, warm and deeply respectful language.
- Explain Sanskrit terms gently within the flow.
- Draw connections to traditional values and timeless wisdom.
- Keep the tone calm, thoughtful and unhurried.
- Keep the response brief and clear (5–6 lines).
- Be compassionate, patient and dignified.""",

    "unknown": """You are VedaMitra, a knowledgeable and friendly guide on ancient Indian wisdom.
- Use clear and accessible language for a general adult audience.
- Provide balanced depth — neither too simple nor too complex.
- Keep the answer concise and limited to key points only (5 lines max).
- Explain Sanskrit terms naturally when used.""",
}

# ---------------- BUILD CHAIN (age + response language aware) ----------------
def build_chain(retriever, llm, memory, age_group, lang_instruction):
    system_prompt = AGE_SYSTEM_PROMPTS.get(age_group, AGE_SYSTEM_PROMPTS["unknown"])

    prompt_template = f"""{system_prompt}

LANGUAGE INSTRUCTION (mandatory — highest priority):
{lang_instruction}
Your entire response — every word, heading and bullet — must follow the language rule above.
NOTE: The user's question will always be written in English. That does NOT affect the language of your answer.

Use the following context retrieved from ancient Indian texts to answer the question.
If the context lacks sufficient information, draw on your own knowledge while staying true
to the spirit and accuracy of Indian philosophy and tradition.

Context:
{{context}}

Chat History:
{{chat_history}}

Question: {{question}}

Answer:"""

    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=prompt_template,
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
    )

# ---------------- LOAD RAG COMPONENTS ----------------
@st.cache_resource
def load_rag_components(namespace):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    vectorstore = PineconeVectorStore(
        index=pc.Index("iks-rag-v2"), embedding=embeddings, namespace=namespace
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    llm = ChatMistralAI(
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        model="mistral-large-latest",
        temperature=0.3,
        max_tokens=500,
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
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

st.markdown('<p class="section-heading">✦ Select Knowledge Domain ✦</p>', unsafe_allow_html=True)

domain_cols = st.columns(5)
for col, (label, namespace) in zip(domain_cols, CATEGORY_NAMESPACE_MAP.items()):
    with col:
        if st.button(label, use_container_width=True, key=f"domain_{namespace}"):
            st.session_state.pop("qa_chain", None)
            st.session_state.selected_category = namespace

if st.session_state.selected_category:
    display_name = {v: k for k, v in CATEGORY_NAMESPACE_MAP.items()}[st.session_state.selected_category]
    st.success(f"Domain: {display_name}")
else:
    st.warning("Please select a knowledge domain to continue.")
    st.stop()

# ── Build / reuse chain ──
if "qa_chain" not in st.session_state:
    _retriever, _llm, _memory = load_rag_components(st.session_state.selected_category)
    _lang_cfg = RESPONSE_LANG_CONFIG[st.session_state.response_language]
    st.session_state.qa_chain = build_chain(
        _retriever, _llm, _memory,
        st.session_state.age_group,
        _lang_cfg["instruction"],
    )

qa_chain = st.session_state.qa_chain

# ---------------- IMAGE UTILITIES ----------------
@st.cache_resource
def load_image_map():
    with open("data/image_map.json") as f:
        return json.load(f)

image_map = load_image_map()

def fetch_images(query, num_images=3):
    try:
        results = GoogleSearch({
            "engine": "google_images",
            "q": query,
            "api_key": os.getenv("SERPAPI_KEY"),
            "num": num_images,
        }).get_dict()
        return [img["original"] for img in results.get("images_results", [])[:num_images]]
    except Exception:
        return []

def fetch_image(query):
    # Query is always English so plain ASCII matching is fine
    cleaned = re.sub(r'[^\w\s]', '', query.lower())
    for key, path in image_map.items():
        if key in cleaned:
            return [os.path.join("data", path)]
    suffix = " yoga pose" if "yoga" in cleaned else (
             " indian mythology character" if any(k in cleaned for k in ["who","sita","rama","krishna"]) else " india")
    return fetch_images(cleaned + suffix, num_images=3)

# ---------------- TTS / TEXT UTILITIES ----------------
def remove_diacritics(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))

def clean_text_for_tts(text, lang_code):
    """Strip markdown; keep Devanagari intact for hi/mr."""
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    text = re.sub(r'[-•]\s', '', text)
    text = re.sub(r'\n', ' ', text)
    if lang_code == "en":
        text = remove_diacritics(text)
    return text.strip()

def format_response(text):
    text = re.sub(r"(###)", r"\n\n\1", text)
    text = text.replace("\n", "\n\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return markdown.markdown(text)

async def generate_audio(text, voice):
    filename = f"temp_{uuid.uuid4().hex}.mp3"
    await edge_tts.Communicate(text, voice=voice, rate="+10%").save(filename)
    return filename

# ---------------- WHISPER — English only ----------------
@st.cache_resource
def load_whisper():
    return WhisperModel("tiny", device="cpu", compute_type="int8")

whisper_model = load_whisper()

def record_audio():
    st.info("🎙️ Listening for 6 seconds… (speak in English)")
    audio = sd.rec(int(6 * 16000), samplerate=16000, channels=1)
    sd.wait()
    return audio.flatten()

def transcribe(audio):
    # language="en" keeps Whisper focused on English only
    segments, _ = whisper_model.transcribe(audio, beam_size=3, language="en")
    return " ".join(seg.text for seg in segments).strip()

# ---------------- SHLOKA ----------------
@st.cache_resource
def load_shloka_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    vs = PineconeVectorStore(index=pc.Index("iks-shlokas"), embedding=embeddings)
    return vs.as_retriever(search_kwargs={"k": 1})

shloka_retriever = load_shloka_retriever()

def get_relevant_shloka(query):
    docs = shloka_retriever.invoke(query)
    if not docs:
        return ""
    text = re.sub(r"।।.*?।।", "", docs[0].page_content).replace("\\n", "\n").strip()
    return f'<div class="shloka-box"><span class="shloka-title">✦ Shloka ✦</span>{text}</div>'

# ---------------- DIVIDER ----------------
st.markdown("<hr>", unsafe_allow_html=True)

# ── English-only hint ──
st.markdown(
    '<p style="text-align:center;font-size:0.82rem;color:var(--muted);'
    'font-family:\'Cinzel\',serif;letter-spacing:0.1em;">'
    '✦ Please type or speak your question in English ✦</p>',
    unsafe_allow_html=True,
)

# ---------------- INPUT ROW ----------------
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input(
        label="question",
        placeholder="Ask about yoga, dharma, healing, or ancient wisdom…",
        label_visibility="collapsed",
    )
with col2:
    if st.button("🎙️", use_container_width=True, help="Speak in English"):
        audio_data = record_audio()
        user_input  = transcribe(audio_data)

# ---------------- RESPONSE ----------------
if user_input:
    # User bubble — question is always English
    st.markdown(
        f'<div class="user-bubble"><span class="bubble-label">You</span>{user_input}</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Consulting the ancient texts…"):
        result = qa_chain.invoke({"question": user_input})
        answer = result["answer"]

    # Images fetched from the English query
    images = fetch_image(user_input)
    if images:
        img_cols = st.columns(len(images))
        for i, img in enumerate(images):
            with img_cols[i]:
                if isinstance(img, str) and os.path.exists(img):
                    st.image(img, use_container_width=True)
                else:
                    st.image(img, use_container_width=True)

    # Stream answer word-by-word
    resp_lang_cfg = RESPONSE_LANG_CONFIG[st.session_state.response_language]
    placeholder   = st.empty()
    temp = ""
    for word in answer.split():
        temp += word + " "
        placeholder.markdown(
            f'<div class="bot-bubble"><span class="bubble-label">VedaMitra</span>'
            f'{format_response(temp)}</div>',
            unsafe_allow_html=True,
        )
        time.sleep(0.04)

    # TTS in the selected response language
    tts_text   = clean_text_for_tts(answer, resp_lang_cfg["lang_code"])
    audio_file = asyncio.run(generate_audio(tts_text, resp_lang_cfg["tts_voice"]))
    st.audio(audio_file)