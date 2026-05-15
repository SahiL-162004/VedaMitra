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
import cv2
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

st.set_page_config(page_title="VedaMitra", page_icon="ॐ", layout="centered")

# ============================================================
# Ultrasonic sensor — same pattern as smart_robot_opencv.py
# ============================================================
ULTRASONIC_TRIG_PIN = 23
ULTRASONIC_ECHO_PIN = 24
THRESHOLD = 60  # cm


def _is_raspberry_pi():
    try:
        with open("/proc/cpuinfo") as f:
            return "Raspberry" in f.read() or "BCM" in f.read()
    except Exception:
        return False


@st.cache_resource
def _get_sensor():
    if not _is_raspberry_pi():
        return None
    try:
        from gpiozero import DistanceSensor
        return DistanceSensor(echo=ULTRASONIC_ECHO_PIN, trigger=ULTRASONIC_TRIG_PIN)
    except Exception:
        return None


def check_ultrasonic_presence():
    sensor = _get_sensor()
    if sensor is None:
        return None
    distance = sensor.distance * 100
    if distance == 0 or distance > 400:
        return False
    return 10 < distance <= THRESHOLD


# ============================================================
# Session state
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "face_detection"
if "detected_age" not in st.session_state:
    st.session_state.detected_age = None
if "age_group" not in st.session_state:
    st.session_state.age_group = None


# ============================================================
# GLOBAL THEME (shared across all 4 screens) – LIGHT, ENERGETIC
#   Palette: Cream, Ivory, Turmeric, Saffron, Peacock Teal,
#            Lotus Pink, Deep Maroon, Indigo, Marigold Gold
# ============================================================
GLOBAL_THEME = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;600;700&family=Yatra+One&family=Tiro+Devanagari+Hindi&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=EB+Garamond:ital,wght@0,400;0,500;1,400&display=swap');

:root{
  --cream:        #FFF7E6;
  --cream-2:      #FFF1D0;
  --ivory:        #FFFBF2;
  --turmeric:     #F4B43C;
  --saffron:      #FF8A1E;
  --marigold:     #E8A33D;
  --gold:         #C9962B;
  --gold-soft:    #E7C76A;
  --peacock:      #0E8F8A;
  --peacock-dk:   #0A6B68;
  --lotus:        #E84A7F;
  --lotus-soft:   #F58FAE;
  --maroon:       #7A1F2B;
  --indigo:       #2A2A6B;
  --indigo-ink:   #1B1C46;
  --paper-line:   rgba(122,31,43,0.18);
}

html, body, [data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1100px 700px at 12% -10%, rgba(244,180,60,0.30) 0%, transparent 60%),
    radial-gradient(900px 700px at 100% 0%, rgba(232,74,127,0.18) 0%, transparent 55%),
    radial-gradient(900px 700px at 50% 110%, rgba(14,143,138,0.18) 0%, transparent 55%),
    linear-gradient(180deg, #FFF8E7 0%, #FFF1D0 100%) !important;
  background-attachment: fixed !important;
  color: var(--indigo-ink) !important;
  font-family: 'Plus Jakarta Sans', 'EB Garamond', Georgia, serif;
  font-size: 17px;
}

/* Subtle mandala paper texture overlay */
[data-testid="stAppViewContainer"]::before{
  content:"";
  position:fixed; inset:0;
  pointer-events:none;
  background-image:
    radial-gradient(circle at 20% 30%, rgba(201,150,43,0.07) 0 1px, transparent 2px),
    radial-gradient(circle at 70% 80%, rgba(122,31,43,0.06) 0 1px, transparent 2px),
    radial-gradient(circle at 80% 20%, rgba(14,143,138,0.06) 0 1px, transparent 2px);
  background-size: 38px 38px, 52px 52px, 44px 44px;
  opacity:0.7;
  z-index:0;
}

[data-testid="stHeader"]{ background: transparent !important; }
[data-testid="block-container"]{
  padding-top: 1.5rem; padding-bottom: 3rem;
  position: relative; z-index: 1;
}

/* Decorative corner mandalas (SVG via CSS mask) */
.corner-mandala{
  position: fixed; width: 220px; height: 220px;
  pointer-events:none; opacity: 0.22; z-index: 0;
  background: conic-gradient(from 0deg, var(--saffron), var(--lotus), var(--peacock), var(--turmeric), var(--saffron));
  -webkit-mask: radial-gradient(circle, transparent 28%, #000 30%, #000 36%, transparent 38%, #000 42%, #000 46%, transparent 48%, #000 52%, #000 58%, transparent 60%);
          mask: radial-gradient(circle, transparent 28%, #000 30%, #000 36%, transparent 38%, #000 42%, #000 46%, transparent 48%, #000 52%, #000 58%, transparent 60%);
  animation: rotateMandala 60s linear infinite;
}
.corner-mandala.tl{ top:-70px; left:-70px; }
.corner-mandala.br{ bottom:-70px; right:-70px; animation-direction: reverse; }

@keyframes rotateMandala { to { transform: rotate(360deg); } }

/* Common keyframes */
@keyframes shimmer { 0%{background-position:0% center;} 100%{background-position:200% center;} }
@keyframes floatY  { 0%,100%{ transform: translateY(0);} 50%{ transform: translateY(-8px);} }
@keyframes breathe { 0%,100%{ transform: scale(1); opacity:.85;} 50%{ transform: scale(1.06); opacity:1;} }
@keyframes rippleOut { 0%{ transform: scale(1); opacity:0.9;} 100%{ transform: scale(3); opacity:0;} }
@keyframes spinSlow { to { transform: rotate(360deg); } }
@keyframes fadeUp { from{ opacity:0; transform: translateY(14px);} to{ opacity:1; transform: translateY(0);} }
@keyframes glowPulse {
  0%,100%{ box-shadow: 0 8px 30px rgba(232,163,61,0.25), 0 0 0 1px rgba(201,150,43,0.25);}
  50%   { box-shadow: 0 14px 48px rgba(232,74,127,0.30), 0 0 0 1px rgba(232,74,127,0.30);}
}

/* Buttons – marigold/saffron with 3D press */
.stButton > button{
  background: linear-gradient(135deg, #FFB347 0%, #FF8A1E 55%, #E84A7F 100%) !important;
  border: none !important;
  border-radius: 999px !important;
  color: #fff !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-weight: 700 !important;
  letter-spacing: 0.04em !important;
  padding: 0.85rem 1.6rem !important;
  box-shadow:
     0 10px 24px rgba(232,74,127,0.28),
     0 4px 0 rgba(122,31,43,0.18),
     inset 0 1px 0 rgba(255,255,255,0.45) !important;
  transform: translateZ(0);
  transition: all .25s cubic-bezier(.2,.8,.2,1) !important;
}
.stButton > button:hover{
  transform: translateY(-2px) scale(1.02) !important;
  box-shadow:
     0 16px 34px rgba(232,74,127,0.36),
     0 6px 0 rgba(122,31,43,0.18),
     inset 0 1px 0 rgba(255,255,255,0.55) !important;
}
.stButton > button:active{
  transform: translateY(1px) scale(0.99) !important;
  box-shadow: 0 4px 14px rgba(232,74,127,0.30), inset 0 2px 6px rgba(0,0,0,0.10) !important;
}

/* Alerts */
[data-testid="stAlert"]{
  background: linear-gradient(135deg, rgba(244,180,60,0.18), rgba(14,143,138,0.10)) !important;
  border: 1px solid rgba(201,150,43,0.45) !important;
  border-left: 5px solid var(--saffron) !important;
  color: var(--indigo-ink) !important;
  border-radius: 14px !important;
}

/* Text input */
[data-testid="stTextInput"] input{
  background: linear-gradient(180deg, #FFFDF6, #FFF6E1) !important;
  border: 2px solid rgba(122,31,43,0.18) !important;
  border-radius: 999px !important;
  color: var(--indigo-ink) !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-size: 1rem !important;
  padding: 0.85rem 1.3rem !important;
  transition: all .25s ease !important;
}
[data-testid="stTextInput"] input:focus{
  border-color: var(--lotus) !important;
  box-shadow: 0 0 0 4px rgba(232,74,127,0.18), 0 8px 24px rgba(232,74,127,0.12) !important;
}
[data-testid="stTextInput"] input::placeholder{
  color: rgba(27,28,70,0.55) !important;
  font-style: italic;
}

/* Scrollbar */
::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-track { background: var(--cream); }
::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, var(--turmeric), var(--lotus));
  border-radius: 10px;
}

/* Audio */
[data-testid="stAudio"] audio{
  filter: saturate(1.05);
  border-radius: 999px;
  width: 100%;
}

/* Images */
[data-testid="stImage"] img{
  border-radius: 18px;
  border: 1px solid rgba(122,31,43,0.18);
  box-shadow:
     0 18px 40px rgba(122,31,43,0.18),
     0 0 0 4px rgba(255,255,255,0.7) inset;
  transition: transform .35s ease, box-shadow .35s ease;
}
[data-testid="stImage"] img:hover{
  transform: translateY(-3px) scale(1.015);
  box-shadow:
     0 24px 48px rgba(232,74,127,0.22),
     0 0 0 4px rgba(255,255,255,0.8) inset;
}

/* Dividers */
hr{
  border: none; height: 18px;
  background:
    radial-gradient(circle at 50% 50%, var(--saffron) 0 3px, transparent 4px),
    linear-gradient(90deg, transparent 0%, var(--gold-soft) 18%, var(--lotus) 50%, var(--peacock) 82%, transparent 100%);
  background-repeat: no-repeat;
  background-size: 8px 8px, 100% 2px;
  background-position: center, center;
  margin: 1.8rem 0;
  opacity: 0.85;
}

/* Section heading */
.section-heading{
  font-family: 'Cinzel', serif;
  font-size: 0.9rem;
  letter-spacing: 0.32em;
  text-transform: uppercase;
  text-align: center;
  margin: 1.6rem 0 1rem;
  background: linear-gradient(90deg, var(--maroon), var(--saffron), var(--lotus), var(--peacock), var(--maroon));
  background-size: 200% auto;
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: shimmer 5s linear infinite;
}
.section-heading::before, .section-heading::after{
  content: "❖"; color: var(--saffron); margin: 0 .6rem; -webkit-text-fill-color: var(--saffron);
}
</style>
<div class="corner-mandala tl"></div>
<div class="corner-mandala br"></div>
"""

# Inject global theme always
st.markdown(GLOBAL_THEME, unsafe_allow_html=True)


# Reusable SVG motifs ----------------------------------------------------------
OM_SVG = """
<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" class="om-svg">
  <defs>
    <radialGradient id="omGrad" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#FFD27A"/>
      <stop offset="55%" stop-color="#FF8A1E"/>
      <stop offset="100%" stop-color="#7A1F2B"/>
    </radialGradient>
  </defs>
  <circle cx="100" cy="100" r="92" fill="none" stroke="url(#omGrad)" stroke-width="3" opacity="0.7"/>
  <text x="50%" y="58%" text-anchor="middle" font-family="Tiro Devanagari Hindi, serif"
        font-size="120" fill="url(#omGrad)">ॐ</text>
</svg>
"""

LOTUS_SVG = """
<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg" class="lotus-svg" aria-hidden="true">
  <g transform="translate(60,70)">
    <ellipse cx="0" cy="6" rx="40" ry="10" fill="#0E8F8A" opacity="0.35"/>
    <g fill="#E84A7F">
      <path d="M0,-40 Q14,-18 0,6 Q-14,-18 0,-40" opacity="0.95"/>
      <path d="M0,-40 Q14,-18 0,6 Q-14,-18 0,-40" transform="rotate(40)" opacity="0.85"/>
      <path d="M0,-40 Q14,-18 0,6 Q-14,-18 0,-40" transform="rotate(-40)" opacity="0.85"/>
      <path d="M0,-40 Q14,-18 0,6 Q-14,-18 0,-40" transform="rotate(75)" opacity="0.75"/>
      <path d="M0,-40 Q14,-18 0,6 Q-14,-18 0,-40" transform="rotate(-75)" opacity="0.75"/>
    </g>
    <circle r="6" fill="#F4B43C"/>
  </g>
</svg>
"""

SRI_YANTRA_SVG = """
<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" class="yantra-svg" aria-hidden="true">
  <g fill="none" stroke="#7A1F2B" stroke-width="1.2" opacity="0.55">
    <circle cx="100" cy="100" r="92"/>
    <circle cx="100" cy="100" r="78"/>
    <polygon points="100,18 178,150 22,150"/>
    <polygon points="100,182 22,50 178,50"/>
    <polygon points="100,38 158,140 42,140"/>
    <polygon points="100,162 42,60 158,60"/>
    <polygon points="100,58 138,128 62,128"/>
    <polygon points="100,142 62,72 138,72"/>
    <circle cx="100" cy="100" r="6" fill="#FF8A1E" stroke="none"/>
  </g>
</svg>
"""


# ============================================================
# PRESENCE DETECTION  (Ultrasonic)
# ============================================================
if st.session_state.page == "presence_detection":
    st.markdown(f"""
    <style>
      .presence-wrap{{
        text-align:center; padding: 2.5rem 1rem 1rem; position:relative;
        animation: fadeUp .8s ease both;
      }}
      .yantra-bg{{
        position:absolute; left:50%; top:46%; transform: translate(-50%,-50%);
        width: 360px; height: 360px; opacity:.55; z-index:0;
        animation: spinSlow 90s linear infinite;
      }}
      .om-hero{{
        width:140px; height:140px; margin: .2rem auto .6rem;
        position:relative; z-index:1;
        filter: drop-shadow(0 12px 30px rgba(232,74,127,0.25));
        animation: floatY 4.5s ease-in-out infinite;
      }}
      .presence-title{{
        font-family:'Cinzel',serif;
        font-size: clamp(2.4rem, 5vw, 3.6rem);
        font-weight:700; letter-spacing:.08em;
        margin: .4rem 0 .2rem; position:relative; z-index:1;
        background: linear-gradient(90deg,#7A1F2B,#FF8A1E,#E84A7F,#0E8F8A,#7A1F2B);
        background-size: 200% auto; -webkit-background-clip:text;
        -webkit-text-fill-color: transparent; background-clip:text;
        animation: shimmer 6s linear infinite;
      }}
      .presence-sub{{
        font-family:'EB Garamond',serif; font-style:italic;
        color: var(--maroon); font-size: 1.15rem;
        position:relative; z-index:1;
      }}
      .wave-stage{{
        position:relative; width:220px; height:220px;
        margin: 1.8rem auto 1rem; z-index:1;
      }}
      .wave-stage .ring{{
        position:absolute; inset:50% auto auto 50%;
        width: 70px; height: 70px; margin:-35px 0 0 -35px;
        border-radius: 50%;
        border: 2px solid rgba(255,138,30,0.55);
        animation: rippleOut 2.4s ease-out infinite;
      }}
      .wave-stage .ring:nth-child(2){{ animation-delay:.6s; border-color: rgba(232,74,127,0.55); }}
      .wave-stage .ring:nth-child(3){{ animation-delay:1.2s; border-color: rgba(14,143,138,0.55); }}
      .wave-stage .core{{
        position:absolute; inset:50% auto auto 50%;
        width: 90px; height:90px; margin:-45px 0 0 -45px;
        border-radius:50%;
        background: radial-gradient(circle, #FFF1D0 0%, #FFB347 60%, #E84A7F 100%);
        display:flex; align-items:center; justify-content:center;
        box-shadow: 0 14px 40px rgba(232,74,127,0.35), inset 0 0 24px rgba(255,255,255,0.5);
        animation: breathe 3s ease-in-out infinite;
        font-size: 2rem;
      }}
      .scan-line{{
        display:inline-block; font-family:'Plus Jakarta Sans',sans-serif;
        color: var(--peacock-dk); font-weight:600; letter-spacing:.08em;
        padding: .55rem 1.4rem; border-radius:999px;
        background: rgba(14,143,138,0.10);
        border: 1px dashed rgba(14,143,138,0.45);
        animation: breathe 2.6s ease-in-out infinite;
      }}
    </style>
    <div class="presence-wrap">
      <div class="yantra-bg">{SRI_YANTRA_SVG}</div>
      <div class="om-hero">{OM_SVG}</div>
      <h1 class="presence-title">VedaMitra</h1>
      <p class="presence-sub">~ A living gateway to Bhāratīya wisdom ~</p>
      <div class="wave-stage">
        <div class="ring"></div><div class="ring"></div><div class="ring"></div>
        <div class="core">🪔</div>
      </div>
      <div class="scan-line">✦ Awaiting your presence ✦</div>
    </div>
    """, unsafe_allow_html=True)

    result = check_ultrasonic_presence()

    if result is None:
        if st.button("▶ Begin the Journey", use_container_width=True, type="primary"):
            st.session_state.page = "face_detection"
            st.rerun()
        st.stop()

    if result:
        st.session_state.page = "face_detection"
        time.sleep(0.3)
        st.rerun()

    status = st.empty()
    for _ in range(120):
        sensor = _get_sensor()
        if sensor is not None:
            d = sensor.distance * 100
            status.info(f"📏 Distance: {d:.1f} cm  (threshold: 10–{THRESHOLD} cm)")
        if check_ultrasonic_presence():
            st.session_state.page = "face_detection"
            st.rerun()
        time.sleep(0.5)

    st.stop()


# ============================================================
# FACE DETECTION (Camera)
# ============================================================
if st.session_state.page == "face_detection":
    st.markdown(f"""
    <style>
      .face-wrap{{
        text-align:center; padding: 2.5rem 1rem 1rem;
        position:relative; animation: fadeUp .8s ease both;
      }}
      .face-title{{
        font-family:'Cinzel',serif;
        font-size: clamp(2rem, 4.5vw, 3rem); font-weight:700;
        letter-spacing:.08em; margin:0 0 .2rem;
        background: linear-gradient(90deg,#0E8F8A,#0A6B68,#2A2A6B,#0E8F8A);
        background-size: 200% auto; -webkit-background-clip:text;
        background-clip:text; -webkit-text-fill-color: transparent;
        animation: shimmer 5s linear infinite;
      }}
      .face-sub{{ color:var(--maroon); font-family:'EB Garamond',serif; font-style:italic; font-size:1.1rem; }}
      .scan-stage{{
        position:relative; width: 240px; height: 240px;
        margin: 2rem auto 1rem;
      }}
      .scan-stage .arc{{
        position:absolute; inset:0; border-radius:50%;
        border: 3px solid transparent;
        border-top-color: var(--peacock);
        border-right-color: rgba(14,143,138,0.35);
        animation: spinSlow 2.4s linear infinite;
      }}
      .scan-stage .arc.b{{
        inset: 22px; border-top-color: var(--lotus);
        border-right-color: rgba(232,74,127,0.35);
        animation-duration: 3.2s; animation-direction: reverse;
      }}
      .scan-stage .arc.c{{
        inset: 44px; border-top-color: var(--turmeric);
        border-right-color: rgba(244,180,60,0.35);
        animation-duration: 4s;
      }}
      .scan-stage .lens{{
        position:absolute; inset: 70px; border-radius:50%;
        background: radial-gradient(circle, #FFFDF6 0%, #FFE0AD 55%, #E84A7F 100%);
        display:flex; align-items:center; justify-content:center;
        font-size: 2.4rem;
        box-shadow: 0 14px 40px rgba(14,143,138,.30), inset 0 0 22px rgba(255,255,255,0.6);
        animation: breathe 2.6s ease-in-out infinite;
      }}
      .chip{{
        display:inline-block; padding:.5rem 1.1rem; border-radius:999px;
        background: rgba(14,143,138,0.12); border:1px solid rgba(14,143,138,0.4);
        color: var(--peacock-dk); font-family:'Plus Jakarta Sans',sans-serif;
        font-weight:600; letter-spacing:.08em;
      }}
    </style>
    <div class="face-wrap">
      <h1 class="face-title">✦ Detected ✦</h1>
      <p class="face-sub">Presence felt. Reading your essence…</p>
      <div class="scan-stage">
        <div class="arc"></div>
        <div class="arc b"></div>
        <div class="arc c"></div>
        <div class="lens">👁️</div>
      </div>
      <span class="chip">📷 Camera active — please look gently into the lens</span>
    </div>
    """, unsafe_allow_html=True)

    from age_detection import wait_for_face_and_detect_age

    with st.spinner('Reading your essence…'):
        age = wait_for_face_and_detect_age(max_attempts=60, delay=0.5)

    if age:
        st.session_state.detected_age = age
        st.session_state.age_group = get_age_group(age)
        st.session_state.page = "profile"
        st.success(f"✓ Detection complete — Age detected: {age} years")
        time.sleep(1)
        st.rerun()
    else:
        st.warning("No face detected. Please try again.")
        if st.button("↻ Retry Darshana"):
            st.rerun()

    st.stop()


# ============================================================
# PROFILE PAGE
# ============================================================
if st.session_state.page == "profile":
    age = st.session_state.detected_age
    age_group = st.session_state.age_group

    st.markdown(f"""
    <style>
      .profile-wrap{{
        text-align:center; padding: 2rem 1rem 1rem;
        position:relative; animation: fadeUp .8s ease both;
      }}
      .profile-title{{
        font-family:'Cinzel',serif;
        font-size: clamp(2rem, 4.5vw, 3rem); font-weight:700;
        letter-spacing:.08em; margin:0 0 .4rem;
        background: linear-gradient(90deg, #7A1F2B, #E84A7F, #FF8A1E, #C9962B, #7A1F2B);
        background-size: 200% auto; -webkit-background-clip:text;
        background-clip:text; -webkit-text-fill-color: transparent;
        animation: shimmer 6s linear infinite;
      }}
      .age-stage{{
        position:relative; width: 260px; height: 260px;
        margin: 1.6rem auto 1rem;
        perspective: 800px;
      }}
      .yantra-back{{
        position:absolute; inset:-30px; opacity:.55;
        animation: spinSlow 80s linear infinite;
      }}
      .age-card{{
        position:absolute; inset:30px; border-radius: 50%;
        background:
          radial-gradient(circle at 30% 25%, #FFFDF6 0%, #FFE7B5 45%, #FFB347 80%, #E84A7F 100%);
        display:flex; flex-direction:column; align-items:center; justify-content:center;
        box-shadow:
           0 22px 60px rgba(232,74,127,0.30),
           inset 0 0 36px rgba(255,255,255,0.65),
           0 0 0 1px rgba(122,31,43,0.18);
        transform: rotateX(6deg);
        animation: glowPulse 4s ease-in-out infinite, floatY 5s ease-in-out infinite;
      }}
      .age-num{{
        font-family:'Cinzel',serif; font-weight:700;
        font-size: 4.6rem;
        background: linear-gradient(180deg, #7A1F2B, #FF8A1E);
        -webkit-background-clip:text; background-clip:text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
      }}
      .age-lbl{{
        margin-top:.2rem;
        font-family:'Cinzel',serif; letter-spacing:.32em;
        color: var(--maroon); font-size:.85rem; text-transform:uppercase;
      }}
      .group-pill{{
        display:inline-block; margin-top: .4rem;
        padding:.55rem 1.2rem; border-radius:999px;
        background: linear-gradient(135deg, rgba(244,180,60,0.25), rgba(232,74,127,0.20));
        border: 1px solid rgba(122,31,43,0.25);
        color: var(--maroon);
        font-family:'Cinzel',serif; letter-spacing:.22em; text-transform:uppercase;
        font-size: .82rem;
      }}
      .blessing{{
        margin-top: 1rem; color: var(--indigo);
        font-family:'EB Garamond',serif; font-style:italic; font-size: 1.1rem;
      }}
    </style>
    <div class="profile-wrap">
      <h1 class="profile-title">🙏 Welcome, Seeker</h1>
      <div class="age-stage">
        <div class="yantra-back">{SRI_YANTRA_SVG}</div>
        <div class="age-card">
          <div class="age-num">{age}</div>
          <div class="age-lbl">years young</div>
        </div>
      </div>
      <div class="group-pill">✦ {str(age_group).capitalize()} ✦</div>
      <p class="blessing">May this wisdom be personalised, in flow with your spirit.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🪔 Enter VedaMitra", use_container_width=True, type="primary"):
        st.session_state.page = "main_app"
        st.rerun()

    st.stop()


# ============================================================
# MAIN APP
# ============================================================
if st.session_state.page != "main_app":
    st.stop()

# Per-page CSS additions (chat bubbles, header, badges, domain pills, shloka)
st.markdown("""
<style>
.vm-header{
  text-align:center; padding: 1.2rem 0 .4rem;
  position: relative; animation: fadeUp .7s ease both;
}
.vm-header .om-bar{
  height: 4px; border-radius: 4px;
  background: linear-gradient(90deg, transparent, #C9962B, #FF8A1E, #E84A7F, #0E8F8A, #C9962B, transparent);
  background-size: 200% auto; animation: shimmer 5s linear infinite;
  margin-bottom: .9rem;
}
.vm-title{
  font-family: 'Cinzel', serif;
  font-size: clamp(2.4rem, 5vw, 3.6rem); font-weight: 700; letter-spacing: 0.10em;
  background: linear-gradient(90deg, #7A1F2B 0%, #FF8A1E 25%, #E84A7F 50%, #0E8F8A 75%, #7A1F2B 100%);
  background-size: 200% auto; -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; animation: shimmer 6s linear infinite;
  margin: 0; filter: drop-shadow(0 8px 20px rgba(232,74,127,0.20));
}
.vm-title .om{
  display:inline-block; -webkit-text-fill-color: #FF8A1E;
  margin-right:.2em; animation: floatY 4s ease-in-out infinite;
}
.vm-sub{
  font-family:'EB Garamond',serif; font-style:italic;
  color: var(--maroon); font-size: 1.05rem; margin-top: .25rem;
}

/* Badges */
.badge-row{
  display:flex; justify-content:center; gap:.6rem; flex-wrap:wrap;
  margin: .8rem 0 .2rem;
}
.badge{
  display:inline-flex; align-items:center; gap:.4rem;
  padding:.45rem 1.05rem; border-radius:999px;
  font-family:'Plus Jakarta Sans',sans-serif;
  font-size:.78rem; font-weight:600; letter-spacing:.12em; text-transform:uppercase;
  transition: transform .25s ease, box-shadow .25s ease;
}
.badge.age   { background: linear-gradient(135deg, rgba(14,143,138,0.10), rgba(42,42,107,0.10));
               border:1px solid rgba(14,143,138,0.40); color: var(--peacock-dk); }
.badge.lang  { background: linear-gradient(135deg, rgba(255,138,30,0.12), rgba(232,74,127,0.10));
               border:1px solid rgba(255,138,30,0.45); color: var(--maroon); }
.badge:hover{ transform: translateY(-2px); box-shadow: 0 8px 24px rgba(122,31,43,0.10); }

/* Chat bubbles */
.user-bubble{
  background: linear-gradient(135deg, #FFFDF6 0%, #FFE9C7 100%);
  border: 1px solid rgba(122,31,43,0.18);
  border-right: 5px solid var(--lotus);
  border-radius: 22px 22px 6px 22px;
  padding: .9rem 1.2rem; margin: .4rem 0 1rem;
  color: var(--indigo-ink); font-family:'Plus Jakarta Sans',sans-serif;
  box-shadow: 0 10px 28px rgba(232,74,127,0.10);
  animation: fadeUp .35s ease both;
}
.user-bubble .lbl{
  display:block; margin-bottom:.3rem;
  font-family:'Cinzel',serif; font-size:.7rem; letter-spacing:.28em;
  color: var(--lotus); text-transform:uppercase;
}
.bot-bubble{
  position: relative;
  background:
    radial-gradient(120% 80% at 0% 0%, rgba(244,180,60,0.18) 0%, transparent 55%),
    radial-gradient(120% 80% at 100% 100%, rgba(14,143,138,0.14) 0%, transparent 55%),
    linear-gradient(180deg, #FFFDF6 0%, #FFF5DE 100%);
  border: 1px solid rgba(122,31,43,0.18);
  border-left: 5px solid var(--saffron);
  border-radius: 6px 22px 22px 22px;
  padding: 1.2rem 1.4rem 1.1rem;
  margin-bottom: 1.2rem;
  color: var(--indigo-ink);
  font-family:'EB Garamond',serif; font-size: 1.08rem; line-height: 1.75;
  box-shadow: 0 14px 36px rgba(122,31,43,0.10);
  animation: fadeUp .35s ease both;
}
.bot-bubble::before{
  content:"॥"; position:absolute; top:6px; right:14px;
  color: rgba(201,150,43,0.35); font-size: 1.4rem;
}
.bot-bubble .lbl{
  display:block; margin-bottom:.4rem;
  font-family:'Cinzel',serif; font-size:.72rem; letter-spacing:.3em; text-transform:uppercase;
  background: linear-gradient(90deg, var(--maroon), var(--saffron), var(--lotus), var(--peacock), var(--maroon));
  background-size: 200% auto; -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color: transparent; animation: shimmer 5s linear infinite;
}
.bot-bubble h1,.bot-bubble h2,.bot-bubble h3{
  font-family:'Cinzel',serif; color: var(--maroon); margin-top: .8rem;
}
.bot-bubble strong{ color: var(--maroon); }
.bot-bubble em    { color: var(--peacock-dk); }
.bot-bubble ul, .bot-bubble ol{ padding-left: 1.4rem; }
.bot-bubble li::marker{ color: var(--saffron); }

/* Shloka */
.shloka-box{
  position:relative;
  background:
    radial-gradient(120% 100% at 0% 0%, rgba(244,180,60,0.30) 0%, transparent 60%),
    linear-gradient(180deg, #FFF8E0 0%, #FFEBC0 100%);
  border: 1px solid rgba(201,150,43,0.45);
  border-left: 5px solid var(--gold);
  border-radius: 16px;
  padding: 1.1rem 1.4rem;
  margin-top: 1rem;
  font-family:'EB Garamond',serif; font-style:italic;
  color: var(--maroon); line-height: 1.9;
  box-shadow: 0 10px 28px rgba(201,150,43,0.18);
}
.shloka-box::before{
  content:"ॐ"; position:absolute; top:10px; right:16px;
  font-style: normal; color: rgba(122,31,43,0.18); font-size: 1.8rem;
}
.shloka-box .shloka-title{
  display:block; margin-bottom:.5rem;
  font-style: normal;
  font-family:'Cinzel',serif; font-size:.72rem; letter-spacing:.3em; text-transform:uppercase;
  color: var(--saffron);
}

/* Domain & language buttons – styled via column selector */
div[data-testid="column"] button{
  background: linear-gradient(180deg, #FFFDF6 0%, #FFEBC0 100%) !important;
  border: 1px solid rgba(122,31,43,0.22) !important;
  color: var(--maroon) !important;
  font-family:'Plus Jakarta Sans',sans-serif !important;
  font-weight: 600 !important;
  font-size: .85rem !important;
  letter-spacing: .04em !important;
  border-radius: 16px !important;
  padding: .85rem .5rem !important;
  box-shadow:
     0 8px 20px rgba(122,31,43,0.08),
     inset 0 1px 0 rgba(255,255,255,0.7) !important;
  transition: transform .25s cubic-bezier(.2,.8,.2,1), box-shadow .25s ease, border-color .25s ease !important;
}
div[data-testid="column"] button:hover{
  transform: translateY(-3px) !important;
  border-color: var(--lotus) !important;
  color: var(--maroon) !important;
  box-shadow:
     0 14px 30px rgba(232,74,127,0.22),
     inset 0 1px 0 rgba(255,255,255,0.85) !important;
}

/* Voice mic button — last column gets golden ring */
div[data-testid="column"]:last-child button{
  background: radial-gradient(circle, #FFF1D0 0%, #F4B43C 70%, #C9962B 100%) !important;
  border: 2px solid var(--gold) !important;
  color: var(--maroon) !important;
  font-size: 1.3rem !important;
  box-shadow: 0 0 0 4px rgba(244,180,60,0.20), 0 10px 24px rgba(201,150,43,0.30) !important;
}
div[data-testid="column"]:last-child button:hover{
  transform: scale(1.06) !important;
  box-shadow: 0 0 0 6px rgba(244,180,60,0.25), 0 14px 30px rgba(201,150,43,0.35) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="vm-header">
  <div class="om-bar"></div>
  <h1 class="vm-title"><span class="om">ॐ</span> VedaMitra</h1>
  <p class="vm-sub">✦ A Voice & RAG-powered Indian Knowledge Companion ✦</p>
</div>
""", unsafe_allow_html=True)

if "startup_audio_played" not in st.session_state:
    st.session_state.startup_audio_played = True
    if os.path.exists("data/intro.mp3"):
        with open("data/intro.mp3", "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.markdown(f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>',
                    unsafe_allow_html=True)

load_dotenv()

_AGE_ICONS = {"child": "🌟", "teen": "⚡", "adult": "🔱", "senior": "🪷", "unknown": "✨"}
_age_icon  = _AGE_ICONS.get(st.session_state.age_group, "✨")
_age_label = (
    f"~{st.session_state.detected_age} yrs · {str(st.session_state.age_group).capitalize()}"
    if st.session_state.detected_age else "Profile: Unknown"
)

RESPONSE_LANG_CONFIG = {
    "English": {
        "flag":        "🇬🇧",
        "tts_voice":   "en-IN-NeerjaNeural",
        "lang_code":   "en",
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
            "उत्तर पूर्णपणे मराठीत द्या। "
            "सर्व शीर्षके, यादी आणि स्पष्टीकरणे मराठीत लिहा। "
            "संस्कृत शब्द मराठीत सोप्या भाषेत समजावून सांगा। "
            "प्रश्न इंग्रजीत असेल — तरीही उत्तर फक्त मराठीत द्या."
        ),
    },
}

if "response_language" not in st.session_state:
    st.session_state.response_language = "English"

_rl  = st.session_state.response_language
_rlc = RESPONSE_LANG_CONFIG[_rl]
st.markdown(
    f'<div class="badge-row">'
    f'<span class="badge age">{_age_icon} {_age_label}</span>'
    f'<span class="badge lang">{_rlc["flag"]} Response: {_rl}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

st.markdown('<p class="section-heading">Response Language</p>', unsafe_allow_html=True)

lang_cols = st.columns(len(RESPONSE_LANG_CONFIG))
for col, (lang_name, cfg) in zip(lang_cols, RESPONSE_LANG_CONFIG.items()):
    with col:
        btn_label = f"{cfg['flag']} {lang_name}"
        if st.button(btn_label, use_container_width=True, key=f"lang_{lang_name}"):
            if st.session_state.response_language != lang_name:
                st.session_state.response_language = lang_name
                st.session_state.pop("qa_chain", None)
                st.rerun()

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


CATEGORY_NAMESPACE_MAP = {
    "🧘 Yoga":        "yoga",
    "🌿 Ayurveda":    "ayurveda",
    "🏹 Ramayana":    "ramayana",
    "⚔️ Mahabharata": "mahabharata",
    "📜 General":     "general",
}

if "selected_category" not in st.session_state:
    st.session_state.selected_category = None

st.markdown('<p class="section-heading">Knowledge Domain</p>', unsafe_allow_html=True)

domain_cols = st.columns(5)
for col, (label, namespace) in zip(domain_cols, CATEGORY_NAMESPACE_MAP.items()):
    with col:
        if st.button(label, use_container_width=True, key=f"domain_{namespace}"):
            st.session_state.pop("qa_chain", None)
            st.session_state.selected_category = namespace

if st.session_state.selected_category:
    display_name = {v: k for k, v in CATEGORY_NAMESPACE_MAP.items()}[st.session_state.selected_category]
    st.success(f"🔮 Domain: {display_name}")
else:
    st.warning("🪔 Please select a knowledge domain to continue")
    st.stop()

if "qa_chain" not in st.session_state:
    _retriever, _llm, _memory = load_rag_components(st.session_state.selected_category)
    _lang_cfg = RESPONSE_LANG_CONFIG[st.session_state.response_language]
    st.session_state.qa_chain = build_chain(
        _retriever, _llm, _memory,
        st.session_state.age_group,
        _lang_cfg["instruction"],
    )

qa_chain = st.session_state.qa_chain


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
    cleaned = re.sub(r'[^\w\s]', '', query.lower())
    for key, path in image_map.items():
        if key in cleaned:
            return [os.path.join("data", path)]
    suffix = " yoga pose" if "yoga" in cleaned else (
             " indian mythology character" if any(k in cleaned for k in ["who","sita","rama","krishna"]) else " india")
    return fetch_images(cleaned + suffix, num_images=3)


def remove_diacritics(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def clean_text_for_tts(text, lang_code):
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


@st.cache_resource
def load_whisper():
    return WhisperModel("tiny.en", device="cpu", compute_type="int8")


whisper_model = load_whisper()


def record_audio():
    st.info("🎙️ Listening for 6 seconds... (speak in English)")
    audio = sd.rec(int(6 * 16000), samplerate=16000, channels=1)
    sd.wait()
    return audio.flatten()


def transcribe(audio):
    segments, _ = whisper_model.transcribe(audio, beam_size=3, language="en")
    return " ".join(seg.text for seg in segments).strip()


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
    return f'<div class="shloka-box"><span class="shloka-title">✦ Sacred Shloka ✦</span>{text}</div>'


st.markdown("<hr>", unsafe_allow_html=True)

st.markdown(
    '<p style="text-align:center;font-size:.92rem;color:var(--maroon);'
    "font-family:'EB Garamond',serif;font-style:italic;letter-spacing:.04em;\">"
    '✨ Type or speak your question in English ✨</p>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input(
        label="question",
        placeholder="🙏 Ask about yoga, dharma, healing, or ancient wisdom...",
        label_visibility="collapsed",
    )
with col2:
    if st.button("🎙️", use_container_width=True, help="Speak in English"):
        audio_data = record_audio()
        user_input  = transcribe(audio_data)


if user_input:
    st.markdown(
        f'<div class="user-bubble"><span class="lbl">🙏 You</span>{user_input}</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("✨ Consulting the ancient texts..."):
        result = qa_chain.invoke({"question": user_input})
        answer = result["answer"]

    images = fetch_image(user_input)
    if images:
        img_cols = st.columns(len(images))
        for i, img in enumerate(images):
            with img_cols[i]:
                if isinstance(img, str) and os.path.exists(img):
                    st.image(img, use_container_width=True)
                else:
                    st.image(img, use_container_width=True)

    resp_lang_cfg = RESPONSE_LANG_CONFIG[st.session_state.response_language]
    placeholder   = st.empty()
    temp = ""
    for word in answer.split():
        temp += word + " "
        placeholder.markdown(
            f'<div class="bot-bubble"><span class="lbl">🕉️ VedaMitra</span>'
            f'{format_response(temp)}</div>',
            unsafe_allow_html=True,
        )
        time.sleep(0.04)

    tts_text   = clean_text_for_tts(answer, resp_lang_cfg["lang_code"])
    audio_file = asyncio.run(generate_audio(tts_text, resp_lang_cfg["tts_voice"]))
    st.audio(audio_file)
