# VedaMitra

Voice & RAG-powered Indian Knowledge System

## Overview

VedaMitra is an interactive web application that provides answers to questions about ancient Indian wisdom, including Yoga, Ayurveda, Ramayana, and Mahabharata. The app uses voice input, text-to-speech output, and RAG (Retrieval-Augmented Generation) for accurate responses.

## Features

- **Voice Input**: Speak your questions in English (Whisper-powered)
- **Text-to-Speech**: Listen to responses in your preferred language
- **Multi-language Support**: English, Hindi (हिन्दी), Marathi (मराठी)
- **Age Detection**: Responses tailored to age group (child, teen, adult, senior)
- **Knowledge Domains**: Yoga, Ayurveda, Ramayana, Mahabharata, General
- **Image Support**: Relevant images displayed with answers

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: Mistral AI (mistral-large-latest)
- **Vector Store**: Pinecone
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Speech-to-Text**: Faster Whisper
- **Text-to-Speech**: Edge TTS
- **Age Detection**: ONNX model

## Project Structure

```
VedaMitra/
├── app.py                 # Main Streamlit application
├── load_data.py          # Data loading utilities
├── load_shlokas.py       # Shloka retrieval utilities
├── age_detection.py     # Age detection module
├── intro_audio.py        # Intro audio utilities
├── requirements.txt     # Python dependencies
├── data/                 # Data files
│   ├── yoga/           # Yoga pose data
│   ├── philosophy/    # Philosophy data
│   ├── ayurveda/      # Ayurveda data
│   ├── epics/         # Ramayana & Mahabharata data
│   ├── images/        # Image files
│   └── intro.mp3       # Intro audio
├── models/              # ML models
│   ├── face_detector.caffemodel
│   ├── deploy.prototxt
│   └── age_gender.onnx
└── shlokas.json        # Shloka data
```

## Environment Variables

Create a `.env` file with the following:

```env
PINECONE_API_KEY=your_pinecone_api_key
MISTRAL_API_KEY=your_mistral_api_key
SERPAPI_KEY=your_serpapi_key
HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd VedaMitra
```

2. Create and activate virtual environment:

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables in `.env` file.

5. Run the application:

```bash
streamlit run app.py
```

## Usage

1. **Startup**: The app plays an intro audio on first load.

2. **Age Detection**: A camera captures your image to detect age group for personalized responses.

3. **Select Response Language**: Choose from English, Hindi, or Marathi.

4. **Select Knowledge Domain**: Choose from Yoga, Ayurveda, Ramayana, Mahabharata, or General.

5. **Ask Question**: Type or speak your question in English.

6. **Receive Answer**: Get a personalized response with relevant images and audio playback.

## API Keys Required

- **Pinecone**: Vector database for RAG
- **Mistral AI**: Language model for generating answers
- **SerpAPI**: Google Images for fetching relevant images

## Notes

- Questions should be typed or spoken in English
- Voice input uses English-only recognition
- Age detection uses on-device ML (privacy-friendly)
- Response language can be changed anytime
- Knowledge domain selection clears previous chat history