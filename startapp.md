# VedaMitra - Getting Started

## Prerequisites

- Python 3.8+
- API keys (Pinecone, Mistral AI, SerpAPI)

## Setup Steps

### 1. Clone and Navigate

```bash
git clone <repository-url>
cd VedaMitra
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_api_key
MISTRAL_API_KEY=your_mistral_api_key
SERPAPI_KEY=your_serpapi_key
HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

### 5. Run the Application

```bash
streamlit run app.py
```

## First-Time Usage

1. App launches and plays intro audio
2. Camera prompts for age detection (optional, for personalized responses)
3. Select response language (English, Hindi, Marathi)
4. Select knowledge domain (Yoga, Ayurveda, Ramayana, Mahabharata, General)
5. Type or speak your question in English
6. Receive answer with images and TTS audio

## Troubleshooting

- **Microphone permission**: Allow browser access for voice input
- **Camera permission**: Allow for age detection (can skip)
- **API errors**: Check `.env` configuration
- **Port already in use**: Use `streamlit run app.py --server.port 8502`