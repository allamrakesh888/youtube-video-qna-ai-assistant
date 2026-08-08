# YouTube Video AI Assistant

An AI-powered assistant that takes any YouTube video, transcribes it, generates a title and summary, and lets you **chat with the video** using Retrieval-Augmented Generation (RAG).

Available both as a **CLI tool** and as a **Streamlit web app**.

---

## Features

- **Audio extraction & chunking** — Downloads audio from a YouTube URL and splits it into manageable chunks.
- **Transcription** — Converts audio to text locally using [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) (English-only videos supported currently).
- **Title & Summary generation** — Uses an LLM (`gpt-4o` via LangChain) to generate a short title and a bullet-point map-reduce summary of the transcript.
- **RAG-powered chat** — Embeds the transcript into a Qdrant vector store and lets you ask natural-language questions about the video, grounded strictly in its content.
- **Two interfaces**:
  - `main.py` — a terminal-based CLI pipeline
  - `webinterface.py` — a polished Streamlit chat UI with tabs for Chat, Summary, and Full Transcript

---

## How it works

```
YouTube URL
   │
   ▼
audio_processor.py   → download & chunk audio
   │
   ▼
transcriber.py        → transcribe chunks with faster-whisper
   │
   ▼
summarizer.py          → generate title + summary (LLM)
   │
   ▼
vector_store.py        → embed transcript into Qdrant
   │
   ▼
rag_engine.py           → LCEL RAG chain for Q&A
```

---

## Setup

### 1. Prerequisites

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) installed and available on your system `PATH` (required by `yt-dlp` and `pydub` for audio extraction/conversion)
- An [OpenAI API key](https://platform.openai.com/api-keys)

### 2. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 3. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```


### 5. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here

# Optional
WHISPER_MODEL=small          # tiny | base | small | medium | large-v3
QDRANT_URL=                  # only needed if using load_vector_store() with a remote Qdrant instance
```

By default, the app builds an **in-memory Qdrant vector store**, so no external Qdrant server is required to get started.

---

## Usage

### Option A — Command Line Interface

```bash
python main.py
```

You'll be prompted for a YouTube URL. Once processed, you'll see the generated title and summary, followed by an interactive chat prompt:

```
Enter YouTube URL: https://www.youtube.com/watch?v=...
📌 Title: ...
📋 Summary: ...

💬 Chat with your youtube video (type 'exit' to quit)
You: What did they say about X?
🤖 Assistant: ...
```

### Option B — Streamlit Web App

```bash
streamlit run webinterface.py
```

Then, in the browser UI:
1. Paste a YouTube URL in the sidebar
2. Click ** Process video**
3. Explore the **Chat**, **Summary**, and **Full Transcript** tabs

---

## Tech Stack

- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — YouTube audio download
- [pydub](https://github.com/jiaaro/pydub) — Audio chunking
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Speech-to-text
- [LangChain](https://www.langchain.com/) — LLM orchestration & RAG pipeline
- [Qdrant](https://qdrant.tech/) (via `langchain-qdrant`) — Vector store
- [OpenAI](https://platform.openai.com/) — LLM (`gpt-4o`) & embeddings
- [Streamlit](https://streamlit.io/) — Web UI

---
