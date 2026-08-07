"""
AI YouTube Video Assistant — Streamlit UI
------------------------------------------
Wraps the existing pipeline (audio_processor -> transcriber -> summarizer -> rag_engine)
in a chat-style web interface.

Run with:
    streamlit run webinterface.py
"""

import streamlit as st
from dotenv import load_dotenv

from core.audio_processor import fetch_and_chunk
from core.transcriber import transcribe_all
from core.summarizer import generate_summary, generate_title
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="AI YouTube Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Minimal styling polish
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; }
        .video-title { font-size: 1.4rem; font-weight: 700; margin-bottom: 0.2rem; }
        .stChatMessage { border-radius: 12px; }
        .status-pill {
            display: inline-block; padding: 2px 10px; border-radius: 999px;
            background-color: #16a34a22; color: #16a34a; font-size: 0.8rem;
            font-weight: 600; margin-left: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
defaults = {
    "processed": False,
    "processing": False,
    "title": None,
    "transcript": None,
    "summary": None,
    "rag_chain": None,
    "chat_history": [],   # list of (role, content)
    "source": "",
    "language": "english",
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


def reset_session():
    for key, val in defaults.items():
        st.session_state[key] = val


# --------------------------------------------------------------------------
# Sidebar — input & controls
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎬 AI YouTube Video Assistant")
    st.caption("Summarize any YouTube video and chat with it, powered by RAG + Qdrant vec db.")

    st.divider()

    source = st.text_input(
        "YouTube video URL path",
        value=st.session_state.source,
        placeholder="https://www.youtube.com/watch?v=...",
        disabled=st.session_state.processing,
    )

    language = st.selectbox(
        "Transcription language",
        options=["english"],
        index=["english"].index(st.session_state.language),
        disabled=st.session_state.processing,
    )

    process_clicked = st.button(
        "🚀 Process video",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.processing or not source.strip(),
    )

    if st.session_state.processed:
        st.button(
            "🔄 Start over with a new video",
            use_container_width=True,
            on_click=reset_session,
        )

    st.divider()
    with st.expander("ℹ️ How it works"):
        st.markdown(
            "1. **Fetch & chunk** audio from the video\n"
            "2. **Transcribe** using Whisper\n"
            "3. **Summarize** transcript & generate a title\n"
            "4. **Embed & index** transcript into Qdrant vec db\n"
            "5. **Chat** with the video using RAG + vec db"
        )

# --------------------------------------------------------------------------
# Pipeline execution
# --------------------------------------------------------------------------
if process_clicked and source.strip():
    reset_session()
    st.session_state.source = source.strip()
    st.session_state.language = language
    st.session_state.processing = True
    st.rerun()

if st.session_state.processing and not st.session_state.processed:
    status_box = st.empty()
    with status_box.container():
        with st.status("Processing video…", expanded=True) as status:
            try:
                st.write("📥 Fetching and chunking audio…")
                chunks = fetch_and_chunk(st.session_state.source)

                st.write("📝 Transcribing audio…")
                transcript = transcribe_all(chunks)

                st.write("🏷️ Generating title…")
                title = generate_title(transcript)

                st.write("🧾 Summarizing transcript…")
                summary = generate_summary(transcript)

                st.write("📦 Building vector index (Qdrant) for RAG…")
                rag_chain = build_rag_chain(transcript)

                st.session_state.title = title
                st.session_state.transcript = transcript
                st.session_state.summary = summary
                st.session_state.rag_chain = rag_chain
                st.session_state.processed = True
                st.session_state.processing = False

                status.update(label="✅ Video processed successfully!", state="complete")
            except Exception as e:
                st.session_state.processing = False
                status.update(label="❌ Processing failed", state="error")
                st.exception(e)
                st.stop()
    st.rerun()

# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
if not st.session_state.processed:
    st.markdown("# 🎬 AI YouTube Video Assistant")
    st.markdown(
        "Paste a YouTube link in the sidebar and click **Process video** to get "
        "an AI-generated title, summary, and an interactive chat interface "
        "grounded in the video's transcript."
    )
    st.info("👈 Enter a YouTube URL in the sidebar to get started.")

else:
    st.write("  ")
    st.markdown(f"<div class='video-title'>📌 {st.session_state.title}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<span class='status-pill'>Ready</span> "
        f"<span style='color:#6b7280;font-size:0.85rem;'> · language: {st.session_state.language}</span>",
        unsafe_allow_html=True,
    )
    st.write("")

    tab_chat, tab_summary, tab_transcript = st.tabs(["💬 Chat", "📋 Summary", "📄 Full Transcript"])

    # ---------------- Summary tab ----------------
    with tab_summary:
        st.markdown(st.session_state.summary)
        st.download_button(
            "⬇️ Download summary",
            data=st.session_state.summary,
            file_name="summary.txt",
            use_container_width=False,
        )

    # ---------------- Transcript tab ----------------
    with tab_transcript:
        st.text_area("Transcript", st.session_state.transcript, height=500)
        st.download_button(
            "⬇️ Download transcript",
            data=st.session_state.transcript,
            file_name="transcript.txt",
            use_container_width=False,
        )

    # ---------------- Chat tab ----------------
    with tab_chat:
        chat_container = st.container(height=480)
        with chat_container:
            if not st.session_state.chat_history:
                st.caption("Ask anything about the video — e.g. \"Summarize this video\" or \"What did they say about X?\"")
            for role, content in st.session_state.chat_history:
                with st.chat_message(role):
                    st.markdown(content)

        question = st.chat_input("Ask a question about the video…")
        if question:
            st.session_state.chat_history.append(("user", question))
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking…"):
                        try:
                            answer = ask_question(st.session_state.rag_chain, question)
                            if "summarize_video" in answer.lower():
                                answer = st.session_state.summary
                        except Exception as e:
                            answer = f"⚠️ Something went wrong: {e}"
                    st.markdown(answer)
            st.session_state.chat_history.append(("assistant", answer))