from dotenv import load_dotenv
from core.audio_processor import fetch_and_chunk
from core.transcriber import transcribe_all
from core.summarizer import generate_summary, generate_title
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

def run_pipeline(video_url :str) -> dict:
    print("Starting Youtube Video AI Assistant...")

    chunks = fetch_and_chunk(video_url)

    transcript = transcribe_all(chunks)
    print(f"raw transcription [first 300 characters ] {transcript[:300]}")

    title = generate_title(transcript)

    summary = generate_summary(transcript)
    
    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "rag_chain": rag_chain,
    }

if __name__ == "__main__":
    # CLI entry point
    video_url = input("Enter YouTube URL : ").strip()
    result = run_pipeline(video_url)

    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print("=" * 60)

    # Phase 2 — Chat with your youtube video via RAG
    print("\n💬 Chat with your youtube video (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)

        if "summarize_video" in answer.lower():
            answer = result['summary']
        print(f"\n🤖 Assistant: {answer}\n")