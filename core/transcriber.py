from faster_whisper import WhisperModel
import os

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
_model = None

def load_model():
    global _model  

    if _model is None: 
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = WhisperModel("small", device="cpu", compute_type="int8")
        print("Whisper model loaded.")

    return _model 

def transcribe_chunk(chunk_path: str) -> str:
    result = ""
    model = load_model()  

    segments, info = model.transcribe(chunk_path, beam_size=4)
    print(f"Detected language: {info.language} (Probability: {info.language_probability:.2f})")

    if(info.language != 'en'):
        raise Exception("only videos in english language are supported, exiting... !")

    for segment in segments:
        result += segment.text + " "

    return result 


def transcribe_all(chunks: list) -> str:

    full_transcript = "" 
    for i, chunk in enumerate(chunks):  

        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunk(chunk)  
        full_transcript += text + " "  

    print("Transcription complete.")
    return full_transcript.strip()