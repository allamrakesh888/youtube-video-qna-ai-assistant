import os 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

QDRANT_DIR = "qdrant_vector_db"
COLLECTION_NAME = "youtube_video_transcript_embeddings"

def get_embeddings_model():
    return OpenAIEmbeddings(model="text-embedding-3-small")


def build_vector_store(transcript : str)->QdrantVectorStore:
    print("Building vector Store")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata = {'chunk_index' : i})
        for i,chunk in enumerate(chunks)
    ]

    embeddings_model = get_embeddings_model()

    vector_store = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings_model,
        location=":memory:", #local 
        collection_name=COLLECTION_NAME
    )

    return vector_store


def load_vector_store()-> QdrantVectorStore:
    embeddings_model = get_embeddings_model()

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings_model,
        collection_name=COLLECTION_NAME,
        url=os.getenv("QDRANT_URL")
    )

    return vector_store


def get_retriever(vector_store : QdrantVectorStore, k :int = 4):
    return vector_store.as_retriever(
        search_type = 'similarity',
        search_kwargs = {"k":k}
    )