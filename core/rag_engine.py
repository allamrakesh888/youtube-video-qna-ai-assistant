import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever

rag_system_msg_prompt = """You are an expert youtube video assistant. Answer the user's question 
            based ONLY on the youtube video transcript context provided below.

            If the answer is not found in the context, say: 
            "I could not find this information in the youtube video transcript."

            If the answer is not found in the context but the intent of the question is to summarize the youtube video, say: 
            "summarize_video"  

            Always be concise and precise. If quoting someone, mention it clearly.

            Context from youtube video transcript:
            {context}"""


def get_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0.5, api_key=os.getenv("OPENAI_API_KEY"))


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def build_rag_chain(transcript:str):

    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store, k = 4)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [( "system", rag_system_msg_prompt),
         ("human", "{question}")
        ])

    #full LCEL Rag pipeline 
    rag_chain = (
        {"context" : retriever | RunnableLambda(format_docs),
         "question": RunnablePassthrough()
         }
         |prompt|llm|StrOutputParser()
    )

    return rag_chain
    

def ask_question(rag_chain, question:str) -> str:
    answer = rag_chain.invoke(question)
    return answer