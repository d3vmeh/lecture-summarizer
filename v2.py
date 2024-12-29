import os
import requests
from pydub import AudioSegment
import math
import time
import sys
from database import *
import streamlit as st


from ui_chat import load_chat_history, save_chat_history

from langchain_chroma import Chroma

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.llms.ollama import Ollama

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import ConversationChain
from langchain.memory.summary import ConversationSummaryMemory
from langchain_core.messages import HumanMessage, AIMessage

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter



api_key = os.getenv("OPENAI_API_KEY")

def get_youtube_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        formatter = TextFormatter()
        plain_text_transcript = formatter.format_transcript(transcript)
        return plain_text_transcript
    
    except Exception as e:
        return f"An error occurred: {e}"


def get_response(context, question, llm):
    prompt = ChatPromptTemplate.from_messages(
        [
        (
            "system",
            "You are an experienced advisor and international diplomat who is assisting the US government in foreign policy. You use natural language "
         "to answer questions based on structured data, unstructured data, and community summaries. You are thoughtful and thorough in your responses."
        ),
        (
            "user",
            """Answer the question only based on the following context:
            {context}


            Here is the question:
            {question}"""
        ),
        ]
        )
    
    chain = (
         {"context": lambda x: context, "question": RunnablePassthrough()}
         | prompt
         | llm
         | StrOutputParser()
    )
    response_text = chain.invoke(question)
    return response_text

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)

conversation = ConversationChain(llm = llm, memory = ConversationSummaryMemory(llm=llm))
embeddings = OpenAIEmbeddings()

st.title("Chat with Video Transcript")

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()



with st.sidebar:
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        save_chat_history([])

    url = st.text_input("Enter a Youtube URL", value="")
    if st.button("Get Transcription"):
        id = url.split("=")[1]
        transcription = get_youtube_transcript(id)
        st.write(transcription)
        chunks = load_and_split_from_youtube(transcription)
        save_youtube_database(embeddings, chunks)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        db = load_youtube_database(embeddings)
        context = query_database(prompt,db)
        print(context)
        full_response = get_response(context,prompt,llm)
        message_placeholder.markdown(full_response)   
    st.session_state.messages.append({"role": "assistant", "content": full_response})

save_chat_history(st.session_state.messages)






