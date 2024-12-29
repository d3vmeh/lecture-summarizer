import os
import requests
from pydub import AudioSegment
import math
import time
import sys
from database import *
import streamlit as st
import yt_dlp


from ui_chat import load_chat_history, save_chat_history

from langchain_chroma import Chroma

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.llms.ollama import Ollama

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import ConversationChain
from langchain.memory.summary import ConversationSummaryMemory
from langchain_core.messages import HumanMessage, AIMessage

api_key = os.getenv("OPENAI_API_KEY")

def get_transcription_from_youtube(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': './audio/audio.%(ext)s',
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return "./audio/audio.mp3"
    except Exception as e:
        print(f"Error downloading Youtube video: {e}")
        return None


def get_response(context, question, llm):
    #encoded_image = encode_image(path)

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
    #response_text = model.invoke(prompt)
    response_text = chain.invoke(question)
    return response_text
   
file = open("transcription.txt", "r")
transcription = file.read()
file.close()

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
llm = Ollama(model="llama3.2",temperature=0.5)

conversation = ConversationChain(llm = llm, memory = ConversationSummaryMemory(llm=llm))


#response = get_response(transcription, "Please summarize this transcript", llm)

#response_text = response["choices"][0]["message"]["content"]
embeddings = OpenAIEmbeddings()
#chunks = load_and_split()
#save_database(embeddings, chunks)
db = load_database(embeddings)
print("Ready to answer questions")

st.title("Chat with Video Transcript")

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()


with st.sidebar:
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        save_chat_history([])

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        context = query_database(prompt,db)
        print(context)
        full_response = get_response(context,prompt,llm)
        message_placeholder.markdown(full_response)   
    st.session_state.messages.append({"role": "assistant", "content": full_response})

save_chat_history(st.session_state.messages)






