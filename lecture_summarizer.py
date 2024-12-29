from faster_whisper import WhisperModel
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

api_key = os.getenv("OPENAI_API_KEY")


def get_transcription_from_audio(audio_path, model_size = "base"):
   # Run on GPU with FP16
    model = WhisperModel(model_size, device="cpu", compute_type="int8_float16")
    segments, info = model.transcribe(audio_path, beam_size=5)

    #for s in segments:
    #  print(s.text)
    transcriptions_list = []
    transcription_chunk = ""

    # for s in segments:
    #     transcription_chunk += s.text
    #     if len(transcription_chunk) > 2000:
    #         transcriptions_list.append(transcription_chunk)
    #         transcription_chunk = ""
    transcription = ' '.join([segment.text for segment in segments])
    return transcription, segments

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
    

def split_mp3(audio_path, chunks):#, chunk_length_ms=30000):  # chunk_length_ms is in milliseconds
    audio = AudioSegment.from_mp3(audio_path)
    #chunks = math.ceil(len(audio) / chunk_length_ms)
    chunk_length_ms = len(audio)/chunks
    # Create a directory to store the chunks if it doesn't exist
    chunk_dir = os.path.join(os.path.dirname(audio_path), "chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    for i in range(chunks):
        start = i * chunk_length_ms
        end = start + chunk_length_ms
        chunk = audio[start:end]
        chunk_name = f"chunk{i}.mp3"
        chunk_path = os.path.join(chunk_dir, chunk_name)
        chunk.export(chunk_path, format="mp3")
        print(f"Exported {chunk_path}")

def transcribe_file():
    audio_files= os.listdir("audio/")
    audio_paths = []
    for f in audio_files:
        if "mp3" in f:
            audio_path = f"./audio/{f}"
            audio_paths.append(audio_path)
    #audio_path = "./audio/audio.mp3"
    #split_mp3(audio_path)
    audio_size_mb = int(os.stat(audio_path).st_size/(1024**2))
    transcriptions = []
    num_files = 1
    count = 0
    if audio_size_mb > 5:
        print("Audio file is too large. Will be split into chunks")
        num_chunks = math.ceil(audio_size_mb/5)
        split_mp3(audio_path, num_chunks)
        audio_path  = f"./audio/chunks/chunk{count}.mp3"
        num_files = num_chunks

        for i in range(num_files):
            print(f"transcribing: {i}")
            transcription, segments =  get_transcription_from_audio(f"./audio/chunks/chunk{i}.mp3", model_size= "tiny")
            transcriptions.append(transcription)
            print(f"transcription {i} completed")
            time.sleep(2)

        print(len(transcriptions))
        for f in os.listdir("./audio/chunks"):
            os.remove(f"./audio/chunks/{f}")
            print("removed",f)
        os.rmdir("./audio/chunks")
        print("all chunks removed")


    else:
        audio_path = f"./audio/audio.mp3"
        transcription, segments = get_transcription_from_audio(f"./audio/audio.mp3", model_size= "medium.en")
        transcriptions.append(transcription)

    f = open("transcription.txt", "w")
    f.close()
    for t in transcriptions:
        f = open("transcription.txt", "a")
        f.write(t)
        f.close()


    print("transcription completed")


transcribe_file()


file = open("transcription.txt", "r")
transcription = file.read()
file.close()

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
llm = Ollama(model="llama3.1",temperature=0.5)

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






