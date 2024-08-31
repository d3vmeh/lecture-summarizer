from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate

import os

def load_transcription():
    file = open("transcription.txt", "r")
    transcription = file.read()
    file.close()
    return transcription

def load_and_split():
    transcription = load_transcription()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 60, length_function = len, is_separator_regex  = False)
    return text_splitter.split_documents(transcription)

def save_database(embeddings, chunks, path="lecture-summarizer/Chroma"):    
    database = Chroma.from_documents(chunks,embeddings,persist_directory=path)
    database.persist()
    print(f"Saved {len(chunks)} chunks to Chroma")

def load_database(embeddings, path="standard-rag-foreign-policy/Chroma"):
    database = Chroma(persist_directory=path,embedding_function=embeddings)
    return database