from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate

import os

def load_transcription():
    file = open("transcription.txt", "r")
    transcription = file.read()
    file.close()
    document = [Document(transcription)]
    return document

def load_and_split(transcription):
    document = load_transcription()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 60, length_function = len, is_separator_regex  = False)
    return text_splitter.split_documents(document)

def load_and_split_from_youtube(transcription):
    document = [Document(transcription)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 60, length_function = len, is_separator_regex  = False)
    return text_splitter.split_documents(document)

def save_database(embeddings, chunks, path="Chroma"):    
    database = Chroma.from_documents(chunks,embeddings,persist_directory=path)
    database.persist()
    print(f"Saved {len(chunks)} chunks to Chroma")

def load_database(embeddings, path="DBs"):
    database = Chroma(persist_directory=path,embedding_function=embeddings)
    return database


def query_database(query, database, num_responses = 10, similarity_threshold = 0.5):
    results = database.similarity_search_with_relevance_scores(query,k=num_responses)
    try:
        if results[0][1] < similarity_threshold:
            print("Could not find results")
    except:
        print("Error")
    return results