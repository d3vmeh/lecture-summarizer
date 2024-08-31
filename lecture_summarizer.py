from faster_whisper import WhisperModel
import os
import requests
from pydub import AudioSegment
import math
import time
import sys
from database import *
from langchain_openai import OpenAIEmbeddings


api_key = os.getenv("OPENAI_API_KEY")
sys.path.append('/Users/devm2/Downloads/FFMPEP_DONT_DELETE/ffmpeg')


def get_transcription_from_audio(audio_path, model_size = "base"):
   # Run on GPU with FP16
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
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

def get_response(context, instructions):
    #encoded_image = encode_image(path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": f"{instructions} Use the following context to answer the user: {context} "},
        ]
    }

    payload = {
        "model": "gpt-4o-mini",
        "temperature": 0.5,
        "messages": [message],
        "max_tokens": 800
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()

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
        transcription, segments =  get_transcription_from_audio(f"./audio/chunks/chunk{i}.mp3", model_size= "medium.en")
        transcriptions.append(transcription)
        print(f"transcription {i} completed")

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


#for t in transcriptions:
#    print(t)
f = open("transcription.txt", "w")
f.close()
for t in transcriptions:
    f = open("transcription.txt", "a")
    f.write(t)
    f.close()


print("transcription completed")

# exit()
response = get_response(transcription, "Please summarize this transcript")

response_text = response["choices"][0]["message"]["content"]
embeddings = OpenAIEmbeddings()

chunks = load_and_split()
save_database(embeddings, chunks)


print("here is a summary:\n\n",response_text)
db = load_database(embeddings)
while True:
    q = input("What would you like to ask? ")

    if q.lower() == 'q':
        exit()

    context = query_database(q, db)
    response = get_response(context, q)
    response_text = response["choices"][0]["message"]["content"]
    print(response_text)
    print("\n\n\n")

