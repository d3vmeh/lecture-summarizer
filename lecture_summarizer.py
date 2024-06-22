from faster_whisper import WhisperModel
import os
import requests
from pydub import AudioSegment
import math

api_key = os.getenv("OPENAI_API_KEY")


def get_transcription_from_audio(audio_path, model_size = "base"):
   # Run on GPU with FP16
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    segments, info = model.transcribe(audio_path, beam_size=5)

    #for s in segments:
      #print(s.text)
    transcriptions_list = []
    transcription_chunk = ""

    # for s in segments:
    #     transcription_chunk += s.text
    #     if len(transcription_chunk) > 2000:
    #         transcriptions_list.append(transcription_chunk)
    #         transcription_chunk = ""
    transcription = ' '.join([segment.text for segment in segments])
    return transcription, segments

def get_summary(transcription):
    #encoded_image = encode_image(path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": f"Please summarize this transcript, you can use bullet points if you need to: {transcription}"},
        ]
    }

    payload = {
        "model": "gpt-4o",
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

audiofile_path = os.listdir("audio/")[-1]

audio_path = "./audio/audio.mp3"
#split_mp3(audio_path)

audio_size_mb = int(os.stat(audio_path).st_size/(1024**2))

transcriptions = []

count = 0
num_files = 1
if audio_size_mb > 25:
    print("Audio file is too large. Will be split into chunks")
    num_chunks = math.ceil(audio_size_mb/25)
    split_mp3(audio_path, num_chunks)
    audio_path  = f"./audio/chunks/chunk{count}.mp3"
    num_files = num_chunks





for i in range(num_files):
    print(f"transcribing: {i}")
    transcription, segments =  get_transcription_from_audio(f"./audio/chunks/chunk{i}.mp3", model_size= "tiny")
    transcriptions.append(transcription)
    count += 1


#for t in transcriptions:
#    print(t)

f = open("transcription.txt", "w")
f.write(transcription)
f.close()


print("transcription completed")

# exit()
response = get_summary(transcription)

response_text = response["choices"][0]["message"]["content"]

print("here is a summary:\n\n",response_text)
