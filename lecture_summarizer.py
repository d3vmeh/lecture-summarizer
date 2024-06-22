from faster_whisper import WhisperModel


def get_transcription_from_audio(audio_path, model_size = "base"):
   # Run on GPU with FP16
   model = WhisperModel(model_size, device="cuda", compute_type="float16")
   segments, info = model.transcribe(audio_path, beam_size=5)

   transcription = ' '.join([segment.text for segment in segments])
   return transcription

