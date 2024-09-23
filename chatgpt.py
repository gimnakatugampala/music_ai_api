
import assemblyai as aai

aai.settings.api_key = "e55c7255e11a408ebecbdefa28600f36"
transcriber = aai.Transcriber()

transcript = transcriber.transcribe("https://cdn1.suno.ai/08d5f041-f46b-4afd-9355-b738282b08f3.mp3")
# transcript = transcriber.transcribe("./my-local-audio-file.wav")

print(transcript.text)