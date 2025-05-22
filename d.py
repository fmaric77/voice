import yt_dlp
from pydub import AudioSegment
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
from speechbrain.pretrained.interfaces import foreign_class

def download_and_extract_audio(video_url, output_audio_path="audio.wav"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'temp_audio.%(ext)s',
        'quiet': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    for ext in ['wav', 'mp3', 'm4a', 'webm']:
        fname = f"temp_audio.{ext}"
        if os.path.exists(fname):
            if ext != 'wav':
                audio = AudioSegment.from_file(fname)
                audio.export(output_audio_path, format="wav")
                os.remove(fname)
            else:
                os.rename(fname, output_audio_path)
            return output_audio_path
    raise FileNotFoundError("Audio extraction failed.")

def debug_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    plt.figure(figsize=(10, 2))
    plt.plot(np.linspace(0, len(y)/sr, num=len(y)), y)
    plt.title('Extracted Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

# Load the SpeechBrain English accent classifier
accent_classifier = foreign_class(
    source="Jzuluaga/accent-id-commonaccent_xlsr-en-english",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier"
)

def analyze_accent(audio_path):
    # The classifier expects a path to a wav file
    out_prob, score, index, text_lab = accent_classifier.classify_file(audio_path)
    accent = text_lab[0] if isinstance(text_lab, list) else text_lab
    confidence = float(score[0]) if hasattr(score, '__getitem__') else float(score)
    summary = f"Detected accent: {accent} with confidence {confidence:.2f}."
    return accent, confidence, summary

if __name__ == "__main__":
    video_url = input("Enter public video URL: ")
    audio_path = download_and_extract_audio(video_url)
    # debug_audio(audio_path)  # Uncomment to listen and plot
    accent, confidence, summary = analyze_accent(audio_path)
    print(f"Accent: {accent}")
    print(f"English Accent Confidence: {confidence}%")
    print(f"Summary: {summary}")