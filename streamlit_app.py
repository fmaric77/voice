import streamlit as st
import os
from d import download_and_extract_audio, analyze_accent

st.title("English Accent Classifier")
st.write("""
Upload a public video URL (e.g., YouTube, Loom, or direct MP4 link). The tool will extract the audio, analyze the speaker’s accent, and provide a confidence score.
""")

video_url = st.text_input("Enter public video URL:")

if st.button("Analyze Accent") and video_url:
    with st.spinner("Downloading and extracting audio..."):
        try:
            audio_path = download_and_extract_audio(video_url)
        except Exception as e:
            st.error(f"Audio extraction failed: {e}")
            st.stop()
    st.success("Audio extracted successfully!")
    st.audio(audio_path)
    with st.spinner("Analyzing accent (downloading model if needed)..."):
        try:
            accent, confidence, summary = analyze_accent(audio_path)
        except Exception as e:
            st.error(f"Accent analysis failed: {e}")
            st.stop()
    st.markdown(f"**Accent:** {accent}")
    st.markdown(f"**English Accent Confidence:** {confidence:.2f}%")
    st.markdown(f"**Summary:** {summary}")
