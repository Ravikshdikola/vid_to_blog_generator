import streamlit as st
from pipeline import process_video_to_blog  # adjust import if needed

st.title("🎬 Video to Blog Generator (via Summary)")

video_file = st.file_uploader("Upload a video (.mp4)", type=["mp4"])

if video_file:
    with st.spinner("Processing..."):
        blog = process_video_to_blog(video_file)
    st.subheader("📝 Blog Post Output")
    st.markdown(blog)
