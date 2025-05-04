import streamlit as st
from moviepy import VideoFileClip
from pydub import AudioSegment
import tempfile
import whisper
from transformers import pipeline as hf_pipeline
from io import BytesIO

def process_video_to_blog(video_file):
    # Save and extract audio from video
    video_bytes = video_file.read()
    video_io = BytesIO(video_bytes)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_vid:
        temp_vid.write(video_io.read())
        temp_vid_path = temp_vid.name

    clip = VideoFileClip(temp_vid_path)
    audio = clip.audio

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        audio.write_audiofile(temp_audio.name)
        audio_path = temp_audio.name

    # Convert audio to mono, 16-bit
    audio_segment = AudioSegment.from_file(audio_path)
    audio_segment = audio_segment.set_channels(1).set_sample_width(2)
    audio_segment.export(audio_path, format="wav")

    clip.close()
    audio.close()

    # Step 1: Transcribe using Whisper
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    transcript = result["text"]

    # Step 2: Summarize transcript
    summarizer = hf_pipeline("text-generation", model="Phi-3.5-mini-instruct", device_map="auto", torch_dtype="auto")
    summary_prompt = f"Summarize the following transcript for a blog post:\n\n{transcript}\n\nSummary:"
    summary_result = summarizer(summary_prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9)
    summary = summary_result[0]["generated_text"].split("Summary:")[-1].strip()

    # Step 3: Generate blog post from summary
    blog_prompt = f"""You are a blog writer. Use the following summary to write a detailed blog post.

    Summary:
    {summary}

    The blog post should include:
    - A compelling title (start with "# Title")
    - An introduction
    - Two or more sections discussing the main points
    - A conclusion

    Start writing the blog below:
    """
    blog_result = summarizer(blog_prompt, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.95)
    blog_text = blog_result[0]["generated_text"]
    start = blog_text.find("# Title")
    blog_post = blog_text[start:] if start != -1 else blog_text

    return blog_post
