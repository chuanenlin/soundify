# from __future__ import unicode_literals
import streamlit as st
from pytube import YouTube
# import pafy
import cv2
from PIL import Image
import clip as openai_clip
import torch
import math
import numpy as np
import plotly.express as px
import datetime
# import youtube_dl
# import os.path
import SessionState

@st.cache()
def fetch_video(url):
  streams = YouTube(url).streams.filter(adaptive=True, subtype="mp4", resolution="360p", only_video=True)
  if len(streams) == 0:
    raise "No suitable stream found for this YouTube video!"
  type(streams[0].url)
  video = streams[0].url
  return video

@st.cache()
def extract_frames(video):
  # The frame images will be stored in video_frames
  frames = []

  # Open the video file
  capture = cv2.VideoCapture(video)
  # capture = cv2.VideoCapture("video.mkv")
  fps = capture.get(cv2.CAP_PROP_FPS)

  current_frame = 0
  while capture.isOpened():
    # Read the current frame
    ret, frame = capture.read()

    # Convert it to a PIL image (required for CLIP) and store it
    if ret == True:
      frames.append(Image.fromarray(frame[:, :, ::-1]))
    else:
      break

    # Skip N frames
    current_frame += N
    capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

  # Print some statistics
  print(f"Frames extracted: {len(frames)}")

  return frames, fps

@st.cache()
def encode_frames(video_frames):
  # You can try tuning the batch size for very large videos, but it should usually be OK
  batch_size = 256
  batches = math.ceil(len(video_frames) / batch_size)

  # The encoded features will bs stored in video_features
  video_features = torch.empty([0, 512], dtype=torch.float16).to(device)

  # Process each batch
  for i in range(batches):
    print(f"Processing batch {i+1}/{batches}")

    # Get the relevant frames
    batch_frames = video_frames[i*batch_size : (i+1)*batch_size]

    # Preprocess the images for the batch
    batch_preprocessed = torch.stack([preprocess(frame) for frame in batch_frames]).to(device)

    # Encode with CLIP and normalize
    with torch.no_grad():
      batch_features = model.encode_image(batch_preprocessed)
      batch_features /= batch_features.norm(dim=-1, keepdim=True)

    # Append the batch to the list containing all features
    video_features = torch.cat((video_features, batch_features))

  # Print some stats
  print(f"Features: {video_features.shape}")
  return video_features

def text_search(search_query, display_results_count=3):
  with torch.no_grad():
    text_features = model.encode_text(openai_clip.tokenize(search_query).to(device))
    text_features /= text_features.norm(dim=-1, keepdim=True)
  similarities = (100.0 * ss.video_features @ text_features.T)
  values, best_photo_idx = similarities.topk(display_results_count, dim=0)
  for frame_id in best_photo_idx:
    st.image(ss.video_frames[frame_id])
    seconds = round(frame_id.cpu().numpy()[0] * N / ss.fps)
    time = datetime.timedelta(seconds=seconds)
    st.write("[" + str(time) + "](" + url + "&t=" + str(seconds) + "s)")

def img_search(search_query, display_results_count=3):
  with torch.no_grad():
    image_features = model.encode_image(preprocess(Image.open(search_query)).unsqueeze(0).to(device))
    image_features /= image_features.norm(dim=-1, keepdim=True)
  similarities = (100.0 * ss.video_features @ image_features.T)
  values, best_photo_idx = similarities.topk(display_results_count, dim=0)
  for frame_id in best_photo_idx:
    st.image(ss.video_frames[frame_id])
    seconds = round(frame_id.cpu().numpy()[0] * N / ss.fps)
    time = datetime.timedelta(seconds=seconds)
    st.write("[" + str(time) + "](" + url + "&t=" + str(seconds) + "s)")

def text_and_img_search(text_search_query, image_search_query, display_results_count=3):
  with torch.no_grad():
    image_features = model.encode_image(preprocess(Image.open(image_search_query)).unsqueeze(0).to(device))
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features = model.encode_text(openai_clip.tokenize(text_search_query).to(device))
    text_features /= text_features.norm(dim=-1, keepdim=True)
    hybrid_features = image_features + text_features
  similarities = (100.0 * ss.video_features @ hybrid_features.T)
  values, best_photo_idx = similarities.topk(display_results_count, dim=0)
  for frame_id in best_photo_idx:
    st.image(ss.video_frames[frame_id])
    seconds = round(frame_id.cpu().numpy()[0] * N / ss.fps)
    time = datetime.timedelta(seconds=seconds)
    st.write("[" + str(time) + "](" + url + "&t=" + str(seconds) + "s)")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            * {font-family: Avenir;}
            h1 {text-align: center;}
            .css-h2raq8 a {text-decoration: none;}
            .st-ba {font-family: Avenir;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

ss = SessionState.get(device=None, model=None, preprocess=None, video=None, video_frames=None, video_features=None, fps=None, mode=None, query=None, progress=1)

st.title("Which Frame?")

st.write("Given a video, do a **semantic** search. Which frame has a person wearing sunglasses? Which frame has cityscapes at night? Try searching with **text**, **image**, or a combined **text + image**.")

url = st.text_input("YouTube Video URL (Example: https://www.youtube.com/watch?v=bqSY4MSvFc8)")
N = 30

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = openai_clip.load("ViT-B/32", device=device)

if st.button("Process video"):
  ss.video = fetch_video(url)
  print("Downloaded video")
  ss.video_frames, ss.fps = extract_frames(ss.video)
  print("Extracted frames")
  ss.video_features = encode_frames(ss.video_frames)
  print("Encoded frames")
  ss.progress = 2

if ss.progress == 2:
  ss.mode = st.selectbox("How would you like to search?",("Text", "Image", "Text + Image"))
  if ss.mode == "Text":
    ss.query = st.text_input("Enter text query (Example: big ben at dawn)")
  elif ss.mode == "Image":
    ss.query = st.file_uploader("Upload image query")
  else:
    ss.text_query = st.text_input("Enter text query")
    ss.img_query = st.file_uploader("Upload image query")

  if st.button("Submit"):
    if ss.mode == "Text":
      if ss.query is not None:
        text_search(ss.query)
    elif ss.mode == "Image":
      if ss.query is not None:
        img_search(ss.query)
    else:
      if ss.text_query is not None and ss.img_query is not None:
        text_and_img_search(ss.text_query, ss.img_query)

st.write("This fun experiment was put together by [David](https://chuanenlin.com) at Carnegie Mellon University. The querying is powered by [OpenAI's CLIP neural network](https://openai.com/blog/clip) and the interface with [Streamlit](https://streamlit.io). Many aspects of this project are based on the kind work of [Vladimir Haltakov](https://haltakov.net) and [Haofan Wang](https://haofanwang.github.io).")
