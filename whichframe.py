import streamlit as st
from pytube import YouTube
from pytube import extract
import cv2
from PIL import Image
import clip as openai_clip
import torch
import math
import numpy as np
import SessionState
import tempfile
from humanfriendly import format_timespan

def fetch_video(url):
  yt = YouTube(url)
  streams = yt.streams.filter(adaptive=True, subtype="mp4", resolution="360p", only_video=True)
  length = yt.length
  if length >= 600:
    st.error("Please find a YouTube video shorter than 10 minutes. Sorry about this, my server capacity is limited for the time being.")
    st.stop()
  video = streams[0]
  return video, video.url

@st.cache()
def extract_frames(video):
  frames = []
  capture = cv2.VideoCapture(video)
  fps = capture.get(cv2.CAP_PROP_FPS)
  current_frame = 0
  while capture.isOpened():
    ret, frame = capture.read()
    if ret == True:
      frames.append(Image.fromarray(frame[:, :, ::-1]))
    else:
      break
    current_frame += N
    capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
  # print(f"Frames extracted: {len(frames)}")

  return frames, fps

@st.cache()
def encode_frames(video_frames):
  batch_size = 256
  batches = math.ceil(len(video_frames) / batch_size)
  video_features = torch.empty([0, 512], dtype=torch.float16).to(device)
  for i in range(batches):
    batch_frames = video_frames[i*batch_size : (i+1)*batch_size]
    batch_preprocessed = torch.stack([preprocess(frame) for frame in batch_frames]).to(device)
    with torch.no_grad():
      batch_features = model.encode_image(batch_preprocessed)
      batch_features /= batch_features.norm(dim=-1, keepdim=True)
    video_features = torch.cat((video_features, batch_features))
  # print(f"Features: {video_features.shape}")
  return video_features

def display_results(best_photo_idx):
  st.markdown("**Top-5 matching results**")
  for frame_id in best_photo_idx:
    st.image(ss.video_frames[frame_id])
    seconds = round(frame_id.cpu().numpy()[0] * N / ss.fps)
    # time = datetime.timedelta(seconds=seconds)
    time = format_timespan(seconds)
    if ss.input == "file":
      st.write("Seen at " + str(time))
    else:
      st.markdown("Seen at [" + str(time) + "](" + url + "&t=" + str(seconds) + "s)")

def text_search(search_query, display_results_count=5):
  with torch.no_grad():
    text_features = model.encode_text(openai_clip.tokenize(search_query).to(device))
    text_features /= text_features.norm(dim=-1, keepdim=True)
  similarities = (100.0 * ss.video_features @ text_features.T)
  values, best_photo_idx = similarities.topk(display_results_count, dim=0)
  display_results(best_photo_idx)

def img_search(search_query, display_results_count=5):
  with torch.no_grad():
    image_features = model.encode_image(preprocess(Image.open(search_query)).unsqueeze(0).to(device))
    image_features /= image_features.norm(dim=-1, keepdim=True)
  similarities = (100.0 * ss.video_features @ image_features.T)
  values, best_photo_idx = similarities.topk(display_results_count, dim=0)
  display_results(best_photo_idx)

def text_and_img_search(text_search_query, image_search_query, display_results_count=5):
  with torch.no_grad():
    image_features = model.encode_image(preprocess(Image.open(image_search_query)).unsqueeze(0).to(device))
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features = model.encode_text(openai_clip.tokenize(text_search_query).to(device))
    text_features /= text_features.norm(dim=-1, keepdim=True)
    hybrid_features = image_features + text_features
  similarities = (100.0 * ss.video_features @ hybrid_features.T)
  values, best_photo_idx = similarities.topk(display_results_count, dim=0)
  display_results(best_photo_idx)

st.set_page_config(page_title="Which Frame?", page_icon = "🎥", layout = "centered", initial_sidebar_state = "collapsed")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            * {font-family: Avenir;}
            .css-gma2qf {display: flex; justify-content: center; font-size: 42px; font-weight: bold;}
            a:link {text-decoration: none;}
            a:hover {text-decoration: none;}
            .st-ba {font-family: Avenir;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
# text-align: center;
ss = SessionState.get(url=None, input=None, video=None, video_name=None, video_frames=None, video_features=None, fps=None, mode=None, query=None, progress=1)

st.title("Which Frame?")

st.markdown("Given a video, do a **semantic** search. Which frame has a person wearing sunglasses? Which frame has bright cityscapes at night? Search with **text**, **image**, or a combined **text + image**.")

video_file = st.file_uploader("Upload a video", type=["mp4"])
url = st.text_input("or link to a YouTube video (Example: https://www.youtube.com/watch?v=BapSQFJPMM0)")

N = 30

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = openai_clip.load("ViT-B/32", device=device)

if st.button("Process video (this may take a while)"):
  ss.progress = 1
  ss.video_start_time = 0
  if video_file:
    ss.input = "file"
    ss.video = video_file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    ss.video_name = tfile.name
  elif url:
    ss.input = "link"
    ss.video, ss.video_name = fetch_video(url)
    id = extract.video_id(url)
    ss.url = "https://www.youtube.com/watch?v=" + id
  else:
    st.error("Please upload a video or link to a valid YouTube video")
    st.stop()
  # print("Downloaded video")
  ss.video_frames, ss.fps = extract_frames(ss.video_name)
  # print("Extracted frames")
  ss.video_features = encode_frames(ss.video_frames)
  # print("Encoded frames")
  ss.progress = 2

if ss.input == "file":
  st.video(ss.video)
elif ss.input == "link":
  st.video(ss.url)

if ss.progress == 2:
  ss.mode = st.selectbox("How would you like to search?",("Text", "Image", "Text + Image"))
  if ss.mode == "Text":
    ss.text_query = st.text_input("Enter text query (Example: pyramids)")
  elif ss.mode == "Image":
    ss.img_query = st.file_uploader("Upload image query", type=["png", "jpg", "jpeg"])
  else:
    ss.text_query = st.text_input("Enter text query (Example: pyramids)")
    ss.img_query = st.file_uploader("Upload image query", type=["png", "jpg", "jpeg"])

  if st.button("Submit"):
    if ss.mode == "Text":
      if ss.text_query is not None:
        text_search(ss.text_query)
    elif ss.mode == "Image":
      if ss.img_query is not None:
        img_search(ss.img_query)
    else:
      if ss.text_query is not None and ss.img_query is not None:
        text_and_img_search(ss.text_query, ss.img_query)

st.markdown("This fun experiment was put together by [David](https://chuanenlin.com) at Carnegie Mellon University. The querying is powered by [OpenAI's CLIP neural network](https://openai.com/blog/clip) and the interface was built with [Streamlit](https://streamlit.io). Many aspects of this project are based on the kind work of [Vladimir Haltakov](https://haltakov.net) and [Haofan Wang](https://haofanwang.github.io).")
