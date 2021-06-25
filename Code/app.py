import json
import math
import os
import sys
import clip as openai_clip
import cv2
import more_itertools
import moviepy.audio.fx.all as afx
import moviepy.editor as mpe
import numpy as np
import requests
import scipy.signal as si
import streamlit as st
import torch
from matplotlib import cm
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from natsort import natsorted
from PIL import Image
from pydub import AudioSegment, silence
from pydub.playback import play
from scipy import ndimage
from torchray.attribution.grad_cam import grad_cam
from torchvision.datasets import CIFAR100
import SessionState


@st.cache(show_spinner=False)
def load_model(model):
	return openai_clip.load(model, device=device, jit=False)


@st.cache(show_spinner=False)
def video_to_scenes(video):
	diff_threshold = 0.24
	window_size = 1
	frames = []
	cap = cv2.VideoCapture(video)
	fps = cap.get(cv2.CAP_PROP_FPS)
	# frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
	# print(0, int(frame_num - 1))
	frame_diff = []
	frame_index = []
	frame_count = 0
	while cap.isOpened():
		ret, frame = cap.read()
		if ret:
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			n_pixel = frame.shape[0] * frame.shape[1]
			hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
			hist = hist * (1.0 / n_pixel)
			frame_diff.append(hist)
			frame_index.append(frame_count)
			# frame_count += fps
			frame_count += 1
		else:
			break
		frames.append(Image.fromarray(frame[:, :, ::-1]))
		cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
	# compute the distance between this frame with previous frame
	frame_count = len(frame_diff) - 1
	while frame_count >= 1:
		frame_diff[frame_count] = np.sum(
			np.abs(np.subtract(frame_diff[frame_count], frame_diff[frame_count - 1])))
		frame_count -= 1
	frame_diff[0] = 0
	vector = np.array(frame_diff)
	indexes, _ = si.find_peaks(
		vector, height=diff_threshold, distance=window_size)
	print("Numbers of scenes: " + str(len(indexes) + 1))
	# write down the scene boundary frames' index
	# print("Boundary indexes: ")
	prev_index = 0
	scenes = []
	scene_lengths = []
	counter = 0
	for index in indexes:
		# print(str(index - 1) + ":" + str(index))
		# st.image(frames[index - 1])
		# st.image(frames[index])
		# st.write(len(frames[prev_index:index]))
		scenes.append(frames[prev_index:index][0::int(fps)])
		scene_lengths.append(index / fps - prev_index / fps)
		# ffmpeg_extract_subclip(video, prev_index, prev_index + len(frames[prev_index:index]), targetname = "temp/" + str(counter) + ".mp4")
		ffmpeg_extract_subclip(
			video, prev_index / fps, index / fps, targetname="temp/" + str(counter) + ".mp4")
		prev_index = index
		# print("prev: " + str(prev_index))
		counter += 1
	# st.write(len(frames[prev_index:len(frames)]))
	scenes.append(frames[prev_index:len(frames)][0::int(fps)])
	scene_lengths.append(len(frames) / fps - prev_index / fps)
	# ffmpeg_extract_subclip(video, prev_index, prev_index + len(frames[prev_index:len(frames)]), targetname = "temp/" + str(counter) + ".mp4")
	ffmpeg_extract_subclip(video, prev_index / fps, len(frames) /
						   fps, targetname="temp/" + str(counter) + ".mp4")
	cap.release()
	# SAVE ORIGINAL VIDEO LENGTHS
	return scenes, scene_lengths, fps


@st.cache(show_spinner=False)
def encode_scene(video_frames):
	batch_size = 256
	batches = math.ceil(len(video_frames) / batch_size)
	# video_features = torch.empty([0, 512], dtype=torch.float16).to(device)
	video_features = torch.empty([0, 1024], dtype=torch.float16).to(device)
	for i in range(batches):
		batch_frames = video_frames[i*batch_size: (i+1)*batch_size]
		batch_preprocessed = torch.stack(
			[preprocess(frame) for frame in batch_frames]).to(device)
		with torch.no_grad():
			batch_features = ss.model.encode_image(batch_preprocessed)
			batch_features /= batch_features.norm(dim=-1, keepdim=True)
		video_features = torch.cat((video_features, batch_features))
	# print(f"Features: {video_features.shape}")
	return video_features

@st.cache(show_spinner=False)
def classify_audio_for_scene(word_list, scene_features):
	# text_inputs = torch.cat([openai_clip.tokenize(f"a photo of a {word}") for word in word_list]).to(device)
	text = torch.cat([openai_clip.tokenize(
		f"{word}") for word in word_list]).to(device)
	with torch.no_grad():
		text_features = ss.model.encode_text(text)
		text_features /= text_features.norm(dim=-1, keepdim=True)
	logit_scale = ss.model.logit_scale.exp()
	similarities = (logit_scale * scene_features @
					text_features.t()).softmax(dim=-1)
	probs, word_idxs = similarities[0].topk(5)
	predicted_audio = []
	for prob, word_idx in zip(probs, word_idxs):
		# print(word_list[index] + ": " + str(100 * value.item()))
		predicted_audio.append(word_list[word_idx])
	# print("==========")
	return predicted_audio

def sort_ambient_predictions(ambient_predictions, salient_predictions):
	ambient_text = torch.cat([openai_clip.tokenize(
		f"{ambient}") for ambient in ambient_predictions]).to(device)
	salient_text = torch.cat([openai_clip.tokenize(
		f"{salient}") for salient in salient_predictions]).to(device)
	with torch.no_grad():
		ambient_text_features = ss.model.encode_text(ambient_text)
		ambient_text_features /= ambient_text_features.norm(dim=-1, keepdim=True)
		salient_text_features = ss.model.encode_text(salient_text)
		salient_text_features /= salient_text_features.norm(dim=-1, keepdim=True)
	logit_scale = ss.model.logit_scale.exp()
	similarities = (logit_scale * salient_text_features @
					ambient_text_features.t()).softmax(dim=-1)
	probs, word_idxs = similarities[0].topk(5)
	sorted_ambient_audio = []
	for prob, word_idx in zip(probs, word_idxs):
		# print(word_list[index] + ": " + str(100 * value.item()))
		sorted_ambient_audio.append(ambient_predictions[word_idx])
	# print("==========")
	return sorted_ambient_audio


def min_max_norm(array):
	lim = [array.min(), array.max()]
	array = array - lim[0]
	array.mul_(1 / (1.e-10 + (lim[1] - lim[0])))
	# array = torch.clamp(array, min=0, max=1)
	return array


def torch_to_rgba(img):
	img = min_max_norm(img)
	rgba_im = img.permute(1, 2, 0).cpu()
	if rgba_im.shape[2] == 3:
		rgba_im = torch.cat(
			(rgba_im, torch.ones(*rgba_im.shape[:2], 1)), dim=2)
	assert rgba_im.shape[2] == 4
	return rgba_im


def numpy_to_image(img, size):
	"""
	takes a [0..1] normalized rgba input and returns resized image as [0...255] rgba image
	"""
	resized = Image.fromarray((img*255.).astype(np.uint8)).resize((size, size))
	return resized


def heatmap(image: torch.Tensor, heatmap: torch.Tensor, size=None, alpha=0.6):
	if not size:
		size = image.shape[1]
	img = torch_to_rgba(image).numpy()
	hm = cm.hot(min_max_norm(heatmap).numpy())
	img = np.array(numpy_to_image(img, size))
	hm = np.array(numpy_to_image(hm, size))
	rgb = np.zeros((224, 224, 3))
	rgb[:, :, 0] = hm[:, :, 0]
	rgb[:, :, 1] = hm[:, :, 1]
	rgb[:, :, 2] = hm[:, :, 2]
	hm_rgb = np.asarray(rgb)
	hm_r = np.dot(hm_rgb[..., :3], [1, 0, 0])
	# st.write(hm_r)
	pan = (ndimage.measurements.center_of_mass(hm_r)[1] / 224 - 0.5) * 2
	volume = min((hm_r > 30).sum() / (224 * 224) * 40 - 10, 10)
	# return pan, volume
	# st.image(Image.fromarray((alpha * hm + (1 - alpha) *img).astype(np.uint8)))
	return Image.fromarray((alpha * hm + (1 - alpha) * img).astype(np.uint8)), pan, volume


def align_scene_and_audio(scene, audio_list):
	# print(len(scene))
	text = openai_clip.tokenize(audio_list).to(device)
	scene_meta = []
	for frame in scene:
		image = preprocess(frame).unsqueeze(0).to(device)
		with torch.no_grad():
			text_features = ss.model.encode_text(text)
			text_features_norm = text_features.norm(dim=-1, keepdim=True)
			text_features_new = text_features / text_features_norm
			image_features = ss.model.encode_image(image)
			image_features_norm = image_features.norm(dim=-1, keepdim=True)
			image_features_new = image_features / image_features_norm
			logit_scale = ss.model.logit_scale.exp()
			similarities = (logit_scale * image_features_new @
							text_features_new.t()).softmax(dim=-1)
			probs = similarities[0].cpu().numpy().tolist()
		# hms = []
		pans = []
		volumes = []
		for i in range(len(audio_list)):
			# multiply the normalized text embedding with image norm to get approx image embedding
			text_prediction = (text_features_new[[i]] * image_features_norm)
			saliency = grad_cam(ss.model.visual, image.type(
				ss.model.dtype), text_prediction, saliency_layer="layer4.2.relu")
			# hm = heatmap(image[0], saliency[0][0,].detach().type(torch.float32).cpu(), alpha=0.7)
			hm, pan, volume = heatmap(
				image[0], saliency[0][0, ].detach().type(torch.float32).cpu(), alpha=0.7)
			pans.append(pan)
			volumes.append(volume)
			# st.image(hm)
			# st.write(prob)
			# st.write("Pan:" + str(pan))
			# st.write("Volume:" + str(volume))
			# hms.append(hm)
		frame_meta = []
		for audio, prob, pan, volume in zip(audio_list, probs, pans, volumes):
			if prob > 0.1:
				frame_meta.append([audio, pan, volume])
		scene_meta.append(frame_meta)
	# st.write(scene_meta)
	# st.write(scene_meta[0][0])
	return scene_meta


def generate_audio(audio_label, scene_meta, present_ids, start, scene_length, num_frames):
	start += 0.05
	crossfade_duration = 500
	# st.write(scene_length)
	# st.write(num_frames)
	# find nearest .wav by length
	sound = AudioSegment.from_file(
		"sounds/" + audio_label + ".wav", format="wav")
	first_frame = True
	audio_track = AudioSegment.empty()
	time = 0
	# print(len(present_ids))
	for id in present_ids:
		# st.write(scene_meta)
		pan = scene_meta[id[0]][id[1]][1]
		volume = scene_meta[id[0]][id[1]][2]
		adjusted_sound = sound.pan(float(pan))
		adjusted_sound = adjusted_sound + volume
		if first_frame:  # start
			audio_track = adjusted_sound[(
				time + start) * 1000:((time + start) + 1) * 1000]
			first_frame = False
		elif time == num_frames - 1:  # end
			# st.write("last one " + str(time))
			this_sound = adjusted_sound[(
				time) * crossfade_duration - 500:scene_length * 1000]
			# this_sound = adjusted_sound[(time) * 1000:scene_length * 1000]
			audio_track = audio_track.append(
				this_sound, crossfade=crossfade_duration)
			# crossfade_test = panned_left[0:10000].append(panned_right[00000:20000], crossfade=10000)
		else:  # in between
			# this_sound = adjusted_sound[(time - 1) * 1000:(time + 1) * 1000]
			this_sound = adjusted_sound[(
				time) * 1000 - crossfade_duration:(time + 1) * 1000]
			# this_sound = adjusted_sound[(time) * 1000:(time + 1) * 1000]
			audio_track = audio_track.append(
				this_sound, crossfade=crossfade_duration)
		time += 1
	silent_track = AudioSegment.silent(duration=scene_length * 1000)
	full_track = silent_track.overlay(audio_track, position=(start) * 1000)
	# st.write(len(silent_track))
	# st.write(len(audio_track))
	# st.write(len(full_track))
	# play(full_track)
	# st.write(full_track.duration_seconds)
	# play(full_track)
	# TODO: FADE IN FADE OUT
	return full_track

def generate_audio_for_scene(scene, audio_list, scene_length):
	scene_meta = align_scene_and_audio(scene, audio_list)
	# st.write(scene_meta)
	audio_tracks = []
	for audio_label in audio_list:  # iterate through each audio
		present_ids = []
		for i in range(len(scene)):  # iterate through frames
			# iterate through present sounds (per frame)
			for j in range(len(scene_meta[i])):
				if audio_label in scene_meta[i][j]:
					present_ids.append([i, j])
		# st.write(audio_label)
		# st.write(present_ids)
		if present_ids:  # audio exists for scene
			# st.write(audio_label + " " + str(present_ids))
			frame_ids = [pair[0] for pair in present_ids]
			for group in more_itertools.consecutive_groups(frame_ids):
				group_list = list(group)
				start = group_list[0]
				end = group_list[-1]
				# st.write(audio_label)
				audio_tracks.append(generate_audio(
					audio_label, scene_meta, present_ids, start, scene_length, len(scene)))
	scene_track = AudioSegment.silent(duration=scene_length * 1000)
	# st.write(len(audio_tracks))
	for audio_track in audio_tracks:
		scene_track = scene_track.overlay(audio_track)
	return scene_track

def generate_ambient_for_scene(audio, scene_length):
	scene_track = AudioSegment.silent(duration=scene_length * 1000)
	if audio != "none":
		audio_track = AudioSegment.from_file("sounds/" + audio + ".wav", format="wav")[:scene_length * 1000] - 5
		scene_track = scene_track.overlay(audio_track, position=0.05 * 1000)
	return scene_track

def merge_scenes():
	output_name = "soundify_output.mp4"
	merge = []
	for root, dirs, files in os.walk(os.getcwd() + "/processed"):
		files = natsorted(files)
		for file in files:
			if os.path.splitext(file)[1] == ".mp4":
				file_path = os.path.join(root, file)
				video_file = mpe.VideoFileClip(file_path)
				merge.append(video_file)
	merge_clip = mpe.concatenate_videoclips(merge)

	# # add backround music
	# background_music = mpe.AudioFileClip("sounds/supersonic.mp3")
	# background_music = afx.volumex(background_music, 0.5)
	# combined_audio = mpe.CompositeAudioClip([merge_clip.audio, background_music])
	# merge_clip = merge_clip.set_audio(combined_audio)

	merge_clip.write_videofile(output_name, temp_audiofile='temp-audio.m4a',
							   remove_temp=True, codec="libx264", audio_codec="aac", fps=ss.fps, logger=None)
	st.success("Soundification complete. See " + output_name + ".")
	# st.balloons()


def user_action(message):
	url = "https://hooks.slack.com/services/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"
	message = (message)
	slack_data = {"text": message}
	byte_length = str(sys.getsizeof(slack_data))
	headers = {'Content-Type': "application/json",
			   'Content-Length': byte_length}
	response = requests.post(url, data=json.dumps(slack_data), headers=headers)
	if response.status_code != 200:
		raise Exception(response.status_code, response.text)


st.set_page_config(page_title="Soundify", page_icon="🔊",
				   layout="centered", initial_sidebar_state="collapsed")
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
st.title("Soundify")
ss = SessionState.get(video=None, video_name=None, scenes=None, scene_lengths=None, fps=None, num_scenes=None,
					  scene_features=None, predicted_salient_audios=None, predicted_ambient_audios=None, salient_audios=None, ambient_audios=None, scenes_with_ambient=None, curr_scene=None, progress=1)

ss.video = st.file_uploader("Upload a video", type=["mp4"])
device = "cuda" if torch.cuda.is_available() else "cpu"
# ['RN50', 'RN101', 'RN50x4', 'ViT-B/32']
with st.spinner("Loading model..."):
	ss.model, preprocess = load_model("RN50")
# print(openai_clip.available_models())

if ss.video and ss.progress == 1:
	ss.video_name = ss.video.name
	with st.spinner("Splitting video to scenes..."):
		ss.scenes, ss.scene_lengths, ss.fps = video_to_scenes(ss.video_name)
	# st.write(ss.scene_lengths)
	ss.num_scenes = len(ss.scenes)
	ss.scene_features = []
	with st.spinner("Encoding scenes..."):
		for scene in ss.scenes:
			scene_feature = encode_scene(scene)
			ss.scene_features.append(scene_feature)
	salient_list = ["car", "lion", "helicopter", "telephone", "cooking", "fire", "bike",
				 "waterfall", "camera", "keyboard", "subway", "go kart", "people", "snow", "forest"]
	ambient_list = ["traffic", "africa", "blizzard", "room", "street"]
	# word_list = ["dog", "beach", "people", "road", "apple", "airplane", "rain", "cars"]
	# word_list = ["people", "taxi", "car", "building", "chess", "night", "camera", "street", "nyc", "bicycle"]
	ss.predicted_salient_audios, ss.salient_audios, ss.predicted_ambient_audios, ss.ambient_audios, ss.scenes_with_ambient = [], [], [], [], []
	with st.spinner("Classifying audio for scenes..."):
		for i in range(ss.num_scenes):
			salient_predictions = classify_audio_for_scene(salient_list, ss.scene_features[i])
			ss.predicted_salient_audios.append(salient_predictions)
			ss.salient_audios.append(salient_predictions[0:1])
			ambient_predictions = classify_audio_for_scene(ambient_list, ss.scene_features[i])
			ss.predicted_ambient_audios.append(ambient_predictions)
			ss.ambient_audios.append("none")
			ss.scenes_with_ambient.append(False)
	ss.progress = 2

if ss.progress == 2:
	ss.curr_scene = st.select_slider(
		"Select scene", options=list(range(1, ss.num_scenes + 1)))
	preview = st.empty()
	ss.salient_audios[ss.curr_scene - 1] = st.multiselect("Select sound(s) for scene", ss.predicted_salient_audios[ss.curr_scene - 1], ss.salient_audios[ss.curr_scene - 1])
	add_ambient = st.checkbox("Add ambient sound", ss.scenes_with_ambient[ss.curr_scene - 1])
	# st.write(ss.ambient_audios[ss.curr_scene - 1])
	if add_ambient:
		ss.predicted_ambient_audios[ss.curr_scene - 1] = sort_ambient_predictions(ss.predicted_ambient_audios[ss.curr_scene - 1], ss.salient_audios[ss.curr_scene - 1])
		ss.ambient_audios[ss.curr_scene - 1] = st.selectbox("Select ambient sound for scene", ss.predicted_ambient_audios[ss.curr_scene - 1])
		ss.scenes_with_ambient[ss.curr_scene - 1] = True
		# st.write(ss.ambient_audios[ss.curr_scene - 1])
	else:
		ss.ambient_audios[ss.curr_scene - 1] = "none"
		ss.scenes_with_ambient[ss.curr_scene - 1] = False
	video_file = open("temp/" + str(ss.curr_scene - 1) + ".mp4", 'rb')
	video_bytes = video_file.read()
	preview.video(video_bytes)

if st.button("Soundify!"):
	# generate_audio_for_scene(ss.scenes[5], ss.salient_audios[5], ss.scene_lengths[5]) # single scene testing
	scene_counter = 0
	with st.spinner("Generating audio for scenes..."):
		for scene, salient_audio, ambient_audio, scene_length in zip(ss.scenes, ss.salient_audios, ss.ambient_audios, ss.scene_lengths):
			salient_track = generate_audio_for_scene(scene, salient_audio, scene_length)
			ambient_track = generate_ambient_for_scene(ambient_audio, scene_length)
			scene_track = salient_track.overlay(ambient_track)
			scene_track.export("temp/" + str(scene_counter) + ".wav", format="wav")
			audio_clip = mpe.AudioFileClip("temp/" + str(scene_counter) + ".wav")
			video_clip = mpe.VideoFileClip("temp/" + str(scene_counter) + ".mp4")
			combined_clip = video_clip.set_audio(audio_clip)
			combined_clip.write_videofile("processed/" + str(scene_counter) + ".mp4", temp_audiofile='temp-audio.m4a',
										remove_temp=True, codec="libx264", audio_codec="aac", fps=ss.fps, logger=None)
			scene_counter += 1
	with st.spinner("Merging scenes..."):
		merge_scenes()
