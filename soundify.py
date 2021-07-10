# 1. Install dependencies via pip install -r requirements.txt
# 2. Download and unzip sound files under "sound" directory from https://drive.google.com/file/d/1Ag1bcTJgJIDn92afHja86zxGt_YDgUta/view?usp=sharing
# 3. Run app via streamlit run soundify.py
# 4. Try out a demo video from https://drive.google.com/file/d/1zaqumFFkAavdAwO-pkgn_xUPiRgpz5iA/view?usp=sharing
import math
import os
import clip as openai_clip
import cv2
import more_itertools
import moviepy.editor as mpe
import numpy as np
import scipy.signal as si
import streamlit as st
import torch
from decord import VideoReader, cpu, gpu
from matplotlib import cm
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from natsort import natsorted
from PIL import Image
from pydub import AudioSegment
from scipy import ndimage
from torchray.attribution.grad_cam import grad_cam
import SessionState

# Loads CLIP
@st.cache(show_spinner=False)
def load_model(model):
	return openai_clip.load(model, device=device, jit=False)

# Processes video stream into frames, also splits video into "scenes" based on color histogram distance between neighboring frames
@st.cache(show_spinner=False)
def video_to_scenes(video):
	diff_threshold = 0.24 # Color histogram distance threshold for determining change of scene
	window_size = 1 # Keeping this small to tolerate smaller peaks, see "distance" parameter in https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
	# Read video with decord
	vr = VideoReader(video, ctx=cpu(0))
	# Get frames per second info
	fps = vr.get_avg_fps()
	frames = []
	frame_diff = []
	frame_count = len(vr)
	# For each frame, compute grayscale color histogram
	for i in range(frame_count):
		frame = vr[i].asnumpy()
		gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		num_pixel = frame.shape[0] * frame.shape[1]
		hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
		hist = hist * (1.0 / num_pixel)
		frame_diff.append(hist)
		frames.append(Image.fromarray(frame[:, :, ::-1]))
	# For each frame, compute color histogram distance between this frame and the previous frame
	frame_count = len(frame_diff) - 1
	while frame_count >= 1:
		# Distance = Sum(|histogram of i - histogram of (i - 1)|)
		frame_diff[frame_count] = np.sum(
			np.abs(np.subtract(frame_diff[frame_count], frame_diff[frame_count - 1])))
		frame_count -= 1
	frame_diff[0] = 0
	vector = np.array(frame_diff)
	# Finds "peaks" above the threshold in color histogram distances, which are the indexes of the first frame of each "scene"
	indexes, _ = si.find_peaks(
		vector, height=diff_threshold, distance=window_size)
	prev_index = 0
	scenes = []
	scene_lengths = []
	counter = 0
	# Collect scenes
	for index in indexes:
		scenes.append(frames[prev_index:index][0::int(fps)])
		scene_lengths.append(index / fps - prev_index / fps)
		# Save scene as temporary .mp4 file for UI purposes
		ffmpeg_extract_subclip(
			video, prev_index / fps, index / fps, targetname="temp/" + str(counter) + ".mp4")
		prev_index = index
		counter += 1
	# Handle off-by-one (last scene)
	scenes.append(frames[prev_index:len(frames)][0::int(fps)])
	scene_lengths.append(len(frames) / fps - prev_index / fps)
	ffmpeg_extract_subclip(video, prev_index / fps, len(frames) /
                        fps, targetname="temp/" + str(counter) + ".mp4")
	return scenes, scene_lengths, fps

# For a scene, converts video frames into video embedding (stacked image embeddings)
@st.cache(show_spinner=False)
def encode_scene(video_frames):
	batch_size = 256 # Process in batches for speed
	batches = math.ceil(len(video_frames) / batch_size)
	video_features = torch.empty([0, 1024], dtype=torch.float16).to(device)
	# In batches, convert scene into stacked image embeddings
	for i in range(batches):
		batch_frames = video_frames[i * batch_size:(i + 1) * batch_size]
		batch_preprocessed = torch.stack(
			[preprocess(frame) for frame in batch_frames]).to(device)
		with torch.no_grad():
			batch_features = ss.model.encode_image(batch_preprocessed)
			batch_features /= batch_features.norm(dim=-1, keepdim=True)
		video_features = torch.cat((video_features, batch_features))
	return video_features

# For a scene, classify its main sound labels
@st.cache(show_spinner=False)
def classify_audio_for_scene(word_list, scene_features):
	# Tokenize sound label texts
	text = torch.cat([openai_clip.tokenize(
		f"{word}") for word in word_list]).to(device)
	# Convert sound label texts into text embeddings
	with torch.no_grad():
		text_features = ss.model.encode_text(text)
		text_features /= text_features.norm(dim=-1, keepdim=True)
	# Compare video embedding with text embeddings (cosine similarities)
	logit_scale = ss.model.logit_scale.exp()
	similarities = (logit_scale * scene_features @
                 text_features.t()).softmax(dim=-1)
	# Select the top-5 sound label texts with the highest similarities
	probs, word_idxs = similarities[0].topk(5)
	predicted_audio = []
	for prob, word_idx in zip(probs, word_idxs):
		predicted_audio.append(word_list[word_idx])
	return predicted_audio

# Adaptively sort the order of predicted ambient sounds based on what the user selects for the main sounds
def sort_ambient_predictions(ambient_predictions, main_predictions):
	# Tokenize sound label texts
	ambient_text = torch.cat([openai_clip.tokenize(
		f"{ambient}") for ambient in ambient_predictions]).to(device)
	main_text = torch.cat([openai_clip.tokenize(
		f"{main}") for main in main_predictions]).to(device)
	# Convert texts into text embeddings
	with torch.no_grad():
		ambient_text_features = ss.model.encode_text(ambient_text)
		ambient_text_features /= ambient_text_features.norm(dim=-1, keepdim=True)
		main_text_features = ss.model.encode_text(main_text)
		main_text_features /= main_text_features.norm(dim=-1, keepdim=True)
	# Compare main sounds (user-selected) with ambient sounds (cosine similarities)
	logit_scale = ss.model.logit_scale.exp()
	similarities = (logit_scale * main_text_features @
                 ambient_text_features.t()).softmax(dim=-1)
	# Select the top-5 ambient sound label texts with the highest similarities
	probs, word_idxs = similarities[0].topk(5)
	sorted_ambient_audio = []
	for prob, word_idx in zip(probs, word_idxs):
		sorted_ambient_audio.append(ambient_predictions[word_idx])
	return sorted_ambient_audio

# Helper function: Normalizing for heatmap function
def min_max_norm(array):
	lim = [array.min(), array.max()]
	array = array - lim[0]
	array.mul_(1 / (1.e-10 + (lim[1] - lim[0])))
	return array

# Helper function: Converts PyTorch to RGBA for heatmap function
def torch_to_rgba(img):
	img = min_max_norm(img)
	rgba_im = img.permute(1, 2, 0).cpu()
	if rgba_im.shape[2] == 3:
		rgba_im = torch.cat(
			(rgba_im, torch.ones(*rgba_im.shape[:2], 1)), dim=2)
	assert rgba_im.shape[2] == 4
	return rgba_im

# Helper function: Converts RGBA numpy [0...1] to RGBA image [0...255] for heatmap function
def numpy_to_image(img, size):
	resized = Image.fromarray((img*255.).astype(np.uint8)).resize((size, size))
	return resized

# Creates heatmap visualization of activation map and computes pan and volume parameters
def heatmap(image: torch.Tensor, heatmap: torch.Tensor, size=None, alpha=0.6):
	if not size:
		size = image.shape[1]
	img = torch_to_rgba(image).numpy()
	hm = cm.hot(min_max_norm(heatmap).numpy())
	img = np.array(numpy_to_image(img, size))
	hm = np.array(numpy_to_image(hm, size))
	# Some messy operations to remove the Alpha channel of RGBA (can be improved)
	rgb = np.zeros((224, 224, 3))
	rgb[:, :, 0] = hm[:, :, 0]
	rgb[:, :, 1] = hm[:, :, 1]
	rgb[:, :, 2] = hm[:, :, 2]
	hm_rgb = np.asarray(rgb)
	hm_r = np.dot(hm_rgb[..., :3], [1, 0, 0])
	# Left-right panning is computed by the x-axis of the center of mass of the 224x224 activation map. Range [-0.5, 0.5].
	pan = (ndimage.measurements.center_of_mass(hm_r)[1] / 224 - 0.5) * 2
	# Volume (dB) is computed by the area of the 224x224 activation map, only counting pixels with red value > 30.
	# Some scaling determined empirically. Capped +10 to prevent R.I.P. headphone users.
	volume = min((hm_r > 30).sum() / (224 * 224) * 40 - 10, 10)
	return Image.fromarray((alpha * hm + (1 - alpha) * img).astype(np.uint8)), pan, volume

# Identifies where sounds need to be generated in the scene, i.e., whether a frame contains the sound emitter, e.g., whether there is a bike in the particular frame
def align_scene_and_audio(scene, audio_list):
	# Tokenize sound label texts
	text = openai_clip.tokenize(audio_list).to(device)
	scene_meta = []
	for frame in scene:
		image = preprocess(frame).unsqueeze(0).to(device)
		with torch.no_grad():
			# Convert user-selected sound label texts into text embeddings
			text_features = ss.model.encode_text(text)
			text_features_norm = text_features.norm(dim=-1, keepdim=True)
			text_features_new = text_features / text_features_norm
			# Convert image frame into image embedding
			image_features = ss.model.encode_image(image)
			image_features_norm = image_features.norm(dim=-1, keepdim=True)
			image_features_new = image_features / image_features_norm
			# Compare image embedding with text embeddings of selected sound labels (cosine similarities)
			logit_scale = ss.model.logit_scale.exp()
			similarities = (logit_scale * image_features_new @
                            text_features_new.t()).softmax(dim=-1)
			# The probability score will be used to determine whether a given sound should appear for this frame
			probs = similarities[0].cpu().numpy().tolist()
		pans = []
		volumes = []
		# Loop through each main sound
		for i in range(len(audio_list)):
			# Get approximated image embedding with normalized text embedding x image norm
			text_prediction = (text_features_new[[i]] * image_features_norm)
			# Get activation map via Grad-CAM on the ReLU activation of the last visual layer (for ResNet-50 architecture)
			activation = grad_cam(ss.model.visual, image.type(
				ss.model.dtype), text_prediction, saliency_layer="layer4.2.relu")
			# Call heatmap function to get heatmap, pan, and volume
			hm, pan, volume = heatmap(
				image[0], activation[0][0, ].detach().type(torch.float32).cpu(), alpha=0.7)
			pans.append(pan)
			volumes.append(volume)
			st.image(hm) # Print heatmap image on UI
			st.write("Pan: " + str(pan) + "; Volume: " + str(volume)) # Print pan and volume values on UI
		# If probability score > 0.1, then take note that the main sound needs to be generated for this frame (along with pan and volume parameters) in the frame's meta data
		frame_meta = []
		for audio, prob, pan, volume in zip(audio_list, probs, pans, volumes):
			if prob > 0.1:
				frame_meta.append([audio, pan, volume])
		scene_meta.append(frame_meta)
	return scene_meta

# Progressively generates/mixes 1-second length sound bits for a scene (actual sound files), most audio operations done with pydub https://github.com/jiaaro/pydub/blob/master/API.markdown
def generate_audio(audio_label, scene_meta, present_ids, start, scene_length, num_frames):
	# Half second crossfades for smoothing between sound bits
	crossfade_duration = 500
	# Retrieve sound file from database
	sound = AudioSegment.from_file(
		"sounds/" + audio_label + ".wav", format="wav")
	first_frame = True
	audio_track = AudioSegment.empty()
	# Keep track of the current time
	time = 0
	for id in present_ids:
		# Adjust panning and volume
		pan = scene_meta[id[0]][id[1]][1]
		volume = scene_meta[id[0]][id[1]][2]
		adjusted_sound = sound.pan(float(pan))
		adjusted_sound = adjusted_sound + volume
		# If first frame of the scene, generate a 1 second length sound bit
		if first_frame:
			audio_track = adjusted_sound[(
				time + start) * 1000:((time + start) + 1) * 1000]
			first_frame = False
		# If last frame of the scene, generate a [crossfade duration + whatever length is left of the scene] length sound bit
		elif time == num_frames - 1:  # end
			this_sound = adjusted_sound[(
				time) * 1000 - crossfade_duration:scene_length * 1000]
			audio_track = audio_track.append(
				this_sound, crossfade=crossfade_duration)
		# If in-between frame of the scene, generate a [crossfade duration + 1 second] length sound bit
		else:
			this_sound = adjusted_sound[(
				time) * 1000 - crossfade_duration:(time + 1) * 1000]
			audio_track = audio_track.append(
				this_sound, crossfade=crossfade_duration)
		time += 1
	# Merge the sound bits together into a full audio track for the scene
	silent_track = AudioSegment.silent(duration=scene_length * 1000)
	full_track = silent_track.overlay(audio_track, position=(start) * 1000)
	return full_track

# For each user-selected main audio of a scene, generate an audio track for it by calling the generate_audio function
def generate_audio_for_scene(scene, audio_list, scene_length):
	# First, identify which sounds need to be generated in which parts of the scene by calling the align_scene_and_audio function, and store this information in scene_meta
	scene_meta = align_scene_and_audio(scene, audio_list) # scene_meta contains the names of the sounds present (+ pan and volume parameters) for each frame of the scene
	# st.write(scene_meta) # Can uncomment this line of code to interpret the values
	audio_tracks = []
	# For each user-selected main sound
	for audio_label in audio_list:
		present_ids = []
		# For each frame of the scene
		for i in range(len(scene)):
			for j in range(len(scene_meta[i])):
				# If this sound is present in this frame, e.g., a bike is detected in this frame, mark this frame's index
				if audio_label in scene_meta[i][j]:
					present_ids.append([i, j])
					# st.write("present_ids: " + str(audio_label) + " " + str(present_ids)) # Can uncomment this line of code to interpret the values
		# If this sound is present in this scene at all
		if present_ids:
			# Get the indices of the frames
			frame_ids = [pair[0] for pair in present_ids]
			# st.write("frame_ids: " + str(frame_ids)) # Can uncomment this line of code to interpret the values
			# Identify intervals (consecutive frames where this sound is present), and generate the audio track for each interval
			for group in more_itertools.consecutive_groups(frame_ids):
				group_list = list(group)
				start = group_list[0]
				audio_tracks.append(generate_audio(
					audio_label, scene_meta, present_ids, start, scene_length, len(scene)))
	scene_track = AudioSegment.silent(duration=scene_length * 1000)
	# Combine all audio tracks (e.g., of different sound categories) into the combined audio track for the scene
	for audio_track in audio_tracks:
		scene_track = scene_track.overlay(audio_track)
	return scene_track

# Generates/mixes the ambient sound track for a scene
def generate_ambient_for_scene(audio, scene_length):
	scene_track = AudioSegment.silent(duration=scene_length * 1000)
	# If the user selects an ambient sound (ticks ambient sound checkbox)
	if audio != "none":
		# The ambient sound track is the same length as the scene, with a -5 dB volume adjustment to reduce clashing with the main sounds
		audio_track = AudioSegment.from_file(
			"sounds/" + audio + ".wav", format="wav")[:scene_length * 1000] - 5
		scene_track = scene_track.overlay(audio_track, position=0.05 * 1000)
	return scene_track

# Merges all the scenes together into an output video
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

	# # Optionally, add a backround music track
	# background_music = mpe.AudioFileClip("sounds/supersonic.mp3")
	# background_music = afx.volumex(background_music, 0.5)
	# combined_audio = mpe.CompositeAudioClip([merge_clip.audio, background_music])
	# merge_clip = merge_clip.set_audio(combined_audio)

	merge_clip.write_videofile(output_name, temp_audiofile="temp-audio.m4a",
                            remove_temp=True, codec="libx264", audio_codec="aac", fps=ss.fps, logger=None)
	st.success("Soundification complete. See " + output_name + ".")
	# st.balloons() # 🎈
	video_file = open(output_name, "rb")
	video_bytes = video_file.read()
	st.video(video_bytes)

# Page and styling configurations
st.set_page_config(page_title="Soundify", page_icon="🔊",
                   layout="centered", initial_sidebar_state="collapsed")
hide_streamlit_style = """
	<style>
	MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
	* {font-family: Avenir;}
	.css-gma2qf {display: flex; justify-content: center; font-size: 36px; font-weight: bold;}
	a:link {text-decoration: none;}
	a:hover {text-decoration: none;}
	.st-ba {font-family: Avenir;}
	</style>
	"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.title("Soundify")

# Streamlit "global variables"
ss = SessionState.get(video=None, video_name=None, scenes=None, scene_lengths=None, fps=None, num_scenes=None,
                      scene_features=None, predicted_main_audios=None, predicted_ambient_audios=None, main_audios=None, ambient_audios=None, scenes_with_ambient=None, curr_scene=None, progress=1)

# Video file uploader
ss.video = st.file_uploader("Upload a video", type=["mp4"])

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP
with st.spinner("Loading model..."):
	ss.model, preprocess = load_model("RN50")

# If user uploads a video
if ss.video and ss.progress == 1:
	ss.video_name = ss.video.name
	# Split uploaded video into scenes
	with st.spinner("Splitting video to scenes..."):
		ss.scenes, ss.scene_lengths, ss.fps = video_to_scenes(ss.video_name)
	ss.num_scenes = len(ss.scenes)
	ss.scene_features = []
	# Encode each scene
	with st.spinner("Encoding scenes..."):
		for scene in ss.scenes:
			scene_feature = encode_scene(scene)
			ss.scene_features.append(scene_feature)
	# List of "main" sounds in the database
	with open("main-sounds.txt") as f:
		main_list = [line.rstrip('\n') for line in f]
	print(main_list)
	# List of "ambient" sounds in the database
	with open("ambient-sounds.txt") as f:
		ambient_list = [line.rstrip('\n') for line in f]
	print(ambient_list)
	ss.predicted_main_audios, ss.main_audios, ss.predicted_ambient_audios, ss.ambient_audios, ss.scenes_with_ambient = [], [], [], [], []
	# For each scene, classify matching sounds
	with st.spinner("Classifying audio for scenes..."):
		for i in range(ss.num_scenes):
			# Classify "main" sounds
			main_predictions = classify_audio_for_scene(
				main_list, ss.scene_features[i])
			ss.predicted_main_audios.append(main_predictions)
			# By default, assign the top matching main sound to be generated
			ss.main_audios.append(main_predictions[0:1])
			# Classify "ambient" sounds
			ambient_predictions = classify_audio_for_scene(
				ambient_list, ss.scene_features[i])
			ss.predicted_ambient_audios.append(ambient_predictions)
			# By default, assign no ambient sound to be generated
			ss.ambient_audios.append("none")
			ss.scenes_with_ambient.append(False)
	ss.progress = 2

if ss.progress == 2:
	# Slider for user to navigate through scenes
	ss.curr_scene = st.select_slider(
		"Select scene", options=list(range(1, ss.num_scenes + 1)))
	preview = st.empty()
	# Multi-selection box for user to select main sound(s) to generate for a scene
	ss.main_audios[ss.curr_scene - 1] = st.multiselect(
		"Select sound(s) for scene", ss.predicted_main_audios[ss.curr_scene - 1], ss.main_audios[ss.curr_scene - 1])
	# Checkbox for user to add an ambient sound or not
	add_ambient = st.checkbox(
		"Add ambient sound", ss.scenes_with_ambient[ss.curr_scene - 1])
	# If user wishes to add an ambient sound
	if add_ambient:
		# Sort predicted ambient sounds based on what the user-selected for the main sound(s)
		ss.predicted_ambient_audios[ss.curr_scene - 1] = sort_ambient_predictions(
			ss.predicted_ambient_audios[ss.curr_scene - 1], ss.main_audios[ss.curr_scene - 1])
		# Single-selection box for user to select ambient sound to generate for a scene
		ss.ambient_audios[ss.curr_scene - 1] = st.selectbox(
			"Select ambient sound for scene", ss.predicted_ambient_audios[ss.curr_scene - 1])
		ss.scenes_with_ambient[ss.curr_scene - 1] = True
	# Otherwise, don't add ambient sound
	else:
		ss.ambient_audios[ss.curr_scene - 1] = "none"
		ss.scenes_with_ambient[ss.curr_scene - 1] = False
	# Let user preview the scene on Streamlit UI
	video_file = open("temp/" + str(ss.curr_scene - 1) + ".mp4", "rb")
	video_bytes = video_file.read()
	preview.video(video_bytes)

# If user clicks "Soundify!" button
if st.button("Soundify!"):
	scene_counter = 0
	with st.spinner("Generating audio for scenes..."):
		# For each scene, generate main audio track(s), ambient audio track, and combine them with original video
		for scene, main_audio, ambient_audio, scene_length in zip(ss.scenes, ss.main_audios, ss.ambient_audios, ss.scene_lengths):
			# Generate main audio track
			main_track = generate_audio_for_scene(scene, main_audio, scene_length)
			# Generate ambient audio track
			ambient_track = generate_ambient_for_scene(ambient_audio, scene_length)
			# Merge main and ambient audio tracks
			scene_track = main_track.overlay(ambient_track)
			# Save temporary merged sound file
			scene_track.export("temp/" + str(scene_counter) + ".wav", format="wav")
			audio_clip = mpe.AudioFileClip("temp/" + str(scene_counter) + ".wav")
			video_clip = mpe.VideoFileClip("temp/" + str(scene_counter) + ".mp4")
			# Combine merged sound file and original video
			combined_clip = video_clip.set_audio(audio_clip)
			# Save temporary .mp4 file for scene
			combined_clip.write_videofile("processed/" + str(scene_counter) + ".mp4", temp_audiofile="temp-audio.m4a",
                                 remove_temp=True, codec="libx264", audio_codec="aac", fps=ss.fps, logger=None)
			scene_counter += 1
	with st.spinner("Merging scenes..."):
		# Merge all temporary .mp4 files for each scene into final output video
		merge_scenes()