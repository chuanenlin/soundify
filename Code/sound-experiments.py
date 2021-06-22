from pydub import AudioSegment, silence
from pydub.playback import play

sound1 = None
sound2 = None
sound3 = None
sound4 = None
sound1 = AudioSegment.from_file("sounds/camera.wav", format="wav")
sound2 = AudioSegment.from_file("sounds/dog-bark-5.wav", format="wav")
sound3 = AudioSegment.from_file("sounds/helicopter-10.wav", format="wav")
sound4 = AudioSegment.from_file("sounds/jungle-21.wav", format="wav")

#get raw data
raw_audio_data = sound1.raw_data # or .get_array_of_samples()

#get length
length_second = sound4.duration_seconds #length in seconds
length_millisecond = len(sound4) #length in ms

#get dBFS (db relative to the maximum possible loudness) or rms (not logarithmic)
loudness = sound1.dBFS
loudness = sound1.rms

#get highest amplitude
peak_amplitude = sound1.max
peak_amplitude = sound1.max_dBFS #in dBFS (relative to the highest possible amplitude value)

#get number of channels
channel_count = sound1.channels

#slice audio
first_10 = sound4[:10000] #first 10 seconds
last_5 = sound4[-5000:] #last 5 seconds

#boost/reduce volume (or use sound4.apply_gain(dB_value))
louder = sound3 + 5 #boost by 6dB
quieter = sound4 - 10 #reduce by 3.5dB
# play(louder)

#concatenate
sound3_and_4 = sound3 + sound4

#crossfade
crossfade = sound3.append(sound4, crossfade=5500) #1.5 seconds crossfade, default 100 ms crossfade

#overlay
played_togther = sound1.overlay(sound2)
sound2_starts_after_delay = sound1.overlay(sound2, position=5000)
volume_of_sound1_reduced_during_overlay = sound1.overlay(sound2, gain_during_overlay=-8)
sound2_repeats_until_sound1_ends = sound1.overlay(sound2, loop=True)
sound2_plays_twice = sound1.overlay(sound2, times=2) #sound2 loops will be truncated to length of sound1, e.g., sound2_plays_a_lot = sound1.overlay(sound2, times=10000)

#fades
fade_louder_for_3_seconds_in_middle = sound4.fade(to_gain=+6.0, start=7500, duration=3000)
fade_quieter_beteen_2_and_3_seconds = sound4.fade(to_gain=-3.5, start=2000, end=3000)
fade_in_the_hard_way = sound1.fade(from_gain=-120.0, start=0, duration=5000)
fade_out_the_hard_way = sound1.fade(to_gain=-120.0, end=0, duration=5000)
fade_in_and_out_the_easy_way = sound4.fade_in(2000).fade_out(3000) #2 seconds fade in, 3 seconds fade out, easy way is to use the .fade_in() convenience method, note: -120dB is basically silent

#repeat
repeat = sound1 * 2

#create empty
empty = AudioSegment.empty()

#create silence
silent_3 = AudioSegment.silent(duration=3000) #3 second silence

#detect silence, try removing silent sections
silent_moments = silence.detect_silence(sound3, min_silence_len=1000, silence_thresh=-16, seek_step=1) #returns a list of silent intervals (nested list), arguments are default values
# .detect_nonsilent(sound3) # is the inverse with same arguments
# .detect_leading_silence(sound3, silence_thresh=-50, chunk_size=10) # returns the millisecond index of when the leading silence ends
# .split_on_silence(sound3, min_silence_len=1000, silence_thresh=-16, seek_step=1, keep_silence=100) # returns a list of audio segments by splitting on silent sections
# silent_segments = silence.split_on_silence(sound3, silence_thresh=-56, keep_silence=100)[0]

#mono to stereo
left_channel = sound3.set_channels(1)
right_channel = sound3.set_channels(1)
stereo_sound = AudioSegment.from_mono_audiosegments(left_channel, right_channel)

#stereo to mono
mono = sound4.split_to_mono() #mono is a list [left, right]

#adjust stereo L-R volume
stereo_balance_adjusted = sound4.apply_gain_stereo(-10, +10)

#pan (takes in value between -1.0 (100% left) and 1.0 (100% right)), try interpolating
panned_right = sound4.pan(+1.00) # pan the sound 15% to the right
panned_left = sound4.pan(-1.00) # pan the sound 50% to the left

#export
sound4.export("sound4.mp3", format="mp3")

crossfade_test = panned_left[0:10000].append(panned_right[00000:20000], crossfade=10000)

play(crossfade_test)
