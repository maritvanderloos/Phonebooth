import pathlib
import wave
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras.src.saving.saving_lib import load_model

import librosa as l
from scipy.io import wavfile

from tensorflow.keras import layers
from tensorflow.keras import models
from pydub import AudioSegment
from scipy.io import wavfile
import itertools

label_names = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']


# load the model and put its weights
MODEL_PATH = "saved_model.keras"
print("Loading model " + MODEL_PATH)
model = load_model(MODEL_PATH)
WEIGHTS_PATH = "saved.weights.h5"
print("Loading weights from " + WEIGHTS_PATH)
model.load_weights(WEIGHTS_PATH)

def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

# make a spectrogram of the waveforms (specific for longer audios)
def get_long_spectrogram(waveform, number_of_frames):
    # Calculate frame_steps the same way as in keywords
    frame_steps = number_of_frames//125
    print(f"Frame steps: {frame_steps}")
    # # Squeeze the waveform to remove the channel dimension: [samples, 1] -> [samples]
    # waveform = tf.squeeze(waveform, axis=-1)


    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=frame_steps)
    # for 16000 and 16000 it is Spectrogram shape: (1, 8193, 1) so width 1 and height 8193
    # 16000/128 = 125, so to get the frame_step, do number of frames/125
    # Obtain the magnitude of the STFT (by dropping the phase).
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

audio_binary = tf.io.read_file('data/mini_speech_commands_extracted/mini_speech_commands/no/01bb6a2a_nohash_0.wav')

def plot_spectrogram(spectrogram, ax):
  print(spectrogram.shape)
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

# #audio_binary = tf.io.read_file("demo.wav")
# raw_audio, _ = tf.audio.decode_wav(audio_binary)
# audio = list(itertools.chain.from_iterable(raw_audio))
# samples = len(audio)
#
# # Calculate the spectrogram
# spectrogram_tf = get_long_spectrogram(audio, samples)
# # Convert the TensorFlow tensor to a NumPy array for plotting
# # spectrogram_np = spectrogram_tf.numpy()

wavefile = wave.open("voice_input/input/demo.wav", "r")
#wavefile = wave.open("voice_input/input/0f3f64d5_nohash_0.wav", "r")
total_frames = wavefile.getnframes()
data_set = tf.keras.utils.audio_dataset_from_directory(
    directory=pathlib.Path("voice_input"),
    output_sequence_length=total_frames,
    seed=0)

data_set = data_set.map(squeeze, tf.data.AUTOTUNE)
for audio, example_labels in data_set.take(1):
    for i in range(1):
        label = label_names[example_labels[i]]
        # these waveforms have len() 16000
        waveform = audio[i]
        # Use the length of the individual waveform, not the batch
        print(waveform.shape[0])
        waveform_length = waveform.shape[0]
        spectrogram = get_long_spectrogram(waveform, waveform_length)

        print('Label:', label)
        print('Waveform shape:', waveform.shape)
        print('Spectrogram shape:', spectrogram.shape)

# samples = len(audio)
# spectrogram_tf = get_long_spectrogram(audio, samples)


audio = l.load("voice_input/input/demo.wav")[0]
x = l.effects.trim(audio, top_db = 50)[0]

print(f"the x is {x}")

fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(waveform.shape[0])
print(f"waveform printed {waveform.numpy()}")
axes[0].plot(timescale, waveform.numpy())
axes[0].set_title('Waveform')
axes[0].set_xlim([0, 16000])

print(f"spectrogram printed {spectrogram.numpy()}")
plot_spectrogram(spectrogram.numpy(), axes[1])
axes[1].set_title('Spectrogram')
plt.suptitle(label.title())
plt.show()