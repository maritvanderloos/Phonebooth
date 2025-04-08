import pathlib
import wave
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras.src.saving.saving_lib import load_model

from tensorflow.keras import layers
from tensorflow.keras import models
from pydub import AudioSegment
from scipy.io import wavfile

label_names = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']


# load the model and put its weights
MODEL_PATH = "saved_model.keras"
print("Loading model " + MODEL_PATH)
model = load_model(MODEL_PATH)
WEIGHTS_PATH = "saved.weights.h5"
print("Loading weights from " + WEIGHTS_PATH)
model.load_weights(WEIGHTS_PATH)


# make a spectrogram of the waveforms
def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT (by dropping the phase).
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

class ExportModel(tf.Module):
  def __init__(self, model):
    self.model = model

    # Accept either a string-filename or a batch of waveforms.
    # YOu could add additional signatures for a single wave, or a ragged-batch.
    #__call__ makes the class object callable
    self.__call__.get_concrete_function(
        x=tf.TensorSpec(shape=(), dtype=tf.string))
    self.__call__.get_concrete_function(
       x=tf.TensorSpec(shape=[None, 16000], dtype=tf.float32))


  @tf.function
  def __call__(self, x):
    # If they pass a string, load the file and decode it.
    if x.dtype == tf.string:
      x = tf.io.read_file(x)
      x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
      x = tf.squeeze(x, axis=-1)
      x = x[tf.newaxis, :]

    x = get_spectrogram(x)
    result = self.model(x, training=False)

    # Apply softmax to get probabilities
    probabilities = tf.nn.softmax(result, axis=-1)

    # Calculate entropy for each prediction
    # Higher entropy means more uncertainty/uniformity in distribution
    entropy = -tf.reduce_sum(
        probabilities * tf.math.log(probabilities + 1e-10),
        axis=-1
    )

    class_ids = tf.argmax(result, axis=-1)
    class_names = tf.gather(label_names, class_ids)

    return {'predictions':result,
            'entropy': entropy,
            'class_ids': class_ids,
            'class_names': class_names}

export = ExportModel(model)
# print("get:")
# print()
# print(export(tf.constant(str('data/get.wav'))))
def plot_waves(example_audio):



  # plot audio waveforms

    plt.figure(figsize=(16, 10))
    rows = 3
    cols = 3
    n = rows * cols
    for i in range(n):
      plt.subplot(rows, cols, i+1)
      audio_signal = example_audio[i]
      plt.plot(audio_signal)
      plt.yticks(np.arange(-1.2, 1.2, 0.2))
      plt.ylim([-1.1, 1.1])
    plt.show()


# make the sliding window

wavefile = wave.open("demo.wav", "r")
total_frames = wavefile.getnframes()
sampling_rate, samples = wavfile.read('demo.wav')

audio = AudioSegment.from_file("demo.wav")

list_frames = []
for frame in range(0, total_frames-15999, 8000):
    window = samples[frame:frame+15999]


    list_frames.append(window)
    #window.export("short_demo.wav", format= "wav")
    print(export(float(tf.constant(window)))["class_names"])
#plot_waves(list_frames)
print(export(tf.constant(str('data/right.wav')))["class_names"])

#print(wavefile.getparams())
