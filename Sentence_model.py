"""
M7 telephone booth project, Floor and Marit
This code is adapted from the TensorFlow tutorial:
'Simple audio recognition: Recognizing keywords'
URL: https://www.tensorflow.org/tutorials/audio/simple_audio#export_the_model_with_preprocessing
Copyright: The TensorFlow Authors
Accessed: April 7, 2025
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
SEQUENCE_LENGTH = 160000


# set the dataset path to the directory containing the sets of sentences
DATASET_PATH = 'data/mini_speech_commands_extracted/mini_speech_commands'
data_dir = pathlib.Path(DATASET_PATH)

# get and print all folders of the sound files
folders = np.array(tf.io.gfile.listdir(str(data_dir)))
folders = folders[(folders != 'README.md') & (folders != '.DS_Store')]
print('Folders:', folders)

# Generate a tf.data.Dataset from audio files in a directory:
train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=16, #set bach size to a number that is not too high, since we only have limited data set
    validation_split=0.2, #saves 20% as validation data
    seed=0,
    output_sequence_length=SEQUENCE_LENGTH, # set every audio clip to 10 second (happens because they are all 16kHz)
    subset='both')

label_names = np.array(train_ds.class_names)
print()
print("label names:", label_names)

# dataset only contains single channel audio; drop extra axis by using tf.squeeze()
def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels

# apply squeeze to both datasets to remove the channel variable
# tf.data.AUTOTUNE tells TensorFlow to automatically determine the best parallelism for this operation
train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

# split the validation data into test and validation
test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

#get some example data to see what we are doing
for example_audio, example_labels in train_ds.take(1):
    print(example_audio.shape)
    print(example_labels.shape)

#function to plot the shape of the wave of the audio file
def plot_waves():
    # plot audio waveforms
    print(label_names[[1,1,3,0]])

    plt.figure(figsize=(16, 10))
    rows = 3
    cols = 3
    n = rows * cols
    for i in range(n):
      plt.subplot(rows, cols, i+1)
      audio_signal = example_audio[i]
      plt.plot(audio_signal)
      plt.title(label_names[example_labels[i]])
      plt.yticks(np.arange(-1.2, 1.2, 0.2))
      plt.ylim([-1.1, 1.1])
    plt.show()
#plot_waves()

#funtion to turn the wave into a spectrogram
def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT. try to make it square here
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
      #for 16000 and 16000 it is Spectrogram shape: (1, 8193, 1) so width 1 and height 8193
      #16000/128 = 125, so to get the frame_step, do number of frames/125
  # Obtain the magnitude of the STFT (by dropping the phase).
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

# display a spectrogram
def plot_spectrogram(spectrogram, ax):
  # if len(spectrogram.shape) > 2:
  #   assert len(spectrogram.shape) == 3
  spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  print(f"log spec is:{log_spec}")
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

def plot_waveform_spectrogram(waveform, spectrogram, label):
  fig, axes = plt.subplots(2, figsize=(12, 8))
  timescale = np.arange(waveform.shape[0])
  print(f"waveform printed {waveform.numpy()}")
  axes[0].plot(timescale, waveform.numpy())
  axes[0].set_title('Waveform')
  axes[0].set_xlim([0, SEQUENCE_LENGTH])

  print(f"spectrogram printed {spectrogram.numpy()}")
  plot_spectrogram(spectrogram.numpy(), axes[1])
  axes[1].set_title('Spectrogram')
  axes[1].set_xlim([0, SEQUENCE_LENGTH])#########################################################################################
  plt.suptitle(label.title())
  plt.show()

for i in range(3):
    label = label_names[example_labels[i]]
    #these waveforms have len() 16000
    waveform = example_audio[i]
    spectrogram = get_spectrogram(waveform)

    print('Label:', label)
    print('Waveform shape:', waveform.shape)
    print('Spectrogram shape:', spectrogram.shape)
    plot_waveform_spectrogram(waveform, spectrogram, label)