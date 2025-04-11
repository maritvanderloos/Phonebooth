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

SEQUENCE_LENGTH = 80000

TRAIN = False
WEIGHTS_PATH = "sentence2.saved.weights.h5"
MODEL_PATH = "sentence2_saved_model.keras"

# Basic audio preprocessing
def normalize_audio(audio):
    """Normalize audio to have consistent volume levels."""
    return audio / tf.reduce_max(tf.abs(audio))

# set the dataset path to the directory containing the sets of sentences
DATASET_PATH = 'data/data_collection'
data_dir = pathlib.Path(DATASET_PATH)

# get and print all folders of the sound files
folders = np.array(tf.io.gfile.listdir(str(data_dir)))
folders = folders[(folders != 'README.md') & (folders != '.DS_Store')]
print('Folders:', folders)

# Generate a tf.data.Dataset from audio files in a directory:
train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size= 4, #set bach size to a number that is not too high, since we only have limited data set
    validation_split=0.2, #saves 20% as validation data
    seed=0,
    output_sequence_length=SEQUENCE_LENGTH, # set every audio clip to 10 second (happens because they are all 16kHz)
    subset='both')

label_names = np.array(train_ds.class_names)
print()
print("label names:", label_names)

# dataset only contains single channel audio; drop extra axis by using tf.squeeze()
def squeeze_and_normalize(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    normalize_audio(audio)
    return audio, labels

# apply squeeze to both datasets to remove the channel variable
# tf.data.AUTOTUNE tells TensorFlow to automatically determine the best parallelism for this operation
train_ds = train_ds.map(squeeze_and_normalize, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze_and_normalize, tf.data.AUTOTUNE)

# split the validation data into test and validation
test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

#get some example data to see what we are doing
for example_audio, example_labels in train_ds.take(1):
    print(example_audio.shape)
    print(example_labels.shape)
    normalize_audio(example_audio)

#function to plot the shape of the wave of the audio file
def plot_waves():
    # plot audio waveforms
    print(label_names[[1,1,0,0]])

    plt.figure(figsize=(16, 10))
    rows = 1
    cols = 2
    n = rows * cols
    for i in range(n):
      plt.subplot(rows, cols, i+1)
      audio_signal = example_audio[i]
      plt.plot(audio_signal)
      plt.title(label_names[example_labels[i]])
      plt.yticks(np.arange(-1.2, 1.2, 0.2))
      plt.ylim([-1.1, 1.1])
    plt.show()


#funtion to turn the wave into a spectrogram
def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT. try to make it square here
  spectrogram = tf.signal.stft(
      waveform, frame_length=1024, frame_step=256)
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

#display the waveform with the corresponding spectogram
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

# for i in range(2):
#     label = label_names[example_labels[i]]
#     #these waveforms have len() 16000
#     waveform = example_audio[i]
#     spectrogram = get_spectrogram(waveform)
#
#     print('Label:', label)
#     print('Waveform shape:', waveform.shape)
#     print('Spectrogram shape:', spectrogram.shape)
#     plot_waveform_spectrogram(waveform, spectrogram, label)

# make spectrogram datasets from the audio datasets
def make_spec_ds(ds):
  return ds.map(
      map_func=lambda audio,label: (get_spectrogram(audio), label),
      num_parallel_calls=tf.data.AUTOTUNE)

#making the training, validation and testing datasets
train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)
test_spectrogram_ds = make_spec_ds(test_ds)

for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
  break

def plot_multiple_spectrograms():
    rows = 3
    cols = 3
    n = rows * cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 9))

    for i in range(n):
      r = i // cols
      c = i % cols
      ax = axes[r][c]
      plot_spectrogram(example_spectrograms[i].numpy(), ax)
      ax.set_title(label_names[example_spect_labels[i].numpy()])

    plt.show()

"""building and training model"""
# reduce latency:
train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

input_shape = example_spectrograms.shape[1:]
print('Input shape:', input_shape)
num_labels = len(label_names)

# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Resizing(32, 32),
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(num_labels, activation='softmax'),
])

model.summary()

# configuring the keras model with Adam optimizer and cross-entropy loss
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

if TRAIN:
    # Train the model over 10 epochs for demonstration purposes:
    EPOCHS = 30
    print(f"Training over {EPOCHS} epochs")
    history = model.fit(
        train_spectrogram_ds,
        validation_data=val_spectrogram_ds,
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=4),
    )
    model.save(MODEL_PATH)
    model.save_weights(WEIGHTS_PATH)
else: # just load the weights
    print("Loading weights from " + WEIGHTS_PATH)
    model.load_weights(WEIGHTS_PATH)

# plot training and validation loss curves
def plot_loss_curves():
  metrics = history.history
  plt.figure(figsize=(16,6))
  plt.subplot(1,2,1)
  plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
  plt.legend(['loss', 'val_loss'])
  plt.ylim([0, max(plt.ylim())])
  plt.xlabel('Epoch')
  plt.ylabel('Loss [CrossEntropy]')

  plt.subplot(1,2,2)
  plt.plot(history.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
  plt.legend(['accuracy', 'val_accuracy'])
  plt.ylim([0, 100])
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy [%]')
  plt.show()

class ExportModel(tf.Module):
  def __init__(self, model):
    self.model = model

    # Accept either a string-filename or a batch of waveforms.
    # YOu could add additional signatures for a single wave, or a ragged-batch.
    #__call__ makes the class object callable
    self.__call__.get_concrete_function(
        x=tf.TensorSpec(shape=(), dtype=tf.string))
    self.__call__.get_concrete_function(
       x=tf.TensorSpec(shape=[None, SEQUENCE_LENGTH], dtype=tf.float32))


  @tf.function
  def __call__(self, x):
    # If they pass a string, load the file and decode it.
    if x.dtype == tf.string:
      x = tf.io.read_file(x)
      x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=SEQUENCE_LENGTH,)
      x = tf.squeeze(x, axis=-1)
      normalize_audio(x)
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


print(f"model: {model}")
export = ExportModel(model)
print("unkind answer from marit")
print()
print(export(tf.constant(str('data/kind_test2.wav'))))