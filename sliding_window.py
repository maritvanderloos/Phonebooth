import pathlib
import wave
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras.src.saving.saving_lib import load_model
import librosa
from scipy.signal import find_peaks
from collections import deque

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

# make a spectrogram of the waveforms (specific for longer audios)
def get_long_spectrogram(waveform, number_of_frames):
    frame_steps = number_of_frames/125
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=frame_steps)
    # Obtain the magnitude of the STFT (by dropping the phase).
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def plot_spectrogram(spectrogram, ax):
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

wavefile = wave.open("demo.wav", "r")
total_frames = wavefile.getnframes()

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

    x = get_long_spectrogram(x, )
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
export(tf.constant(str('demo.wav')))

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


# wavefile = wave.open("demo.wav", "r")
# total_frames = wavefile.getnframes()
# sampling_rate, samples = wavfile.read('demo.wav')
#
# audio = AudioSegment.from_file("demo.wav")
#
# spectrogram = get_long_spectrogram(wavefile, total_frames)
#
# fig, axes = plt.subplots(1, figsize=(12, 8))
# plot_spectrogram(spectrogram.numpy(), axes[1])
# axes[1].set_title('Spectrogram')
# plt.show()

# list_frames = []
# for frame in range(0, total_frames-15999, 8000):
#     window = samples[frame:frame+15999]
#
#
#     list_frames.append(window)
#     #window.export("short_demo.wav", format= "wav")
#     print(export(float(tf.constant(window)))["class_names"])
# #plot_waves(list_frames)
# print(export(tf.constant(str('data/right.wav')))["class_names"])

#print(wavefile.getparams())

class KeywordDetector:
    def __init__(self, window_size=16000, hop_size=8000, confidence_threshold=0.7):
        """
        Initialize the keyword detector.
        
        Args:
            window_size: Size of the sliding window in samples (default: 16000 for 1 second at 16kHz)
            hop_size: Number of samples to move the window (default: 8000 for 50% overlap)
            confidence_threshold: Minimum confidence score to consider a detection valid
        """
        self.window_size = window_size
        self.hop_size = hop_size
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = deque(maxlen=5)  # Store last 5 predictions for smoothing
        
    def get_spectrogram(self, waveform):
        """Convert waveform to spectrogram using the same parameters as training."""
        spectrogram = tf.signal.stft(
            waveform, frame_length=255, frame_step=128)
        spectrogram = tf.abs(spectrogram)
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram
    
    def process_audio(self, audio_path):
        """
        Process a longer audio file and detect keywords.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            List of detected keywords with their timestamps
        """
        # Load and preprocess audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Apply voice activity detection
        intervals = librosa.effects.split(audio, top_db=20)
        
        detections = []
        
        # Process each speech segment
        for start, end in intervals:
            segment = audio[start:end]
            
            # Apply sliding window
            for i in range(0, len(segment) - self.window_size, self.hop_size):
                window = segment[i:i + self.window_size]
                
                # Convert to spectrogram
                spec = self.get_spectrogram(tf.convert_to_tensor(window))
                spec = tf.expand_dims(spec, 0)  # Add batch dimension
                
                # Get model predictions
                predictions = model(spec, training=False)
                probabilities = tf.nn.softmax(predictions, axis=-1)
                
                # Get highest probability and class
                max_prob = tf.reduce_max(probabilities)
                predicted_class = tf.argmax(predictions, axis=-1)
                
                if max_prob > self.confidence_threshold:
                    # Add to smoothing window
                    self.smoothing_window.append((predicted_class, max_prob))
                    
                    # If we have enough samples in the smoothing window
                    if len(self.smoothing_window) == self.smoothing_window.maxlen:
                        # Get most common prediction in the window
                        classes, probs = zip(*self.smoothing_window)
                        most_common = max(set(classes), key=classes.count)
                        avg_prob = sum(p for c, p in self.smoothing_window if c == most_common) / classes.count(most_common)
                        
                        if avg_prob > self.confidence_threshold:
                            timestamp = start + i
                            detections.append({
                                'keyword': label_names[most_common],
                                'timestamp': timestamp / sr,  # Convert to seconds
                                'confidence': float(avg_prob)
                            })
        
        return detections

def main():
    # Example usage
    detector = KeywordDetector()
    audio_path = "demo.wav"
    
    print("Processing audio file...")
    detections = detector.process_audio(audio_path)
    
    print("\nDetected keywords:")
    for detection in detections:
        print(f"Keyword: {detection['keyword']}")
        print(f"Time: {detection['timestamp']:.2f} seconds")
        print(f"Confidence: {detection['confidence']:.2f}")
        print("---")

if __name__ == "__main__":
    main()
