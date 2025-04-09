import numpy as np
import tensorflow as tf
from keras.src.saving.saving_lib import load_model
import librosa
import wave
from scipy.signal import find_peaks
from collections import deque

# Load the trained model
MODEL_PATH = "saved_model.keras"
model = load_model(MODEL_PATH)
WEIGHTS_PATH = "saved.weights.h5"
model.load_weights(WEIGHTS_PATH)

label_names = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']


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
                        avg_prob = sum(p for c, p in self.smoothing_window if c == most_common) / classes.count(
                            most_common)

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