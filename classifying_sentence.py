import time
import tensorflow as tf
from keras.src.saving.saving_lib import load_model

#recording audio
from recording import Recording
#imports for playing a sound
from pydub import AudioSegment
from pydub.playback import play

#start model
import keyboard  # using module keyboard

#reading data from the arduino
from serial.tools import list_ports
import serial
import time
import csv

# Identify the correct port
ports = list_ports.comports()
for port in ports: print(port)

def read_turns(turn_amount):
    # Open the serial com
    serialCom = serial.Serial('COM3',115200)

    # How many data points to record
    kmax = turn_amount

    # Loop through and collect data as it is available
    for k in range(kmax):
        try:
            # Read the line
            s_bytes = serialCom.readline()
            decoded_bytes = s_bytes.decode("utf-8").strip('\r\n')
            # print(decoded_bytes)

            # read the values
            value = [float(x) for x in decoded_bytes.split()]
            print(value)
        except:
            print("Error encountered, line was not recorded.")
    return True

# Load the trained model
MODEL_PATH = "sentence2_saved_model.keras"
model = load_model(MODEL_PATH, compile=True)
WEIGHTS_PATH = "sentence2.saved.weights.h5"
model.load_weights(WEIGHTS_PATH)

label_names = ['kind', 'unkind']
SEQUENCE_LENGTH = 80000
record = Recording(400)
start_record = Recording(800)

print(model)

#funtion to turn the wave into a spectrogram
def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT. try to make it square here
  spectrogram = tf.signal.stft(
      waveform, frame_length=1024, frame_step=256)
  # Obtain the magnitude of the STFT (by dropping the phase).
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

# Basic audio preprocessing
def normalize_audio(audio):
    """Normalize audio to have consistent volume levels."""
    return audio / tf.reduce_max(tf.abs(audio))

class predict_label(tf.Module):
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

#put the model into the predict class
predicted = predict_label(model)
restart = False
def blurred_line():
    #start_record.start_recording('data/start_sound.wav')
    intro = AudioSegment.from_wav('data/Recording (13).wav')
    play(intro)

    kindness_counter = 0
    data_path_question = 'data/question_participant.wav'
    for i in range(1, 4):
        #record question
        record.start_recording(data_path_question)

        #predict label
        predicted_info = predicted(tf.constant(str(data_path_question)))
        predicted_class = predicted_info['class_names'].numpy()[0]

        # check whether 'unkind' or 'kind' and direct to appropriate file
        file_dir = 'data/answers'
        print(kindness_counter)
        if predicted_class.find(b'unkind') == 0:
            file_dir = str(f'{file_dir}/unkind/unkind_answer{i}.wav')
            print('unkind')
        else:
            file_dir = str(f'{file_dir}/kind/kind_answer{i}.wav')
            kindness_counter += 1
            print("kind")
        #play the right answer sound
        answer = AudioSegment.from_wav(file_dir)
        play(answer)

    if kindness_counter >= 2:
        outro = AudioSegment.from_wav('data/kind_outro.wav')
    else:
        outro = AudioSegment.from_wav('data/unkind_outro.wav')
    play(outro)
    print("program ended, waiting for turns to start the next session")
    time.sleep(30)
    #restart = True
    pass

#controlling the start and restart
while True:  # making a loops
    if read_turns(4):  # when enough turns are  made, the program starts
        restart = True
    if restart:
        blurred_line()

