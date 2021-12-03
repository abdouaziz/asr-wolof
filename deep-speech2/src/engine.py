
import tensorflow as tf
import librosa
import pandas as pd
import numpy as np


def get_alphabet(args):

  df = pd.read_csv(args.train_file)

  uniques = set()

  for line in df["transcription"]:

    uniques.update(w.lower() for w in line)

  return sorted(uniques)


def create_spectrogram(signals):

    stfts = tf.signal.stft(signals, frame_length=256,
                           frame_step=80, fft_length=256)
    spectrograms = tf.math.pow(tf.abs(stfts), 0.5)
    return spectrograms


def generate_input_from_audio_file(path_to_audio_file, resample_to=8000):
    signal, sample_rate = librosa.core.load(path_to_audio_file)
    #signal = pad_audio(signal , sample_rate , 6)

    if signal.shape[0] == 2:
        signal = np.mean(signal, axis=0)
    signal_resampled = librosa.core.resample(signal, sample_rate, resample_to)

    X = create_spectrogram(signal_resampled)

    # Normalisation

    means = tf.math.reduce_mean(X, 1, keepdims=True)
    stddevs = tf.math.reduce_std(X, 1, keepdims=True)
    X = tf.divide(tf.subtract(X, means), stddevs)

    return X


def generate_target_output_from_text(target_text , args):

    #space_token = ' '
    #end_token = '>'
    #blank_token = '%'
    #pp = ['é', 'è', 'œ','ù']

    alphabet = get_alphabet(args)

    char_to_index = {}

    for idx, char in enumerate(alphabet):
        char_to_index[char] = idx

    y = []

    for char in target_text:
        y.append(char_to_index[char.lower()])

    #y=tf.expand_dims(tf.convert_to_tensor(y) , axis=0)

    return y
