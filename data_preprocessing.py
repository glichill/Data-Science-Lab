import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from tqdm import tqdm
import pandas as pd
from tensorflow import set_random_seed


def explore_dataset(metadata, audio_dataset_path):
    channels = []
    frame_rate = []
    frame_width = []
    length = []
    frame_count = []
    intensity = []
    for index_num, row in tqdm(metadata.iterrows()):
        file_name = os.path.join(os.path.abspath(audio_dataset_path), str(row["path"]))
        audio_segment = AudioSegment.from_file(file_name)
        channels.append(audio_segment.channels)
        frame_rate.append(audio_segment.frame_rate)
        frame_width.append(audio_segment.frame_width)
        length.append(len(audio_segment))
        frame_count.append(audio_segment.frame_count())
        intensity.append(audio_segment.dBFS)

    # Plot duration
    plt.hist(np.array(length), bins=200)
    plt.gca().set(xlabel='Durations(ms)', ylabel='Count')

    length = np.array(length)
    print(f'Index of longest audio: {length.argmax()}')

    # Plot wave of a 20sec audio
    long, long_rate = librosa.load(os.path.join(os.path.abspath(audio_dataset_path), str('dsl_data/audio/speakers/2BqVo8kVB2Skwgyb/8554a780-4479-11e9-a9a5-5dbec3b8816a.wav')),sr=16000)
    librosa.display.waveshow(long)
    plt.gca().set(xlabel='Durations(s)', ylabel='Amplitude')

    # Count how many audio for class
    print(metadata['class'].value_counts())


def preprocess_data(metadata, audio_dataset_path, rate=16000):
    # Now we iterate through every audio file and extract features using Mel-Frequency Cepstral Coefficients
    X = []
    y = []
    for index_num, row in tqdm(metadata.iterrows()):
        file_name = os.path.join(os.path.abspath(audio_dataset_path), str(row["path"]))
        final_class_labels = row["class"]
        sig, rate = librosa.load(file_name, sr=rate)
        sig = librosa.util.fix_length(sig, size=3 * rate)
        mfcc_feat = librosa.feature.mfcc(sig, rate, n_mfcc=13, hop_length=512, n_fft=2048)
        delta_mfcc = librosa.feature.delta(mfcc_feat)
        delta2_mfcc = librosa.feature.delta(mfcc_feat, order=2)
        final_mfcc = np.concatenate((mfcc_feat, delta_mfcc, delta2_mfcc))
        y.append(final_class_labels)
        X.append(final_mfcc)
    Y = pd.get_dummies(y)
    Y = np.array(Y)
    X = np.array(X)

    # Fix random seed for reproducibility
    seed = 7
    set_random_seed(seed)

    return X, Y