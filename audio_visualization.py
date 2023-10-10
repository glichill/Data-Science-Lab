import os
import random
from pydub import AudioSegment
import librosa
import librosa.display
import matplotlib.pyplot as plt


# Get a random audio file from your specified path
def get_random_audio_file(base_path):
    speaker_dir = random.choice([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    speaker_path = os.path.join(base_path, speaker_dir)
    audio_file = random.choice([f for f in os.listdir(speaker_path) if f.endswith('.wav')])
    return os.path.join(speaker_path, audio_file)


# Audio analysis and visualization
def analyze_audio(filename):
    audio_segment = AudioSegment.from_file(filename)
    print(f"Channels: {audio_segment.channels}")
    print(f"Sample width: {audio_segment.sample_width}")
    print(f"Frame rate (sample rate): {audio_segment.frame_rate}")
    print(f"Frame width: {audio_segment.frame_width}")
    print(f"Length (ms): {len(audio_segment)}")
    print(f"Frame count: {audio_segment.frame_count()}")
    print(f"Intensity: {audio_segment.dBFS}")

    sig_audio, rate = librosa.load(filename, sr=16000)
    sig_audio = librosa.util.fix_length(sig_audio, size=3 * 16000)
    mfcc_audio = librosa.feature.mfcc(sig_audio, rate, n_mfcc=13, hop_length=512, n_fft=2048)

    delta_mfcc_audio = librosa.feature.delta(mfcc_audio)
    delta2_mfcc_audio = librosa.feature.delta(mfcc_audio, order=2)

    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    librosa.display.specshow(mfcc_audio)
    plt.ylabel('MFCC')
    plt.colorbar()

    plt.subplot(3, 1, 2)
    librosa.display.specshow(delta_mfcc_audio)
    plt.ylabel('MFCC-$\Delta$')
    plt.colorbar()

    plt.subplot(3, 1, 3)
    librosa.display.specshow(delta2_mfcc_audio, sr=16000, x_axis='time')
    plt.ylabel('MFCC-$\Delta^2$')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
