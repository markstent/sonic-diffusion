"""
Module for converting audio to mel spectrogram and reconstructing audio from the spectrogram.

Usage:
    python audio_reconstruction.py [--audio_path AUDIO_PATH] [--target_folder TARGET_FOLDER]

Arguments:
    --audio_path AUDIO_PATH:
        The path to the audio file or folder containing audio files.

    --target_folder TARGET_FOLDER:
        The path to the target folder to save the reconstructed audio files.
"""

import os
import librosa
import librosa.display
import soundfile as sf
from tqdm import tqdm
import logging
from pathlib import Path

def convert_audio_to_reconstructed_audio(audio_path, target_folder):
    """
    Convert audio to mel spectrogram and save the spectrogram image.
    Then, convert the spectrogram image back to audio and save the audio file.

    Args:
        audio_path (str): Path to the audio file or folder containing audio files.
        target_folder (str): Path to the target folder to save the reconstructed audio files.

    Returns:
        None
    """
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    
    audio_path_converted = Path(audio_path)

    if os.path.isdir(audio_path_converted):
        # Input is a folder
        file_list = [os.path.join(audio_path_converted, f) for f in os.listdir(audio_path_converted) if f.endswith(('.mp3', '.wav', '.flac'))]
        print(file_list)
    elif os.path.isfile(audio_path_converted) and audio_path_converted.endswith(('.mp3', '.wav', '.flac')):
        # Input is a single audio file
        file_list = [audio_path_converted]
    else:
        print("Invalid input. Please provide a valid audio file or folder path.")
        return

    os.makedirs(target_folder, exist_ok=True)

    for audio_file in tqdm(file_list, desc="Converting files"):
        # Load audio
        audio, sr = librosa.load(audio_file)

        # Convert audio to mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)

        # Convert mel spectrogram back to audio
        reconstructed_audio = librosa.feature.inverse.mel_to_audio(mel_spectrogram)

        # Save the reconstructed audio as MP3
        reconstructed_audio_filename = os.path.splitext(os.path.basename(audio_file))[0] + '_reconstructed.mp3'
        reconstructed_audio_path = os.path.join(target_folder, reconstructed_audio_filename)
        sf.write(reconstructed_audio_path, reconstructed_audio, sr, format='mp3')

        logging.info(f"Conversion completed: {audio_file} -> {reconstructed_audio_path}")



audio_path = "/content/drive/MyDrive/data/mp3/"  # Replace with your audio file or folder path
target_folder = "/content/drive/MyDrive/data/mp3_mel_mp3/"  # Replace with the desired target folder path

convert_audio_to_reconstructed_audio(audio_path, target_folder)