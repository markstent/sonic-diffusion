import os
import librosa
import librosa.display
import soundfile as sf
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import skimage.io
from skimage.transform import resize

def convert_audio_to_reconstructed_audio(audio_path, target_folder):
    """
    Convert audio to mel spectrogram and save the spectrogram image (256x256) for the first 5 seconds.
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

        # Select the first 5 seconds of the audio
        duration = 5  # Duration in seconds
        audio_duration = librosa.get_duration(y=audio, sr=sr)
        if audio_duration > duration:
            audio = audio[:int(sr * duration)]  # Slice the audio array

        print(f"Audio file: {audio_file}")
        print(f"Original sampling rate: {sr}")
        print(f"Original audio shape: {audio.shape}")

        # Convert audio to magnitude spectrogram
        spectrogram = np.abs(librosa.stft(y=audio))

        # Save the spectrogram as 256x256 image
        spectrogram_image_filename = os.path.splitext(os.path.basename(audio_file))[0] + '_spectrogram.png'
        spectrogram_image_path = os.path.join(target_folder, spectrogram_image_filename)

        # Resize spectrogram to 256x256
        spectrogram_resized = resize(spectrogram, (256, 256), mode='reflect')

        # Normalize spectrogram to 0-255 for skimage.io.imsave
        spectrogram_norm = librosa.amplitude_to_db(spectrogram_resized, ref=np.max)
        spectrogram_norm = (spectrogram_norm - spectrogram_norm.min()) / (spectrogram_norm.max() - spectrogram_norm.min()) * 255
        spectrogram_norm = spectrogram_norm.astype(np.uint8)

        # Save the spectrogram image using skimage.io.imsave
        skimage.io.imsave(spectrogram_image_path, spectrogram_norm, cmap='gray')

        # Convert spectrogram back to audio using the Griffin-Lim algorithm
        reconstructed_audio = librosa.griffinlim(S=spectrogram_resized, n_iter=100)

        # Save the reconstructed audio as MP3 with the original sampling rate
        reconstructed_audio_filename = os.path.splitext(os.path.basename(audio_file))[0] + '_reconstructed.mp3'
        reconstructed_audio_path = os.path.join(target_folder, reconstructed_audio_filename)
        sf.write(reconstructed_audio_path, reconstructed_audio, sr, format='mp3')

        logging.info(f"Conversion completed: {audio_file} -> {reconstructed_audio_path}")

# Rest of the code remains the same




audio_path = "data/mp3/mp3Evalclip"  # Replace with your audio file or folder path
target_folder = "data/mp3_reconstructed"  # Replace with the desired target folder path

convert_audio_to_reconstructed_audio(audio_path, target_folder)