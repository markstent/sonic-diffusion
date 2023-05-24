import numpy as np
import librosa
from sklearn.metrics import pairwise_distances
from fastdtw import fastdtw
import logging
import argparse
import scipy.signal
import soundfile as sf
import sys
import io
import psutil
from pypesq import pesq
import warnings
import os

# Set up logging
logging.basicConfig(level=logging.INFO)

def calculate_zcr_similarity(original_audio, generated_audio):
    """
    Calculate the Zero Crossing Rate (ZCR) similarity between two audio signals.

    Parameters:
        original_audio (ndarray): The original audio signal.
        generated_audio (ndarray): The generated audio signal.

    Returns:
        float: The ZCR similarity value, indicating the degree of similarity in ZCR
            between the original_audio and generated_audio signals.

    The ZCR similarity value ranges from 0 to 1, where a value closer to 1 indicates a higher degree
    of ZCR similarity, suggesting that the ZCR patterns in the original_audio and generated_audio signals
    are similar. A value closer to 0 indicates a lower degree of ZCR similarity, implying differences
    in the ZCR patterns of the two signals.
    """
    original_zcr = np.mean(np.abs(np.diff(np.sign(original_audio))) > 0)
    generated_zcr = np.mean(np.abs(np.diff(np.sign(generated_audio))) > 0)

    zcr_similarity = 1 - np.abs(original_zcr - generated_zcr)
    return zcr_similarity


def calculate_rhythm_similarity(original_audio, generated_audio, sr):
    """
    Calculate the rhythm similarity between two audio signals.

    Parameters:
        original_audio (ndarray): The original audio signal.
        generated_audio (ndarray): The generated audio signal.
        sr (int): The sampling rate of the audio signals.

    Returns:
        float: The rhythm similarity value, indicating the degree of similarity in rhythm
            between the original_audio and generated_audio signals.

    The rhythm similarity metric compares the rhythm patterns of two audio signals to measure their similarity.
    It identifies the onset times in both audio signals and computes a correlation between these onset times.
    """
    # Detect the onsets
    original_onsets = librosa.onset.onset_detect(y=original_audio, sr=sr, units='time')
    generated_onsets = librosa.onset.onset_detect(y=generated_audio, sr=sr, units='time')

    # Create onset vectors
    original_onset_vector = np.zeros_like(original_audio)
    generated_onset_vector = np.zeros_like(generated_audio)
    for onset in original_onsets:
        original_onset_vector[int(onset*sr)] = 1
    for onset in generated_onsets:
        generated_onset_vector[int(onset*sr)] = 1

    # Calculate the correlation between the onset vectors
    rhythm_similarity = np.corrcoef(original_onset_vector, generated_onset_vector)[0,1]
    # Normalise between 0 and 1
    rhythm_similarity = (np.corrcoef(original_onset_vector, generated_onset_vector)[0,1] + 1) / 2
    logging.info(f'Rhythm similarity: {rhythm_similarity}')
    return rhythm_similarity


def calculate_spectral_flux_similarity(original_audio, generated_audio, target_sr):
    """
        Calculate the Spectral Flux similarity between two audio signals.

        Parameters:
            original_audio (ndarray): The original audio signal.
            generated_audio (ndarray): The generated audio signal.
            target_sr (int): The target sampling rate of the audio signals.

        Returns:
            float: The Spectral Flux similarity value, indicating the degree of similarity
                in spectral flux between the original_audio and generated_audio signals.

        The Spectral Flux similarity value ranges from 0 to 1, where a value closer to 1 indicates a higher degree
        of similarity in the spectral flux of the original_audio and generated_audio signals. A value closer to 0
        suggests a lower degree of similarity, implying differences in the spectral flux characteristics of the two signals.
    """
    original_flux = librosa.onset.onset_strength(y=original_audio, sr=target_sr)
    generated_flux = librosa.onset.onset_strength(y=generated_audio, sr=target_sr)

    spectral_flux_similarity = np.corrcoef(original_flux, generated_flux)[0, 1]
    spectral_flux_similarity = (spectral_flux_similarity + 1) / 2
    logging.info(f'Soectral flux similarity: {spectral_flux_similarity}')
    return spectral_flux_similarity



def calculate_energy_envelope_similarity(original_audio, generated_audio):
    """
    Calculate the Energy Envelope similarity between two audio signals.

    Parameters:
        original_audio (ndarray): The original audio signal.
        generated_audio (ndarray): The generated audio signal.

    Returns:
        float: The Energy Envelope similarity value, indicating the degree of similarity
            in energy envelope between the original_audio and generated_audio signals.

    The Energy Envelope similarity value ranges from 0 to 1, where a value closer to 1 indicates a higher degree
    of similarity in the energy envelopes of the original_audio and generated_audio signals. A value closer to 0
    suggests a lower degree of similarity, implying differences in the energy envelope characteristics of the two signals.
    """
    original_energy_envelope = np.abs(original_audio)
    generated_energy_envelope = np.abs(generated_audio)

    energy_envelope_similarity = np.corrcoef(original_energy_envelope, generated_energy_envelope)[0, 1]
    energy_envelope_similarity = (energy_envelope_similarity + 1) / 2
    logging.info(f'Energy envelope similarity: {energy_envelope_similarity}')
    return energy_envelope_similarity


def calculate_perceptual_similarity(original_audio, generated_audio, sr):
    """Calculates the perceptual similarity between two audio arrays.

    Args:
        original_audio: The numpy array of the first audio file.
        enerated_audio: The numpy array of the second audio file.
        sr: Original sampling rate of the audio data

    Returns:
        The perceptual quality as a number between 0 and 1.
    """
  
    # Resample the arrays to 16kHz
    array1_16k = librosa.resample(y=original_audio, orig_sr=sr, target_sr=16000)
    array2_16k = librosa.resample(y=generated_audio, orig_sr=sr, target_sr=16000)

    # Calculate PESQ score
    score = pesq(array1_16k, array2_16k, 16000)

    # Normalize the PESQ score to be between 0 and 1
    score_normalized = (score + 0.5) / 5
    logging.info(f'Perceptual similarity: {score_normalized}')
    # Return the perceptual quality.
    
    return score_normalized

import librosa

def calculate_spectral_contrast_similarity(original_audio, generated_audio, target_sr):
    """
        Calculate the Spectral Contrast similarity between two audio signals.

        Parameters:
            original_audio (ndarray): The original audio signal.
            generated_audio (ndarray): The generated audio signal.
            target_sr (int): The target sampling rate of the audio signals.

        Returns:
            float: The Spectral Contrast similarity value, indicating the degree of similarity
                in spectral contrast between the original_audio and generated_audio signals.

        The Spectral Contrast similarity value ranges from 0 to 1, where a value closer to 1 indicates a higher degree
        of similarity in the spectral contrast of the original_audio and generated_audio signals. A value closer to 0
        suggests a lower degree of similarity, implying differences in the spectral contrast characteristics of the two signals.
    """
    original_contrast = librosa.feature.spectral_contrast(y=original_audio, sr=target_sr)
    generated_contrast = librosa.feature.spectral_contrast(y=generated_audio, sr=target_sr)

    spectral_contrast_similarity = np.mean(original_contrast == generated_contrast)
    logging.info(f'Spectral contrast similarity: {spectral_contrast_similarity}')
    return spectral_contrast_similarity




def calculate_audio_similarity(original_file, generated_file, target_sr):
    """
    Calculate the overall similarity between two audio files.

    Parameters:
        original_file (str): The path to the original audio file.
        generated_file (str): The path to the generated audio file.
        target_sr (int): The target sampling rate for the audio files.

    Returns:
        float: The overall similarity score between the original and generated audio files. 

    The overall similarity metric incorporates several aspects of the audio signals, including spectral, 
    rhythm, melodic, harmonic similarity and perceptual quality. Each of these aspects is calculated separately 
    and then combined using a weighted average to give the overall similarity score.

    The overall similarity score ranges from 0 to 1, where a value closer to 1 indicates a high degree of 
    similarity between the original and generated audio signals, suggesting that the two signals are very similar 
    in their spectral, rhythmic, melodic, harmonic characteristics and perceptual quality. A value closer to 0 
    indicates a lower degree of similarity, implying substantial differences in these aspects of the two signals.
    """
    # Load the audio files
    original_audio, original_sr = librosa.load(original_file, sr=None)
    generated_audio, generated_sr = librosa.load(generated_file, sr=None)

    # Resample the audio signals to a common sampling rate
      # Choose a target sampling rate
    original_audio = librosa.resample(original_audio, orig_sr=original_sr, target_sr=target_sr)
    generated_audio = librosa.resample(generated_audio, orig_sr=generated_sr, target_sr=target_sr)

    # Ensure the audio signals have the same length
    min_length = min(len(original_audio), len(generated_audio))
    original_audio = original_audio[:min_length]
    generated_audio = generated_audio[:min_length]

    # Set the weights for each similarity metric
    weights = {
    'zcr_similarity': 0.3,
    'rhythm_similarity': 0.2,
    'spectral_flux_similarity': 0.15,
    'energy_envelope_similarity': 0.15,
    'spectral_contrast_similarity': 0.1,
    'perceptual_quality': 0.1
}

    # Calculate each similarity metric
    zcr_similarity = calculate_zcr_similarity(original_audio, generated_audio)
    spectral_contrast_similarity = calculate_spectral_contrast_similarity(original_audio, generated_audio, target_sr)
    spectral_flux_similarity = calculate_spectral_flux_similarity(original_audio, generated_audio, target_sr)
    energy_envelope_similarity = calculate_energy_envelope_similarity(original_audio, generated_audio)
    perceptual_quality = calculate_perceptual_similarity(original_audio, generated_audio, target_sr)
    rhythm_similarity = calculate_rhythm_similarity(original_audio, generated_audio, target_sr)

    # Combine the similarity metrics using weighted average
    similarity = (weights['zcr_similarity'] * zcr_similarity +
              weights['rhythm_similarity'] * rhythm_similarity +
              weights['spectral_flux_similarity'] * spectral_flux_similarity +
              weights['energy_envelope_similarity'] * energy_envelope_similarity +
              weights['spectral_contrast_similarity'] * spectral_contrast_similarity +
              weights['perceptual_quality'] * perceptual_quality
              )

    return similarity

def main():

    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Calculate audio similarity.')
    parser.add_argument('--original', help='Path to the original audio file', default='/Users/mark.stent/Dropbox/Data_Science/Python/sonic-diffusion/data/mp3/AC DC - Thunderstruck.mp3')
    parser.add_argument('--comparison', help='Path to the generated audio file', default='/Users/mark.stent/Dropbox/Data_Science/Python/sonic-diffusion/data/mp3/AC DC - Thunderstruck.mp3')
    parser.add_argument('--sr', help='Sample rate', default=44100)
    args = parser.parse_args()
    
    logging.info('Stent Weighted Audio Similarity Score (SWASS)')
    similarity_score = calculate_audio_similarity(args.original, args.comparison, args.sr)
    logging.info('Stent Weighted Audio Similarity Score (SWASS): {}'.format(similarity_score))

if __name__ == '__main__':
    main()