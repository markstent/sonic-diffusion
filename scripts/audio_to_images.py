"""
Script to create a dataset of Mel spectrograms from a directory of audio files.

Usage:
    python audio_to_images.py [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR]
                              [--resolution RESOLUTION] [--hop_length HOP_LENGTH]
                              [--sample_rate SAMPLE_RATE] [--n_fft N_FFT]

Arguments:
    --input_dir INPUT_DIR:
        The directory path where the audio files are located. Default is "/Users/mark.stent/Dropbox/Data_Science/Python/sonic-diffusion/data/test".

    --output_dir OUTPUT_DIR:
        The directory path where the generated dataset will be saved. Default is "/Users/mark.stent/Dropbox/Data_Science/Python/sonic-diffusion/data/test/data".

    --resolution RESOLUTION:
        The resolution of the generated spectrograms. Can be provided as a single integer for a square resolution
        or as a string in the format "width,height". Default is "256".

    --hop_length HOP_LENGTH:
        The hop length used in the Mel spectrogram calculation. Default is 512.

    --sample_rate SAMPLE_RATE:
        The sample rate of the audio files. Default is 22050.

    --n_fft N_FFT:
        The number of FFT points used in the Mel spectrogram calculation. Default is 2048.
"""

import argparse
import io
import logging
import os
import re

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Image, Value
from diffusers.pipelines.audio_diffusion import Mel
from tqdm.auto import tqdm

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger("audio_to_images")


def main(args):
    
    """
    Main function to convert audio files to Mel spectrograms and create a dataset.

    Args:
        args: An object containing the command-line arguments.

    Returns:
        None
    """
    
    mel = Mel(
        x_res=args.resolution[0],
        y_res=args.resolution[1],
        hop_length=args.hop_length,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    audio_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(args.input_dir)
        for file in files
        if re.search("\.(mp3|wav|m4a)$", file, re.IGNORECASE)
    ]
    examples = []
    try:
        for audio_file in tqdm(audio_files):
            try:
                mel.load_audio(audio_file)
            except KeyboardInterrupt:
                raise
            except:
                continue
            for slice in range(mel.get_number_of_slices()):
                image = mel.audio_slice_to_image(slice)
                assert image.width == args.resolution[0] and image.height == args.resolution[1], "Wrong resolution"
                # skip completely silent slices
                if all(np.frombuffer(image.tobytes(), dtype=np.uint8) == 255):
                    logger.warn("File %s slice %d is completely silent", audio_file, slice)
                    continue
                with io.BytesIO() as output:
                    image.save(output, format="PNG")
                    bytes = output.getvalue()
                examples.extend(
                    [
                        {
                            "image": {"bytes": bytes},
                            "audio_file": audio_file,
                            "slice": slice,
                        }
                    ]
                )
    except Exception as e:
        print(e)
    finally:
        if len(examples) == 0:
            logger.warn("No valid audio files were found.")
            return
        ds = Dataset.from_pandas(
            pd.DataFrame(examples),
            features=Features(
                {
                    "image": Image(),
                    "audio_file": Value(dtype="string"),
                    "slice": Value(dtype="int16"),
                }
            ),
        )
        dsd = DatasetDict({"train": ds})
        dsd.save_to_disk(os.path.join(args.output_dir))
        print(f"Completed...{len(ds)} spectrograms created")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset of Mel spectrograms from directory of audio files.")
    parser.add_argument("--input_dir", type=str, default="/Users/mark.stent/Dropbox/Data_Science/Python/sonic-diffusion/data/test")
    parser.add_argument("--output_dir", type=str, default="/Users/mark.stent/Dropbox/Data_Science/Python/sonic-diffusion/data/test/data")
    parser.add_argument(
        "--resolution",
        type=str,
        default="256",
        help="Either square resolution or width,height.",
    )
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--n_fft", type=int, default=2048)
    args = parser.parse_args()

    if args.input_dir is None:
        raise ValueError("You must specify an input directory for the audio files.")

    # Handle the resolutions.
    try:
        args.resolution = (int(args.resolution), int(args.resolution))
    except ValueError:
        try:
            args.resolution = tuple(int(x) for x in args.resolution.split(","))
            if len(args.resolution) != 2:
                raise ValueError
        except ValueError:
            raise ValueError("Resolution must be a tuple of two integers or a single integer.")
    assert isinstance(args.resolution, tuple)
    
    
    main(args)
