![Sonic Diffusion header image](./sonicdiffusion_logo.png)


Sonic Diffusion is an AI music generation application that uses Melspectrograms with diffusion modeling to create unique and original compositions. The application is written in Python and is built on top of the PyTorch and NumPy libraries. 

Note: This repository requires Weights and Biases for logging and tracking. To use Weights and Biases, you need to obtain an API key from [www.wandb.ai.](http:www.wandb.ai)

## Sample Audio and Mel spectrograms

Below is a link to sample audio files and their corresponding melspectrogram generated using Sonic Diffusion:

https://markstent.github.io/sonic-diffusion/

![Mel spectrogram](https://github.com/markstent/sonic-diffusion/blob/main/samples/images/4.png?raw=true)

## Installation

To install Sonic Diffusion, first clone the repository:

git clone https://github.com/yourusername/sonic-diffusion.git


Then, navigate to the `sonic-diffusion` directory and install the required Python packages using `pip`:

'cd sonic-diffusion
pip install -r requirements.txt


## Usage

To train the model you will need a folder of Mel spectrograms (there are utility scripts at the bottom of this document).

To train a model using Sonic Diffusion, use the following example script:

```python
python scripts/train_unet.py \
  --dataset_name path/to/dataset \
  --output_dir path/to/save/model \
  --num_epochs 100 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  --gradient_accumulation_steps 8 \
  --save_images_epochs 5 \
  --save_model_epochs 1 \
  --scheduler ddim \
  --model_resume_name name_of_model_in_wandb (if resuming training) \
  --run_id name_of_wandb_model (if resuming training)

```

The script takes several command-line arguments to configure the training process. You need to provide the path to the dataset, the output directory to save the trained model, and other parameters such as the number of epochs, batch size, and gradient accumulation steps. The script also supports resuming training from a previously saved model using the model_resume_name and run_id arguments.

There are a number of hyperparamers that can be set, these can be found in the [training file](scripts/train_unet.py) for now.

## Utility scripts:

Sonic Diffusion provides the following utility scripts:

- [scripts/audio_to_images.py](scripts/audio_to_images.py): This script creates a dataset of Mel spectrograms from a directory of audio files.
- [scripts/audio_mel_audio.py](scripts/audio_mel_audio.py): This module is used for converting audio to mel spectrogram and reconstructing audio from the spectrogram.
- [scripts/train_unet.py](scripts/train_unet.py:): Use this script to train a model.
- [notebooks/train_model.ipynb](notebooks/train_model.ipynb): This Jupyter notebook demonstrates how to convert images and train the model.

## Audio Evaluation

You can evaluate your audio results using the `Stent Weighted Audio Similarity Score (SWASS)`, the repository can be found [here](https://github.com/markstent/audio-similarity).

## To Do

- Documentation
- Add other forms of logging eg Tensorboard

## Citations

Sonic Diffusion uses the following libraries and resources:

- Audio Diffusion: [Audio Diffusion Github](https://github.com/teticio/audio-diffusion)

## Acknowledgement

Thank you to [Robert Dargavel Smith](https://github.com/teticio) for all the help and inspiration for this project.

## License

Sonic Diffusion is licensed under the GNU License. See the `LICENSE` file for more information.
