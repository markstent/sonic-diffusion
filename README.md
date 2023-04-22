![Sonic Diffusion header image](./sonicdiffusion_logo.png)

Sonic Diffusion is an AI music generation application that uses Melspectrograms with diffusion modeling to create unique and original compositions. The application is written in Python and is built on top of the PyTorch and NumPy libraries.

## Installation

To install Sonic Diffusion, first clone the repository:

git clone https://github.com/yourusername/sonic-diffusion.git


Then, navigate to the `sonic-diffusion` directory and install the required Python packages using `pip`:

cd sonic-diffusion
pip install -r requirements.txt


## Usage

 To generate AI music using Sonic Diffusion, run the `generate_music.py` script:

python generate_music.py


 This will generate a new music composition using the default settings. You can customize the music generation process by modifying the settings in the `config.yaml` file.

## Citations

Sonic Diffusion uses the following libraries and resources:

- PyTorch: [https://pytorch.org/](https://pytorch.org/)
- NumPy: [https://numpy.org/](https://numpy.org/)
- Audio Diffusion: [Audio Diffusion Github](https://github.com/teticio/audio-diffusion)

## Acknowledgement

Thank you to [Robert Dargavel Smith](https://github.com/teticio) for all the help and inspiration for this project.

## License

Sonic Diffusion is licensed under the GNU License. See the `LICENSE` file for more information.
