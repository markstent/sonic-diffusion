from datasets import Dataset
import os

arrow_file_path = "/Users/mark.stent/Dropbox/Data_Science/Python/sonic-diffusion/data/test/data/train/data-00000-of-00001.arrow"

# Load the Arrow file as a dataset
dataset = Dataset.from_file(arrow_file_path)

# Convert the dataset to a list
samples = dataset.to_list()

# Iterate through the samples and print them
for index, sample in enumerate(samples):
    print(f"Sample {index + 1}:")
    print(sample)
    print("\n")
