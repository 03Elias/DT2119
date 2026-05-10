
import torch

from lab4_proto import (
    dataProcessing,
    test_audio_transform
)

# Load example batch provided in the lab
example = torch.load("lab4_example.pt")

# Get input batch
data = example["data"]

# Run data processing
spectrograms, labels, input_lengths, label_lengths = dataProcessing(
    data,
    test_audio_transform
)

# Print results
print("spectrograms shape:", spectrograms.shape)
print("labels shape:", labels.shape)
print("input_lengths:", input_lengths)
print("label_lengths:", label_lengths)