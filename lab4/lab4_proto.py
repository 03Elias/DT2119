
# DT2119, Lab 4 End-to-end Speech Recognition
import torch
from torch import nn
import torchaudio

# Variables to be defined --------------------------------------
''' 
train-time audio transform object, that transforms waveform -> spectrogram, with augmentation
''' 
train_audio_transform = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=80
    ),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    torchaudio.transforms.TimeMasking(time_mask_param=35)
)
'''
test-time audio transform object, that transforms waveform -> spectrogram, without augmentation 
'''
test_audio_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_mels=80
)

# Functions to be implemented ----------------------------------
#Elmira
def intToStr(labels):
    '''
        convert list of integers to string
    Args: 
        labels: list of ints
    Returns:
        string with space-separated characters
    '''
    int_to_char = {
        0: "'",
        1: " ",
        2: "a",
        3: "b",
        4: "c",
        5: "d",
        6: "e",
        7: "f",
        8: "g",
        9: "h",
        10: "i",
        11: "j",
        12: "k",
        13: "l",
        14: "m",
        15: "n",
        16: "o",
        17: "p",
        18: "q",
        19: "r",
        20: "s",
        21: "t",
        22: "u",
        23: "v",
        24: "w",
        25: "x",
        26: "y",
        27: "z"
    }

    text = ""

    for label in labels:
        text += int_to_char[int(label)]

    return text
    
    
#Elias
def strToInt(text):
    '''
        convert string to list of integers
    Args:
        text: string
    Returns:
        list of ints
    '''
    char_to_int = {
        "'": 0,
        " ": 1,
        "a": 2,
        "b": 3,
        "c": 4,
        "d": 5,
        "e": 6,
        "f": 7,
        "g": 8,
        "h": 9,
        "i": 10,
        "j": 11,
        "k": 12,
        "l": 13,
        "m": 14,
        "n": 15,
        "o": 16,
        "p": 17,
        "q": 18,
        "r": 19,
        "s": 20,
        "t": 21,
        "u": 22,
        "v": 23,
        "w": 24,
        "x": 25,
        "y": 26,
        "z": 27,
    }

    labels = []
    for char in text.lower():
        if char in char_to_int:
            labels.append(char_to_int[char])

    return labels
      
#Elmira
def dataProcessing(data, transform):
    '''
    process a batch of speech data
    arguments:
        data: list of tuples, representing one batch. Each tuple is of the form
            (waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id)
        transform: audio transform to apply to the waveform
    returns:
        a tuple of (spectrograms, labels, input_lengths, label_lengths) 
        -   spectrograms - tensor of shape B x C x T x M 
            where B=batch_size, C=channel, T=time_frames, M=mel_band.
            spectrograms are padded the longest length in the batch.
        -   labels - tensor of shape B x L where L is label_length. 
            labels are padded to the longest length in the batch. 
        -   input_lengths - list of half spectrogram lengths before padding
        -   label_lengths - list of label lengths before padding
    '''
    # empty lists for one batch
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    # loop over all samples in the batch
    for waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id in data:

        # 1. Convert waveform to mel-spectrogram
        spec = transform(waveform)

        # 2. Change shape:
        # from (channel, mel, time)
        # to   (time, mel)
        spec = spec.squeeze(0).transpose(0, 1)

        # 3. Save spectrogram
        spectrograms.append(spec)

        # 4. Convert text to integer labels
        label = torch.tensor(strToInt(utterance.lower()), dtype=torch.long)
        labels.append(label)

        # 5. Save lengths before padding
        input_lengths.append(spec.shape[0] // 2)
        label_lengths.append(len(label))

    # 6. Pad spectrograms to same length
    spectrograms = nn.utils.rnn.pad_sequence(
        spectrograms,
        batch_first=True
    )

    # 7. Pad labels to same length
    labels = nn.utils.rnn.pad_sequence(
        labels,
        batch_first=True
    )

    # 8. Change shape:
    # from (batch, time, mel)
    # to   (batch, channel, mel, time)
    spectrograms = spectrograms.unsqueeze(1).transpose(2, 3)

    return spectrograms, labels, input_lengths, label_lengths
   
    
    
#Elias 
def greedyDecoder(output, blank_label=28):
    '''
    decode a batch of utterances 
    arguments:
        output: network output tensor, shape B x T x C where B=batch_size, T=time_steps, C=characters
        blank_label: id of the blank label token
    returns:
        list of decoded strings
    '''
    # Pick the most probable label at each time step.
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []

    for batch_idx in range(arg_maxes.size(0)):
        decode = []
        prev = None
        for idx in arg_maxes[batch_idx]:
            idx = int(idx.item())
            if idx != blank_label and idx != prev:
                decode.append(idx)
            prev = idx
        decodes.append(intToStr(decode))

    return decodes
#Elmira
def levenshteinDistance(ref,hyp):
    '''
    calculate levenshtein distance (edit distance) between two sequences
    arguments:
        ref: reference sequence
        hyp: sequence to compare against the reference
    output:
        edit distance (int)
    '''
    # number of rows and columns
    rows = len(ref) + 1
    cols = len(hyp) + 1

    # create matrix filled with zeros
    matrix = [[0 for j in range(cols)] for i in range(rows)]

    # fill first column
    for i in range(rows):
        matrix[i][0] = i

    # fill first row
    for j in range(cols):
        matrix[0][j] = j

    # fill the rest of the matrix
    for i in range(1, rows):
        for j in range(1, cols):

            # if characters are the same
            if ref[i - 1] == hyp[j - 1]:
                matrix[i][j] = matrix[i - 1][j - 1]

            # if characters are different
            else:
                insertion = matrix[i][j - 1] + 1
                deletion = matrix[i - 1][j] + 1
                substitution = matrix[i - 1][j - 1] + 1

                matrix[i][j] = min(
                    insertion,
                    deletion,
                    substitution
                )

    # final distance
    return matrix[len(ref)][len(hyp)]

