
import torch

from lab4_proto import (
    strToInt,
    intToStr,
    dataProcessing,
    test_audio_transform,
    greedyDecoder,
)


def _print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _print_check(name, passed):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")


def test_text_mapping_roundtrip():
    _print_section("Test 1: strToInt / intToStr Mapping")
    text = "it's a test"
    print(f"Input text: {text}")

    labels = strToInt(text)
    print(f"Encoded labels: {labels}")

    reconstructed = intToStr(labels)
    print(f"Decoded text: {reconstructed}")

    expected = "it's a test"
    print(f"Expected decoded text: {expected}")

    _print_check("Roundtrip text equality", reconstructed == expected)
    assert reconstructed == expected


def test_data_processing_against_example():
    _print_section("Test 2: dataProcessing Against lab4_example.pt")
    print("Loading example file: lab4_example.pt")
    example = torch.load("lab4_example.pt")
    data = example["data"]
    print(f"Loaded example batch size: {len(data)}")

    print("Running dataProcessing(...) with test_audio_transform")
    spectrograms, labels, input_lengths, label_lengths = dataProcessing(
        data,
        test_audio_transform,
    )

    print(f"Computed spectrograms shape: {tuple(spectrograms.shape)}")
    print(f"Expected spectrograms shape: {tuple(example['spectrograms'].shape)}")
    spectrogram_shape_ok = spectrograms.shape == example["spectrograms"].shape
    _print_check("Spectrogram shape", spectrogram_shape_ok)
    assert spectrogram_shape_ok

    print(f"Computed labels shape: {tuple(labels.shape)}")
    print(f"Expected labels shape: {tuple(example['labels'].shape)}")
    labels_shape_ok = labels.shape == example["labels"].shape
    _print_check("Labels shape", labels_shape_ok)
    assert labels_shape_ok

    spectrogram_values_ok = torch.allclose(
        spectrograms,
        example["spectrograms"],
        atol=1e-6,
    )
    _print_check("Spectrogram values (allclose, atol=1e-6)", spectrogram_values_ok)
    assert spectrogram_values_ok

    labels_values_ok = torch.equal(labels, example["labels"])
    _print_check("Label values (exact match)", labels_values_ok)
    assert labels_values_ok

    print(f"Computed input_lengths:  {input_lengths}")
    print(f"Expected input_lengths:  {example['input_lengths']}")
    input_lengths_ok = input_lengths == example["input_lengths"]
    _print_check("input_lengths", input_lengths_ok)
    assert input_lengths_ok

    print(f"Computed label_lengths:  {label_lengths}")
    print(f"Expected label_lengths:  {example['label_lengths']}")
    label_lengths_ok = label_lengths == example["label_lengths"]
    _print_check("label_lengths", label_lengths_ok)
    assert label_lengths_ok


def test_greedy_decoder_ctc_behavior():
    _print_section("Test 3: greedyDecoder CTC Behavior")
    print("Creating synthetic output tensor with clear argmax targets")

    # output: (batch=2, time=8, classes=29)
    output = torch.full((2, 8, 29), -10.0)

    # Sample 1: [a, a, blank, a, b, b, blank, space] -> "aab "
    seq1 = [2, 2, 28, 2, 3, 3, 28, 1]
    for t, cls in enumerate(seq1):
        output[0, t, cls] = 10.0

    # Sample 2: [blank, blank, d, d, d, blank, d, '] -> "dd'"
    seq2 = [28, 28, 5, 5, 5, 28, 5, 0]
    for t, cls in enumerate(seq2):
        output[1, t, cls] = 10.0

    print(f"Sample 1 argmax class sequence: {seq1}")
    print(f"Sample 2 argmax class sequence: {seq2}")
    print("Running greedyDecoder(output, blank_label=28)")
    decoded = greedyDecoder(output, blank_label=28)

    expected = ["aab ", "dd'"]
    print(f"Decoded output:  {decoded}")
    print(f"Expected output: {expected}")
    decoder_ok = decoded == expected
    _print_check("CTC greedy decode result", decoder_ok)
    assert decoder_ok


def run_all_tests_verbose():
    _print_section("Starting Verbose Lab 4 Proto Tests")
    test_text_mapping_roundtrip()
    test_data_processing_against_example()
    test_greedy_decoder_ctc_behavior()
    _print_section("All tests completed successfully")


if __name__ == "__main__":
    run_all_tests_verbose()