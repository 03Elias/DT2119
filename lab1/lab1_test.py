# DT2119, Lab 1 Feature Extraction - Testing and Visualization
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from lab1_proto import enframe, preemp, windowing, powerSpectrum, logMelSpectrum, mfcc, dtw
from lab1_tools import trfbank, tidigit2labels

def plot_frames_mesh(frames, title="Framed speech"):
    """Plot framed speech samples as a time-frame mesh."""
    plt.figure()
    plt.pcolormesh(frames)
    plt.title(title)
    plt.xlabel("Sample index within frame")
    plt.ylabel("Frame index")
    plt.colorbar(label="Amplitude")
    plt.tight_layout()
    plt.show()

# Load reference data from file
example = np.load('lab1_example.npz', allow_pickle=True)['example'].item()

# Test enframe function
print("=" * 50)
print("Testing enframe function")
print("=" * 50)
sr = example['samplingrate']
winlen = int(0.02 * sr)     
winshift = int(0.01 * sr)   
my_frames = enframe(example['samples'], winlen, winshift)
print("ref shape:", example['frames'].shape)
print("enframe shape:", my_frames.shape)
print("exact match:", np.array_equal(my_frames, example['frames']))
print("max abs diff:", np.max(np.abs(my_frames - example['frames'])))
print("first 10 ref:", example['frames'][0, :10])
print("first 10 enframe:", my_frames[0, :10])
plot_frames_mesh(my_frames, "enframe output")

# Test windowing function
print("\n" + "=" * 50)
print("Testing windowing function")
print("=" * 50)
preemph_frames = preemp(my_frames, 0.97)
windowed_frames = windowing(preemph_frames)
print("ref windowed shape:", example['windowed'].shape)
print("windowed shape:", windowed_frames.shape)
print("exact match:", np.array_equal(windowed_frames, example['windowed']))
print("max abs diff:", np.max(np.abs(windowed_frames - example['windowed'])))

# Plot the Hamming window shape
print("\nPlotting Hamming window shape...")
plt.figure(figsize=(10, 4))
win = windows.hamming(winlen, sym=False)
plt.plot(win, linewidth=2)
plt.title("Hamming Window Shape", fontsize=14)
plt.xlabel("Sample index within frame", fontsize=12)
plt.ylabel("Window amplitude", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot windowed frames for comparison
plot_frames_mesh(windowed_frames, "Windowed frames output")

# Test logMelSpectrum function
print("\n" + "=" * 50)
print("Testing logMelSpectrum function")
print("=" * 50)
spec_frames = powerSpectrum(windowed_frames, 512)
mspec_frames = logMelSpectrum(spec_frames, sr)
print("ref mspec shape:", example['mspec'].shape)
print("mspec shape:", mspec_frames.shape)
print("exact match:", np.array_equal(mspec_frames, example['mspec']))
print("max abs diff:", np.max(np.abs(mspec_frames - example['mspec'])))

print("\nPlotting Mel filterbank shapes in linear frequency scale...")
nfft = spec_frames.shape[1]
fbank = trfbank(sr, nfft)
freq_axis = np.linspace(0, sr / 2, nfft // 2 + 1)
plt.figure(figsize=(10, 4))
for filt in fbank:
    plt.plot(freq_axis, filt[: nfft // 2 + 1], linewidth=1)
plt.title("Mel Triangular Filterbank (Linear Frequency Scale)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Filter gain")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nPlotting Mel filterbank log-spectrum outputs...")
plt.figure()
plt.pcolormesh(mspec_frames)
plt.title("Mel Filterbank Log-Spectrum (mspec)")
plt.xlabel("Mel filter index")
plt.ylabel("Frame index")
plt.colorbar(label="Log energy")
plt.tight_layout()
plt.show()

# Comparing utterances with DTW
print("\n" + "=" * 50)
print("Comparing utterances with DTW")
print("=" * 50)

data = np.load('lab1_data.npz', allow_pickle=True)['data']
n_utts = len(data)
print(f"Loaded {n_utts} utterances")

print("Computing MFCCs for all utterances...")
mfcc_features = [
    mfcc(utt['samples'], samplingrate=utt['samplingrate'])
    for utt in data
]

print("Computing pairwise DTW distances (this may take a while)...")
D = np.zeros((n_utts, n_utts), dtype=float)

for i in range(n_utts):
    for j in range(i + 1, n_utts):
        d, _, _, _ = dtw(
            mfcc_features[i],
            mfcc_features[j],
            lambda a, b: np.linalg.norm(a - b),
        )
        D[i, j] = d
        D[j, i] = d  # symmetry

print("\nPlotting DTW global distance matrix D...")
plt.figure(figsize=(8, 6))
plt.pcolormesh(D, shading='auto')  # added shading='auto' for cleaner plotting
plt.title("Pairwise DTW Distance Matrix (44x44)")
plt.xlabel("Utterance index")
plt.ylabel("Utterance index")
plt.colorbar(label="Global DTW distance")
plt.tight_layout()
plt.show()

digits = np.array([utt['digit'] for utt in data])
speakers = np.array([utt['speaker'] for utt in data])

upper = np.triu(np.ones((n_utts, n_utts), dtype=bool), 1)
same_digit = digits[:, None] == digits[None, :]
same_speaker = speakers[:, None] == speakers[None, :]

within_digit = D[upper & same_digit]
across_digits = D[upper & (~same_digit)]
within_digit_across_speakers = D[upper & same_digit & (~same_speaker)]

print("\nDistance comparison summary")
print("mean(within same digit):", np.mean(within_digit))
print("mean(across different digits):", np.mean(across_digits))
print("mean(within same digit, across speakers):", np.mean(within_digit_across_speakers))

if np.median(within_digit) < np.median(across_digits):
    print("Observation: Distances tend to be smaller within the same digit than across digits.")
else:
    print("Observation: Separation between same-digit and different-digit distances is weak.")

print("\nRunning hierarchical clustering (complete linkage)...")
labels = tidigit2labels(data)
Z = linkage(squareform(D, checks=False), method='complete')

plt.figure(figsize=(14, 6))
dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=8)
plt.title("Hierarchical Clustering of Utterances (DTW, complete linkage)")
plt.xlabel("Utterance")
plt.ylabel("Cluster distance")
plt.tight_layout()
plt.show()