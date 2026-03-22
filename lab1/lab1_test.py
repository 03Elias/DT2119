# DT2119, Lab 1 Feature Extraction - Testing and Visualization
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
from lab1_proto import enframe, preemp, windowing

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
