# DT2119, Lab 1 Feature Extraction - Testing and Visualization
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import windows
from lab1_proto import enframe, preemp, windowing, powerSpectrum, logMelSpectrum, cepstrum, mfcc
from lab1_tools import trfbank, lifter


def prepare_plot_dir(folder_name):
    plot_dir = Path(__file__).resolve().parent / folder_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    for old_png in plot_dir.glob("*.png"):
        old_png.unlink()
    return plot_dir


def save_and_show(plot_dir, filename):
    plt.savefig(plot_dir / filename, dpi=150)
    plt.show()

def plot_frames_mesh(frames, plot_dir, filename, title="Framed speech"):
    """Plot framed speech samples as a time-frame mesh."""
    plt.figure()
    plt.pcolormesh(frames)
    plt.title(title)
    plt.xlabel("Sample index within frame")
    plt.ylabel("Frame index")
    plt.colorbar(label="Amplitude")
    plt.tight_layout()
    save_and_show(plot_dir, filename)


plot_dir = prepare_plot_dir("proto_plots")

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
plot_frames_mesh(my_frames, plot_dir, "01_enframe_output.png", "enframe output")

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
save_and_show(plot_dir, "02_hamming_window_shape.png")

# Plot windowed frames for comparison
plot_frames_mesh(windowed_frames, plot_dir, "03_windowed_frames_output.png", "Windowed frames output")

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
save_and_show(plot_dir, "04_mel_triangular_filterbank.png")

print("\nPlotting Mel filterbank log-spectrum outputs...")
plt.figure()
plt.pcolormesh(mspec_frames)
plt.title("Mel Filterbank Log-Spectrum (mspec)")
plt.xlabel("Mel filter index")
plt.ylabel("Frame index")
plt.colorbar(label="Log energy")
plt.tight_layout()
save_and_show(plot_dir, "05_mel_filterbank_log_spectrum.png")

# Test logMelSpectrum function (continuation)
print("\n" + "=" * 50)
print("Testing power spectrum, cepstrum, and liftered MFCC")
print("=" * 50)

print("\nPlotting power spectrum...")
plt.figure()
plt.pcolormesh(spec_frames)
plt.title("Power Spectrum (spec)")
plt.xlabel("FFT bin / frequency-bin index")
plt.ylabel("Frame index")
plt.colorbar(label="Power")
plt.tight_layout()
save_and_show(plot_dir, "06_power_spectrum.png")

# Test cepstrum and generate plot
my_ceps = cepstrum(mspec_frames, 13)
print("\nPlotting cepstrum / MFCC coefficients...")
plt.figure()
plt.pcolormesh(my_ceps)
plt.title("MFCC / Cepstrum Coefficients (mfcc)")
plt.xlabel("Coefficient index")
plt.ylabel("Frame index")
plt.colorbar(label="Coefficient value")
plt.tight_layout()
save_and_show(plot_dir, "07_mfcc_cepstrum_coefficients.png")

#"********************************************************"

print("=== TEST PREEMP ===")
my_preemph = preemp(example["frames"], 0.97)
print("shape:", my_preemph.shape)
print("correct shape:", example["preemph"].shape)
print("allclose:", np.allclose(my_preemph, example["preemph"]))
print("max diff:", np.max(np.abs(my_preemph - example["preemph"])))




print("=== TEST POWERSPECTRUM ===")
my_spec = powerSpectrum(example["windowed"], 512)
print("shape:", my_spec.shape)
print("correct shape:", example["spec"].shape)
print("allclose:", np.allclose(my_spec, example["spec"]))
print("max diff:", np.max(np.abs(my_spec - example["spec"])))




print("=== TEST CEPSTRUM ===")
my_mfcc = cepstrum(example["mspec"], 13)
print("shape:", my_mfcc.shape)
print("correct shape:", example["mfcc"].shape)
print("allclose:", np.allclose(my_mfcc, example["mfcc"]))
print("max diff:", np.max(np.abs(my_mfcc - example["mfcc"])))


print("\n=== TEST FULL MFCC PIPELINE (LIFTERED) ===")
my_lmfcc = mfcc(example["samples"], winlen=400, winshift=200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=sr, liftercoeff=22)
print("shape:", my_lmfcc.shape)
print("correct shape:", example["lmfcc"].shape)
print("allclose:", np.allclose(my_lmfcc, example["lmfcc"]))
print("max diff:", np.max(np.abs(my_lmfcc - example["lmfcc"])))

print("\nPlotting liftered MFCC...")
plt.figure()
plt.pcolormesh(my_lmfcc)
plt.title("Liftered MFCC (lmfcc)")
plt.xlabel("Coefficient index")
plt.ylabel("Frame index")
plt.colorbar(label="Coefficient value")
plt.tight_layout()
save_and_show(plot_dir, "08_liftered_mfcc.png")

print(f"\nSaved plots to: {plot_dir}")