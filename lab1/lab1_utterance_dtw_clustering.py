# DT2119, Lab 1 - Section 7: Comparing Utterances with DTW and Hierarchical Clustering
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import euclidean

from lab1_proto import mfcc, dtw
from lab1_tools import tidigit2labels


def load_data(path="lab1_data.npz"):
    """Load the TIDIGITS data array."""
    return np.load(path, allow_pickle=True)["data"]


def prepare_plot_dir(folder_name):
    plot_dir = Path(__file__).resolve().parent / folder_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    for old_png in plot_dir.glob("*.png"):
        old_png.unlink()
    return plot_dir


def extract_mfcc_features(data, nceps=13):
    """
    Compute MFCC features for each utterance in the dataset.
    
    Args:
        data: array of utterance dictionaries
        nceps: number of cepstral coefficients
        
    Returns:
        mfcc_per_utt: list of MFCC arrays, one per utterance
    """
    mfcc_per_utt = []
    
    for utt in data:
        samples = utt["samples"]
        sr = int(utt["samplingrate"])
        
        utt_mfcc = mfcc(
            samples,
            winlen=400,
            winshift=200,
            preempcoeff=0.97,
            nfft=512,
            nceps=nceps,
            samplingrate=sr,
            liftercoeff=22,
        )
        mfcc_per_utt.append(utt_mfcc)
    
    return mfcc_per_utt


def compute_pairwise_dtw_distances(mfcc_per_utt):
    """
    Compute pairwise DTW distances for all utterance pairs.
    
    For each pair (i, j) where i < j:
    1. Compute local Euclidean distances between MFCC vectors
    2. Apply DTW to get normalized global distance
    3. Store in upper triangle of symmetric matrix
    
    Args:
        mfcc_per_utt: list of MFCC feature matrices (N_utterances,)
        
    Returns:
        D: 44×44 symmetric distance matrix
    """
    n_utt = len(mfcc_per_utt)
    D = np.zeros((n_utt, n_utt), dtype=float)
    
    print(f"Computing pairwise DTW distances for {n_utt} utterances...")
    for i in range(n_utt):
        for j in range(i + 1, n_utt):
            # Compute DTW distance between utterances i and j
            # dtw expects a distance function
            dist_val, _, _, _ = dtw(mfcc_per_utt[i], mfcc_per_utt[j], euclidean)
            D[i, j] = dist_val
            D[j, i] = dist_val
            
            if (i * n_utt + j) % 100 == 0:
                print(f"  computed {i * n_utt + j + 1} / {n_utt * (n_utt - 1) // 2} pairs")
    
    print(f"Complete. Distance matrix shape: {D.shape}")
    return D


def plot_distance_matrix(D, data, save_path=None):
    """
    Plot the pairwise distance matrix with pcolormesh.
    
    Args:
        D: 44×44 distance matrix
        data: utterance array (unused but kept for context)
    """
    plt.figure(figsize=(10, 9))
    plt.pcolormesh(D, shading="auto", cmap="viridis")
    plt.title("Pairwise DTW Distance Matrix (44×44 utterances)")
    plt.xlabel("Utterance index")
    plt.ylabel("Utterance index")
    plt.colorbar(label="DTW distance (normalized)")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)


def analyze_distance_structure(D, data):
    """
    Print summary statistics comparing within-digit and across-digit distances.
    
    Args:
        D: distance matrix
        data: utterance array
    """
    # Group utterances by digit
    digit_to_indices = {}
    for i, utt in enumerate(data):
        digit = str(utt["digit"])
        if digit not in digit_to_indices:
            digit_to_indices[digit] = []
        digit_to_indices[digit].append(i)
    
    print("\nDistance analysis:")
    print(f"  Total unique digits: {len(digit_to_indices)}")
    
    # Compute within-digit distances (same digit, different utterances)
    within_digit_dists = []
    for digit, indices in digit_to_indices.items():
        if len(indices) > 1:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    within_digit_dists.append(D[indices[i], indices[j]])
    
    # Compute across-digit distances (different digits)
    across_digit_dists = []
    digits_list = list(digit_to_indices.keys())
    for i_d, d1 in enumerate(digits_list):
        for j_d in range(i_d + 1, len(digits_list)):
            d2 = digits_list[j_d]
            for idx1 in digit_to_indices[d1]:
                for idx2 in digit_to_indices[d2]:
                    across_digit_dists.append(D[idx1, idx2])
    
    within_digit_dists = np.array(within_digit_dists)
    across_digit_dists = np.array(across_digit_dists)
    
    print(f"\n  Within-digit distances (same word, different utterances):")
    print(f"    count: {len(within_digit_dists)}")
    print(f"    mean:  {float(np.mean(within_digit_dists)):.6f}")
    print(f"    std:   {float(np.std(within_digit_dists)):.6f}")
    print(f"    min:   {float(np.min(within_digit_dists)):.6f}")
    print(f"    max:   {float(np.max(within_digit_dists)):.6f}")
    
    print(f"\n  Across-digit distances (different words):")
    print(f"    count: {len(across_digit_dists)}")
    print(f"    mean:  {float(np.mean(across_digit_dists)):.6f}")
    print(f"    std:   {float(np.std(across_digit_dists)):.6f}")
    print(f"    min:   {float(np.min(across_digit_dists)):.6f}")
    print(f"    max:   {float(np.max(across_digit_dists)):.6f}")
    
    separation_ratio = float(np.mean(across_digit_dists)) / float(np.mean(within_digit_dists))
    print(f"\n  Separation ratio (across / within): {separation_ratio:.4f}")
    if separation_ratio > 1.5:
        print("  interpretation: strong separation; digits are well-separated even between speakers.")
    elif separation_ratio > 1.2:
        print("  interpretation: moderate separation; digit classes are distinguishable but with overlap.")
    else:
        print("  interpretation: weak separation; significant overlap between digit classes.")


def hierarchical_clustering_dendrogram(D, data, save_path=None):
    """
    Run hierarchical clustering and plot dendrogram.
    
    Args:
        D: distance matrix (symmetric, 44×44)
        data: utterance array (for labels)
    """
    # Convert symmetric distance matrix to condensed form for scipy.cluster.hierarchy
    n = D.shape[0]
    condensed = D[np.triu_indices(n, k=1)]
    
    # Run hierarchical clustering with complete linkage
    Z = linkage(condensed, method="complete")
    
    # Get labels using tidigit2labels
    labels = tidigit2labels(data)
    
    # Plot dendrogram
    plt.figure(figsize=(20, 8))
    dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=8)
    plt.title("Hierarchical Clustering Dendrogram (Complete Linkage)\nDTW distances, labeled by gender_speaker_digit_repetition")
    plt.xlabel("Utterance")
    plt.ylabel("DTW distance (normalized)")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    
    # Print interpretation guide
    print("\n=== Dendrogram interpretation ===")
    print("Structure to observe:")
    print("  1. Do clusters group utterances of the same digit together?")
    print("  2. Are different speakers (ac vs aw) separated or mixed within digit clusters?")
    print("  3. Are repetitions (a vs b) grouped or separated?")
    print("  4. Do acoustically similar digits (e.g., 'one'/'seven', 'two'/'to') cluster nearby?")
    print("\nCommon patterns in speech data:")
    print("  - Within-digit clusters: utterances of same word tend to group (if well-separated)")
    print("  - Cross-speaker variability: speaker differences can be as large as digit differences")
    print("  - Phonetic similarity: digit pairs with similar phones may be nearby")


def main(show_plots=True):
    plot_dir = prepare_plot_dir("dtw_plots")

    data = load_data("lab1_data.npz")
    print(f"Loaded {len(data)} utterances from lab1_data.npz")
    
    # Extract MFCC features for all utterances
    mfcc_per_utt = extract_mfcc_features(data, nceps=13)
    print(f"Extracted MFCC features for all utterances\n")
    
    # Compute pairwise DTW distance matrix
    D = compute_pairwise_dtw_distances(mfcc_per_utt)
    
    # Plot distance matrix
    plot_distance_matrix(D, data, save_path=plot_dir / "pairwise_dtw_distance_matrix.png")
    
    # Analyze distance structure
    analyze_distance_structure(D, data)
    
    # Hierarchical clustering and dendrogram
    hierarchical_clustering_dendrogram(D, data, save_path=plot_dir / "hierarchical_dendrogram.png")

    print(f"Saved plots to: {plot_dir}")
    
    if show_plots:
        plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DT2119 Lab 1 - Section 7: DTW and hierarchical clustering")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Run computations without opening plot windows.",
    )
    args = parser.parse_args()
    
    main(show_plots=not args.no_show)
