# DT2119, Lab 1 - Sections 5 and 6 analysis
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

from lab1_proto import mfcc, mspec


def load_data(path="lab1_data.npz"):
    return np.load(path, allow_pickle=True)["data"]


def extract_features(data, nceps=13):
    """Compute per-utterance MFCC and Mel filterbank features."""
    mfcc_per_utt = []
    mspec_per_utt = []

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
        utt_mspec = mspec(
            samples,
            winlen=400,
            winshift=200,
            preempcoeff=0.97,
            nfft=512,
            samplingrate=sr,
        )

        mfcc_per_utt.append(utt_mfcc)
        mspec_per_utt.append(utt_mspec)

    all_mfcc = np.vstack(mfcc_per_utt)
    all_mspec = np.vstack(mspec_per_utt)
    return mfcc_per_utt, mspec_per_utt, all_mfcc, all_mspec


def correlation_matrix(features):
    # Correlation between feature coefficients (columns), not between frames.
    return np.corrcoef(features, rowvar=False)


def prepare_plot_dir(folder_name):
    plot_dir = Path(__file__).resolve().parent / folder_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    for old_png in plot_dir.glob("*.png"):
        old_png.unlink()
    return plot_dir


def plot_correlation(corr, title, xlabel, ylabel, save_path=None):
    plt.figure(figsize=(7, 6))
    plt.pcolormesh(corr, shading="auto")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(label="Correlation")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)


def summarize_correlation(name, corr):
    off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
    mean_abs = float(np.mean(np.abs(off_diag)))
    max_abs = float(np.max(np.abs(off_diag)))

    print(f"\n{name} correlation summary")
    print(f"  matrix shape: {corr.shape}")
    print(f"  mean |off-diagonal correlation|: {mean_abs:.4f}")
    print(f"  max  |off-diagonal correlation|: {max_abs:.4f}")

    if mean_abs > 0.2:
        print("  interpretation: features are clearly correlated where diagonal covariance is a rough approximation.")
    elif mean_abs > 0.1:
        print("  interpretation: moderate correlation exists where diagonal covariance may lose information.")
    else:
        print("  interpretation: weak average correlation where diagonal covariance is more defensible.")


def fit_gmms(features, n_components_list, random_state=0):
    gmms = {}
    for k in n_components_list:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            random_state=random_state,
            reg_covar=1e-6,
            max_iter=300,
        )
        gmm.fit(features)
        gmms[k] = gmm
        print(f"Trained GMM with {k:2d} components | converged={gmm.converged_} | lower_bound={gmm.lower_bound_:.4f}")
    return gmms


def group_indices_by_digit(data):
    by_digit = defaultdict(list)
    for i, utt in enumerate(data):
        by_digit[str(utt["digit"])].append(i)
    return by_digit


def choose_seven_indices(data):
    # Required by instructions: utterances 16, 17, 38, 39.
    requested_zero_based = [16, 17, 38, 39]
    if max(requested_zero_based) < len(data) and all(str(data[i]["digit"]) == "7" for i in requested_zero_based):
        return requested_zero_based, "0-based"

    # Fallback if the instructions were 1-based.
    requested_one_based = [15, 16, 37, 38]
    if max(requested_one_based) < len(data) and all(str(data[i]["digit"]) == "7" for i in requested_one_based):
        return requested_one_based, "1-based converted to 0-based"

    # Last resort: take first four utterances with digit '7'.
    all_seven = [i for i, utt in enumerate(data) if str(utt["digit"]) == "7"]
    return all_seven[:4], "auto-detected first four '7' utterances"


def plot_posteriors_for_indices(gmm, mfcc_per_utt, data, indices, title_prefix, save_path=None):
    nrows = len(indices)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12, 2.6 * nrows), squeeze=False)

    for row, idx in enumerate(indices):
        post = gmm.predict_proba(mfcc_per_utt[idx])
        ax = axes[row, 0]
        ax.pcolormesh(post.T, shading="auto")
        utt = data[idx]
        ax.set_ylabel("GMM comp")
        ax.set_title(
            f"idx={idx} | digit={utt['digit']} | speaker={utt['speaker']} | gender={utt['gender']} | rep={utt['repetition']}"
        )

    axes[-1, 0].set_xlabel("Frame index")
    fig.suptitle(title_prefix)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)


def posterior_stability_note(gmm, mfcc_per_utt, data, indices):
    """Simple numeric cue: average frame-wise entropy of posterior distributions."""
    entropies = []
    for idx in indices:
        post = gmm.predict_proba(mfcc_per_utt[idx])
        post = np.clip(post, 1e-12, 1.0)
        frame_entropy = -np.sum(post * np.log(post), axis=1)
        entropies.append((idx, float(np.mean(frame_entropy))))

    print("\nPosterior concentration (lower entropy = crisper class assignment):")
    for idx, e in entropies:
        utt = data[idx]
        print(f"  idx={idx:2d} ({utt['speaker']}_{utt['digit']}_{utt['repetition']}): mean entropy={e:.4f}")

    mean_e = float(np.mean([e for _, e in entropies]))
    if mean_e < 1.5:
        print("  interpretation: relatively peaked posteriors; classes are fairly stable over time.")
    elif mean_e < 2.5:
        print("  interpretation: mixed certainty; some segments map to stable classes, others are diffuse.")
    else:
        print("  interpretation: diffuse posteriors; unsupervised classes are less stable/interpretable.")


def main(show_plots=True):
    plot_dir = prepare_plot_dir("feature_gmm_plots")

    data = load_data("lab1_data.npz")
    print(f"Loaded {len(data)} utterances from lab1_data.npz")

    mfcc_per_utt, mspec_per_utt, all_mfcc, all_mspec = extract_features(data, nceps=13)
    print(f"Concatenated MFCC shape : {all_mfcc.shape}")
    print(f"Concatenated MSPEC shape: {all_mspec.shape}")

    # Section 5: Feature correlation
    corr_mfcc = correlation_matrix(all_mfcc)
    corr_mspec = correlation_matrix(all_mspec)

    plot_correlation(
        corr_mfcc,
        "Feature Correlation - Liftered MFCC",
        "MFCC coefficient index",
        "MFCC coefficient index",
        save_path=plot_dir / "correlation_mfcc.png",
    )
    plot_correlation(
        corr_mspec,
        "Feature Correlation - Mel Filterbank (mspec)",
        "Mel filter index",
        "Mel filter index",
        save_path=plot_dir / "correlation_mspec.png",
    )

    summarize_correlation("MFCC", corr_mfcc)
    summarize_correlation("MSPEC", corr_mspec)

    # Section 6: GMM clustering and posteriors
    n_components_list = [4, 8, 16, 32]
    gmms = fit_gmms(all_mfcc, n_components_list=n_components_list, random_state=0)

    # Show posterior evolution for one word class ("seven") across utterances.
    seven_indices, mode = choose_seven_indices(data)
    print(f"\nUsing 'seven' utterance indices ({mode}): {seven_indices}")

    # Plot for all requested model sizes to observe effect of number of classes.
    for k in n_components_list:
        plot_posteriors_for_indices(
            gmms[k],
            mfcc_per_utt,
            data,
            seven_indices,
            title_prefix=f"GMM posteriors for word 'seven' (K={k})",
            save_path=plot_dir / f"seven_posteriors_k{k}.png",
        )

    # Specific required analysis for K=32 and utterances 16,17,38,39 (or mapped fallback).
    posterior_stability_note(gmms[32], mfcc_per_utt, data, seven_indices)

    print(f"Saved plots to: {plot_dir}")

   
    if show_plots:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DT2119 Lab 1 - Feature correlation and GMM clustering")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Run computations without opening plot windows.",
    )
    args = parser.parse_args()

    main(show_plots=not args.no_show)
