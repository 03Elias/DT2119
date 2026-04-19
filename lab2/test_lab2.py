import numpy as np
from lab2_proto import *
from lab2_tools import *
from prondict import prondict
import matplotlib.pyplot as plt

def plot_example_results(example, obsloglik=None, log_alpha=None, log_beta=None, log_gamma=None, viterbi_path=None):
    """
    Plot each result in a separate figure window.
    """

    plots = [
        ("lmfcc: Liftered MFCCs", example['lmfcc'], "coeff")
    ]

    if obsloglik is not None:
        plots.append(("obsloglik: HMM log likelihood of observation given the state", obsloglik, "state"))

    if log_alpha is not None:
        plots.append(("logalpha: forward log probabilities", log_alpha, "state"))

    if log_beta is not None:
        plots.append(("logbeta: backward log probabilities", log_beta, "state"))

    if log_gamma is not None:
        plots.append(("loggamma: state log posteriors", log_gamma, "state"))

    for title, mat, ylabel in plots:
        plt.figure(figsize=(10, 3))
        plt.imshow(mat.T, aspect='auto', origin='lower')
        plt.title(title, fontsize=10)
        plt.ylabel(ylabel)
        plt.xlabel("time frame")

        if ("loggamma" in title) and (viterbi_path is not None):
            plt.plot(viterbi_path, color='red', linewidth=1.5)

        plt.tight_layout()
        plt.show()
        
        
# load data
data = np.load('lab2_data.npz', allow_pickle=True)['data']
example = np.load('lab2_example.npz', allow_pickle=True)['example'].item()

phoneHMMs = np.load('lab2_models_onespkr.npz', allow_pickle=True)['phoneHMMs'].item()

# build isolated
isolated = {}
for digit in prondict.keys():
    isolated[digit] = ['sil'] + prondict[digit] + ['sil']

# build wordHMMs
wordHMMs = {}
for digit in isolated:
    wordHMMs[digit] = concatHMMs(phoneHMMs, isolated[digit])

print("✔ Models built")

# ===== TEST obsloglik =====
obsloglik = log_multivariate_normal_density_diag(
    example['lmfcc'],
    wordHMMs['o']['means'],
    wordHMMs['o']['covars']
)

print("obsloglik match:", np.allclose(obsloglik, example['obsloglik']))

# ===== TEST gmmloglik =====
print("===== TEST gmmloglik =====")

log_emlik = np.log(np.array([
    [0.2, 0.8],
    [0.5, 0.5]
]))

weights = np.array([0.6, 0.4])

result = gmmloglik(log_emlik, weights)

print("gmmloglik =", result)

manual_1 = np.log(0.6 * 0.2 + 0.4 * 0.8)
manual_2 = np.log(0.6 * 0.5 + 0.4 * 0.5)
manual_total = manual_1 + manual_2

print("manual =", manual_total)
print("close?", np.isclose(result, manual_total))
print()

# ===== TEST gmmloglik with example =====
print("===== TEST gmmloglik with example =====")

log_emlik = example['obsloglik']
M = log_emlik.shape[1]
weights = np.ones(M) / M

result = gmmloglik(log_emlik, weights)

print("gmmloglik (example) =", result)
print("shape log_emlik:", log_emlik.shape)
print("weights:", weights)
print()

# ===== PLOT CURRENT RESULTS =====
plot_example_results(example, obsloglik=obsloglik)