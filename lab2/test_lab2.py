import numpy as np
import os
from lab2_proto import *
from lab2_tools import *
from prondict import prondict
import matplotlib.pyplot as plt

def plot_example_results(example, obsloglik=None, log_alpha=None, log_beta=None, log_gamma=None, viterbi_path=None, output_dir='plots', show_plots=False):
    """
    Plot each result and save all figures in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

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

    for idx, (title, mat, ylabel) in enumerate(plots):
        plt.figure(figsize=(10, 3))
        plt.imshow(mat.T, aspect='auto', origin='lower')
        plt.title(title, fontsize=10)
        plt.ylabel(ylabel)
        plt.xlabel("time frame")

        plt.tight_layout()
        safe_name = title.lower().replace(':', '').replace(' ', '_')
        plt.savefig(os.path.join(output_dir, f"{idx:02d}_{safe_name}.png"), dpi=150)
        if show_plots:
            plt.show()
        else:
            plt.close()

        # Save an additional alpha plot with Viterbi overlay while preserving
        # the original alpha-only image.
        if ("logalpha" in title) and (viterbi_path is not None):
            plt.figure(figsize=(10, 3))
            plt.imshow(mat.T, aspect='auto', origin='lower')
            plt.plot(viterbi_path, color='red', linewidth=1.5)
            plt.title(f"{title} + Viterbi path", fontsize=10)
            plt.ylabel(ylabel)
            plt.xlabel("time frame")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{idx:02d}_{safe_name}_with_viterbi_overlay.png"), dpi=150)
            if show_plots:
                plt.show()
            else:
                plt.close()
        
        
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

# ===== TEST forward =====
print("===== TEST forward =====")

log_alpha = forward(
    example['obsloglik'],
    np.log(wordHMMs['o']['startprob'][:-1]),
    np.log(wordHMMs['o']['transmat'][:-1, :-1])
)

log_likelihood = logsumexp(log_alpha[-1, :])

print("forward match:", np.allclose(log_alpha, example['logalpha']))
print("forward loglik match:", np.allclose(log_likelihood, example['loglik']))
print("forward loglik:", log_likelihood)
print("example loglik:", example['loglik'])
print()

# ===== TEST viterbi =====
print("===== TEST viterbi =====")

viterbi_loglik, viterbi_path = viterbi(
    example['obsloglik'],
    np.log(wordHMMs['o']['startprob'][:-1]),
    np.log(wordHMMs['o']['transmat'][:-1, :-1]),
    forceFinalState=True
)

print("viterbi loglik match:", np.allclose(viterbi_loglik, example['vloglik']))
print("viterbi loglik:", viterbi_loglik)
print("example vloglik:", example['vloglik'])
print("viterbi path length match:", viterbi_path.shape[0] == example['obsloglik'].shape[0])
print("viterbi path indices valid:", np.all((viterbi_path >= 0) & (viterbi_path < example['obsloglik'].shape[1])))
print()

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

# ===== TEST backward =====
log_startprob_o = np.log(wordHMMs['o']['startprob'][:-1])
log_transmat_o = np.log(wordHMMs['o']['transmat'][:-1,:-1])
log_beta = backward(
    example['obsloglik'],
    log_startprob_o,
    log_transmat_o
)

print("backward match:", np.allclose(log_beta, example['logbeta']))
backward_loglik = logsumexp(log_startprob_o + example['obsloglik'][0, :] + log_beta[0, :])
print("backward loglik match:", np.allclose(backward_loglik, example['loglik']))
print("backward loglik:", backward_loglik)
print("example loglik:", example['loglik'])
print()

# ===== TEST gamma =====
log_gamma = statePosteriors(example['logalpha'], log_beta)

print("gamma match:", np.allclose(log_gamma, example['loggamma']))

gamma = np.exp(log_gamma)
print("gamma rows sum to 1:", np.allclose(np.sum(gamma, axis=1), 1.0))


# ===== PLOT CURRENT RESULTS =====
plot_example_results(example,
                     obsloglik=obsloglik, 
                     log_alpha=log_alpha,
                     log_beta=log_beta,
                     log_gamma=log_gamma,
                     viterbi_path=viterbi_path
                     )


def plot_em_loglik(loglik_trace, title, filename, output_dir='plots', show_plots=False):
    """Plot EM likelihood progression across iterations."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 3))
    plt.plot(np.arange(len(loglik_trace)), loglik_trace, marker='o', linewidth=1.5)
    plt.title(title)
    plt.xlabel('iteration')
    plt.ylabel('log P(X|theta)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    if show_plots:
        plt.show()
    else:
        plt.close()


print()
print("===== TEST updateMeanAndVar (example sanity) =====")
test_means, test_covars = updateMeanAndVar(example['lmfcc'], example['loggamma'])
print("means shape correct:", test_means.shape == wordHMMs['o']['means'].shape)
print("covars shape correct:", test_covars.shape == wordHMMs['o']['covars'].shape)
print("covars >= 5.0:", np.all(test_covars >= 5.0))


print()
print("===== TEST Baum-Welch emission retraining (data[10], wordHMMs['4']) =====")

utterance = data[10]
X = utterance['lmfcc']

base_hmm = wordHMMs['4']
log_startprob = np.log(base_hmm['startprob'][:-1])
log_transmat = np.log(base_hmm['transmat'][:-1, :-1])

means = base_hmm['means'].copy()
covars = base_hmm['covars'].copy()

max_iter = 20
improvement_threshold = 1.0
loglik_trace = []

for _ in range(max_iter):
    obsloglik_iter = log_multivariate_normal_density_diag(X, means, covars)
    log_alpha_iter = forward(obsloglik_iter, log_startprob, log_transmat)
    log_beta_iter = backward(obsloglik_iter, log_startprob, log_transmat)
    log_gamma_iter = statePosteriors(log_alpha_iter, log_beta_iter)

    loglik_prev = logsumexp(log_alpha_iter[-1, :])
    loglik_trace.append(loglik_prev)

    new_means, new_covars = updateMeanAndVar(X, log_gamma_iter, varianceFloor=5.0)

    obsloglik_new = log_multivariate_normal_density_diag(X, new_means, new_covars)
    log_alpha_new = forward(obsloglik_new, log_startprob, log_transmat)
    loglik_new = logsumexp(log_alpha_new[-1, :])

    if (loglik_new - loglik_prev) < improvement_threshold:
        break

    means, covars = new_means, new_covars

# Final score after last accepted model update.
obsloglik_final = log_multivariate_normal_density_diag(X, means, covars)
log_alpha_final = forward(obsloglik_final, log_startprob, log_transmat)
final_loglik = logsumexp(log_alpha_final[-1, :])
if len(loglik_trace) == 0 or not np.isclose(loglik_trace[-1], final_loglik):
    loglik_trace.append(final_loglik)

print("initial loglik:", loglik_trace[0])
print("final loglik:", loglik_trace[-1])
print("iterations used:", len(loglik_trace))
print("non-decreasing trace:", np.all(np.diff(loglik_trace) >= -1e-8))

plot_em_loglik(
    loglik_trace,
    title="EM retraining log-likelihood (data[10], model '4')",
    filename="em_retraining_data10_model4.png"
)


print()
print("===== TEST retraining from multiple start models =====")
start_digits = ['4', '3', '9']
summary = []

for start_digit in start_digits:
    hmm = wordHMMs[start_digit]
    log_startprob_s = np.log(hmm['startprob'][:-1])
    log_transmat_s = np.log(hmm['transmat'][:-1, :-1])
    means_s = hmm['means'].copy()
    covars_s = hmm['covars'].copy()
    trace_s = []

    for _ in range(max_iter):
        obsloglik_s = log_multivariate_normal_density_diag(X, means_s, covars_s)
        log_alpha_s = forward(obsloglik_s, log_startprob_s, log_transmat_s)
        log_beta_s = backward(obsloglik_s, log_startprob_s, log_transmat_s)
        log_gamma_s = statePosteriors(log_alpha_s, log_beta_s)

        ll_prev_s = logsumexp(log_alpha_s[-1, :])
        trace_s.append(ll_prev_s)

        new_means_s, new_covars_s = updateMeanAndVar(X, log_gamma_s, varianceFloor=5.0)
        obsloglik_new_s = log_multivariate_normal_density_diag(X, new_means_s, new_covars_s)
        log_alpha_new_s = forward(obsloglik_new_s, log_startprob_s, log_transmat_s)
        ll_new_s = logsumexp(log_alpha_new_s[-1, :])

        if (ll_new_s - ll_prev_s) < improvement_threshold:
            break

        means_s, covars_s = new_means_s, new_covars_s

    obsloglik_end_s = log_multivariate_normal_density_diag(X, means_s, covars_s)
    log_alpha_end_s = forward(obsloglik_end_s, log_startprob_s, log_transmat_s)
    ll_end_s = logsumexp(log_alpha_end_s[-1, :])
    if len(trace_s) == 0 or not np.isclose(trace_s[-1], ll_end_s):
        trace_s.append(ll_end_s)

    summary.append((start_digit, trace_s[0], trace_s[-1], len(trace_s)))

for start_digit, ll0, llf, n_iter in summary:
    print(f"start model '{start_digit}': initial={ll0:.3f}, final={llf:.3f}, iterations={n_iter}")


print()
print("===== END-TO-END PIPELINE TEST =====")


def evaluate_end_to_end_pipeline(data_items, phone_hmms, isolated_map):
    """Run full isolated-word recognition pipeline with Forward and Viterbi scoring."""
    # Build word HMMs from phone HMMs using pronunciation dictionary (+sil at both ends).
    word_hmms = {digit: concatHMMs(phone_hmms, isolated_map[digit]) for digit in isolated_map.keys()}
    candidate_digits = list(word_hmms.keys())

    forward_preds = []
    viterbi_preds = []
    true_digits = []
    forward_scores_all = []
    viterbi_scores_all = []

    for utt in data_items:
        X_utt = utt['lmfcc']
        true_digit = utt['digit']
        true_digits.append(true_digit)

        f_scores = []
        v_scores = []
        for digit in candidate_digits:
            hmm = word_hmms[digit]
            obsloglik_utt = log_multivariate_normal_density_diag(X_utt, hmm['means'], hmm['covars'])
            log_start = np.log(hmm['startprob'][:-1])
            log_trans = np.log(hmm['transmat'][:-1, :-1])

            # Forward score: log P(X|theta)
            alpha = forward(obsloglik_utt, log_start, log_trans)
            ll_forward = logsumexp(alpha[-1, :])

            # Viterbi score: log P(X, S_best|theta)
            ll_viterbi, _ = viterbi(obsloglik_utt, log_start, log_trans, forceFinalState=True)

            f_scores.append(ll_forward)
            v_scores.append(ll_viterbi)

        forward_scores_all.append(f_scores)
        viterbi_scores_all.append(v_scores)
        forward_preds.append(candidate_digits[int(np.argmax(f_scores))])
        viterbi_preds.append(candidate_digits[int(np.argmax(v_scores))])

    true_digits = np.array(true_digits)
    forward_preds = np.array(forward_preds)
    viterbi_preds = np.array(viterbi_preds)
    forward_scores_all = np.array(forward_scores_all)
    viterbi_scores_all = np.array(viterbi_scores_all)

    forward_correct = int(np.sum(forward_preds == true_digits))
    viterbi_correct = int(np.sum(viterbi_preds == true_digits))
    n_total = len(true_digits)

    forward_acc = forward_correct / n_total
    viterbi_acc = viterbi_correct / n_total

    forward_mismatches = [
        (idx, true_digits[idx], forward_preds[idx])
        for idx in range(n_total)
        if forward_preds[idx] != true_digits[idx]
    ]
    viterbi_mismatches = [
        (idx, true_digits[idx], viterbi_preds[idx])
        for idx in range(n_total)
        if viterbi_preds[idx] != true_digits[idx]
    ]

    return {
        'n_total': n_total,
        'candidate_digits': candidate_digits,
        'forward_accuracy': forward_acc,
        'viterbi_accuracy': viterbi_acc,
        'forward_correct': forward_correct,
        'viterbi_correct': viterbi_correct,
        'forward_mismatches': forward_mismatches,
        'viterbi_mismatches': viterbi_mismatches,
        'forward_scores_all': forward_scores_all,
        'viterbi_scores_all': viterbi_scores_all,
    }


pipeline_result_onespkr = evaluate_end_to_end_pipeline(data, phoneHMMs, isolated)

print("onespkr models - utterances:", pipeline_result_onespkr['n_total'])
print(
    "onespkr forward accuracy: "
    f"{pipeline_result_onespkr['forward_accuracy']:.3f} "
    f"({pipeline_result_onespkr['forward_correct']}/{pipeline_result_onespkr['n_total']})"
)
print(
    "onespkr viterbi accuracy: "
    f"{pipeline_result_onespkr['viterbi_accuracy']:.3f} "
    f"({pipeline_result_onespkr['viterbi_correct']}/{pipeline_result_onespkr['n_total']})"
)
print("onespkr forward mistakes:", len(pipeline_result_onespkr['forward_mismatches']))
print("onespkr viterbi mistakes:", len(pipeline_result_onespkr['viterbi_mismatches']))
print(
    "forward scores finite:",
    np.all(np.isfinite(pipeline_result_onespkr['forward_scores_all']))
)
print(
    "viterbi scores finite:",
    np.all(np.isfinite(pipeline_result_onespkr['viterbi_scores_all']))
)

# Keep mismatch output short and readable.
print("sample forward mismatches (up to 10):", pipeline_result_onespkr['forward_mismatches'][:10])
print("sample viterbi mismatches (up to 10):", pipeline_result_onespkr['viterbi_mismatches'][:10])


# Optional comparison requested by instructions: models trained on all speakers.
phoneHMMs_all = np.load('lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()
pipeline_result_allspk = evaluate_end_to_end_pipeline(data, phoneHMMs_all, isolated)

print()
print(
    "all-speakers forward accuracy: "
    f"{pipeline_result_allspk['forward_accuracy']:.3f} "
    f"({pipeline_result_allspk['forward_correct']}/{pipeline_result_allspk['n_total']})"
)
print(
    "all-speakers viterbi accuracy: "
    f"{pipeline_result_allspk['viterbi_accuracy']:.3f} "
    f"({pipeline_result_allspk['viterbi_correct']}/{pipeline_result_allspk['n_total']})"
)
print("all-speakers forward mistakes:", len(pipeline_result_allspk['forward_mismatches']))
print("all-speakers viterbi mistakes:", len(pipeline_result_allspk['viterbi_mismatches']))


# Sanity checks for full pipeline integrity.
assert pipeline_result_onespkr['n_total'] == len(data)
assert pipeline_result_allspk['n_total'] == len(data)
assert pipeline_result_onespkr['forward_scores_all'].shape == (len(data), len(isolated))
assert pipeline_result_onespkr['viterbi_scores_all'].shape == (len(data), len(isolated))