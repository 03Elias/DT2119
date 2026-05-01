import numpy as np
from lab2_tools import *

# already implemented
def concatTwoHMMs(hmm1, hmm2):
    """ Concatenates 2 HMM models

    Args:
       hmm1, hmm2: two dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be different for each)

    Output
       dictionary with the same keys as the input but concatenated models:
          startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models
   
    Example:
       twoHMMs = concatHMMs(phoneHMMs['sil'], phoneHMMs['ow'])

    See also: the concatenating_hmms.pdf document in the lab package
    """
    num_states_hmm1 = len(hmm1['startprob'])-1
    num_states_hmm2 = len(hmm2['startprob'])-1
    num_states_concat = num_states_hmm1+num_states_hmm2+1
    
    startprob = np.concatenate((hmm1['startprob'],hmm2['startprob'][1:]))

    transmat = np.zeros((num_states_concat,num_states_concat))
    transmat[:num_states_hmm1+1,:num_states_hmm1+1] = hmm1['transmat']
    transmat[num_states_hmm1:,num_states_hmm1:] = hmm2['transmat']

    means = np.concatenate((hmm1['means'], hmm2['means']), axis=0)

    covars = np.concatenate((hmm1['covars'], hmm2['covars']), axis=0)

    concatenated_hmm = {'startprob': startprob,
                       'transmat': transmat,
                       'means': means,
                       'covars': covars}
    
    return concatenated_hmm


# already implemented, uses concatTwoHMMs()
def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: dictionary of models indexed by model name. 
       hmmmodels[name] is a dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models:
         startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    concat = hmmmodels[namelist[0]]
    for idx in range(1,len(namelist)):
        concat = concatTwoHMMs(concat, hmmmodels[namelist[idx]])
    return concat

#Elmira
def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """
    log_weights = np.log(weights)
    weighted_log_emlik = log_emlik + log_weights
    return np.sum(logsumexp(weighted_log_emlik, axis=1))
    
#Elias
def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """
    N, M = log_emlik.shape
    forward_prob = np.full((N, M), -np.inf)

    # Keep only emitting-state priors/transitions if a non-emitting final state is present.
    startprob = log_startprob[:M]
    transmat = log_transmat[:M, :M]

    # Initialization: alpha_0(j) = pi_j * b_j(x_0)
    forward_prob[0, :] = startprob + log_emlik[0, :]

    # Recursion: alpha_n(j) = b_j(x_n) * sum_i alpha_{n-1}(i) * a_ij
    for n in range(1, N):
        for j in range(M):
            forward_prob[n, j] = logsumexp(forward_prob[n - 1, :] + transmat[:, j]) + log_emlik[n, j]

    return forward_prob
#Elmira
def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """
    n_frames, n_states = log_emlik.shape
    log_beta = np.zeros((n_frames, n_states))

    # Initialization:
    # log beta_{N-1}(i) = 0
    log_beta[-1, :] = 0.0

    # Recursion backward in time
    for n in range(n_frames - 2, -1, -1):
        for i in range(n_states):
            log_beta[n, i] = logsumexp(
                log_transmat[i, :] + log_emlik[n + 1, :] + log_beta[n + 1, :],
                axis=0
            )

    return log_beta
    
    
#Elias
def viterbi(log_emlik, log_startprob, log_transmat, forceFinalState=True):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j
        forceFinalState: if True, start backtracking from the final state in
                  the model, instead of the best state at the last time step

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """
    n_frames, n_states = log_emlik.shape

    # Keep only emitting-state priors/transitions if a non-emitting final state is present.
    startprob = log_startprob[:n_states]
    transmat = log_transmat[:n_states, :n_states]

    log_v = np.full((n_frames, n_states), -np.inf)
    backptr = np.zeros((n_frames, n_states), dtype=int)

    # Initialization
    log_v[0, :] = startprob + log_emlik[0, :]

    # Recursion
    for n in range(1, n_frames):
        for j in range(n_states):
            candidates = log_v[n - 1, :] + transmat[:, j]
            best_prev = np.argmax(candidates)
            backptr[n, j] = best_prev
            log_v[n, j] = candidates[best_prev] + log_emlik[n, j]

    # Termination
    if forceFinalState:
        last_state = n_states - 1
    else:
        last_state = np.argmax(log_v[-1, :])
    viterbi_loglik = log_v[-1, last_state]

    # Backtracking
    viterbi_path = np.zeros(n_frames, dtype=int)
    viterbi_path[-1] = last_state
    for n in range(n_frames - 1, 0, -1):
        viterbi_path[n - 1] = backptr[n, viterbi_path[n]]

    return viterbi_loglik, viterbi_path

#Elmira
def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """
    loglik = logsumexp(log_alpha[-1, :], axis=0)
    log_gamma = log_alpha + log_beta - loglik
    return log_gamma
    
    
#Elias
def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
    """ Update Gaussian parameters with diagonal covariance

    Args:
         X: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
         varianceFloor: minimum allowed variance scalar
    were N is the lenght of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state
    """
    # Convert posteriors from log domain to linear domain.
    gamma = np.exp(log_gamma)

    # Effective frame count per state.
    gamma_sum = np.sum(gamma, axis=0)

    # Avoid division by zero for states with negligible occupancy.
    gamma_sum_safe = np.maximum(gamma_sum, np.finfo(float).eps)

    # Weighted mean for each state.
    means = (gamma.T @ X) / gamma_sum_safe[:, np.newaxis]

    # Weighted diagonal variance for each state.
    centered = X[:, np.newaxis, :] - means[np.newaxis, :, :]
    weighted_sq = gamma[:, :, np.newaxis] * (centered ** 2)
    covars = np.sum(weighted_sq, axis=0) / gamma_sum_safe[:, np.newaxis]

    # Apply minimum variance to avoid degenerate Gaussians.
    covars = np.maximum(covars, varianceFloor)

    return means, covars
