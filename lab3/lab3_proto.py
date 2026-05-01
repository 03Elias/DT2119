import numpy as np
from lab3_tools import *
from lab1_proto import mfcc, mspec
from lab2_proto import concatHMMs, viterbi
from lab2_tools import log_multivariate_normal_density_diag
from prondict import prondict

def words2phones(wordList, pronDict, addSilence=True, addShortPause=True):
    """ word2phones: converts word level to phone level transcription adding silence

    Args:
       wordList: list of word symbols
       pronDict: pronunciation dictionary. The keys correspond to words in wordList
       addSilence: if True, add initial and final silence
       addShortPause: if True, add short pause model "sp" at end of each word
    Output:
       list of phone symbols
    """
    phoneTrans = []

    # add initial silence
    if addSilence:
        phoneTrans.append('sil')

    # convert each word into phones
    for word in wordList:

        # add phones from dictionary
        phoneTrans.extend(pronDict[word])

        # add short pause
        if addShortPause:
            phoneTrans.append('sp')

    # add final silence
    if addSilence:
        phoneTrans.append('sil')

    return phoneTrans 
    

def forcedAlignment(lmfcc, phoneHMMs, phoneTrans):
    """ forcedAlignmen: aligns a phonetic transcription at the state level

    Args:
       lmfcc: NxD array of MFCC feature vectors (N vectors of dimension D)
              computed the same way as for the training of phoneHMMs
       phoneHMMs: set of phonetic Gaussian HMM models
       phoneTrans: list of phonetic symbols to be aligned including initial and
                   final silence

    Returns:
       list of strings in the form phoneme_index specifying, for each time step
       the state from phoneHMMs corresponding to the viterbi path.
    """
def forcedAlignment(lmfcc, phoneHMMs, phoneTrans):
    """Aligns a phonetic transcription at the state level."""

    # 1. Build one big HMM for the whole utterance
    utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)

    # 2. Number of emitting states for each phone
    nstates = {
        phone: phoneHMMs[phone]['means'].shape[0]
        for phone in phoneHMMs.keys()
    }

    # 3. Map utteranceHMM state index -> unique phone_state name
    stateTrans = [
        phone + '_' + str(stateid)
        for phone in phoneTrans
        for stateid in range(nstates[phone])
    ]

    # 4. Compute log emission likelihoods
    obsloglik = log_multivariate_normal_density_diag(
        lmfcc,
        utteranceHMM['means'],
        utteranceHMM['covars']
    )

    # 5. Run Viterbi
    viterbiLoglik, viterbiPath = viterbi(
        obsloglik,
        np.log(utteranceHMM['startprob']),
        np.log(utteranceHMM['transmat'])
    )

    # 6. Convert state indexes to state names
    viterbiStateTrans = [
        stateTrans[state]
        for state in viterbiPath
    ]

    return viterbiStateTrans
