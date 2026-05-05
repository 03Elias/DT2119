import numpy as np

from lab3_proto import forcedAlignment
from lab3_proto import words2phones
from prondict import prondict
from lab3_tools import frames2trans

example = np.load('lab3_example.npz', allow_pickle=True)['example'].item()

lmfcc = example['lmfcc']
phoneTrans = example['phoneTrans']

phoneHMMs = np.load('lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()

alignment = forcedAlignment(lmfcc, phoneHMMs, phoneTrans)

print(len(alignment))
print(alignment[:20])
print(alignment[-20:])

gold = example['viterbiStateTrans']

print("same length:", len(alignment) == len(gold))
print("exact match:", alignment == list(gold))

diffs = [i for i, (a, g) in enumerate(zip(alignment, gold)) if a != g]

print("number of different frames:", len(diffs))
print("first 20 diffs:", diffs[:20])

for i in diffs[:10]:
    print(i, "ours:", alignment[i], "gold:", gold[i])
    
    
    
#test words2phones
wordTrans = ['z', '4', '3']
phoneTrans_test = words2phones(wordTrans, prondict)

print(phoneTrans_test)


frames2trans(alignment, outfilename='z43a.lab')
print("saved z43a.lab")