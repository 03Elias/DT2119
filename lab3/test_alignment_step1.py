import numpy as np

from lab3_proto import forcedAlignment

example = np.load('lab3/lab3_example.npz', allow_pickle=True)['example'].item()

lmfcc = example['lmfcc']
phoneTrans = example['phoneTrans']

phoneHMMs = np.load('lab3/lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()

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
    
    