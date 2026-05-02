import numpy as np
from collections import defaultdict

datafile = np.load("traindata.npz", allow_pickle=True)
data = datafile["data"]
stateList = datafile["stateList"]

speakers = defaultdict(list)

for item in data:
    key = item["gender"] + "_" + item["speaker"]
    speakers[key].append(item)

speaker_keys = sorted(speakers.keys())

train_items = []
val_items = []

for i, spk in enumerate(speaker_keys):
    if i % 10 == 0:
        val_items.extend(speakers[spk])
    else:
        train_items.extend(speakers[spk])

print("train utterances:", len(train_items))
print("val utterances:", len(val_items))

np.savez(
    "splitdata.npz",
    train_items=train_items,
    val_items=val_items,
    stateList=stateList
)

print("saved splitdata.npz")