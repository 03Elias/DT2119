import os
import numpy as np

from lab3_tools import path2info, loadAudio
from lab1_proto import mfcc, mspec
from lab3_proto import words2phones, forcedAlignment
from prondict import prondict

phoneHMMs = np.load(
    "lab2_models_all.npz",
    allow_pickle=True
)["phoneHMMs"].item()

phones = sorted(phoneHMMs.keys())

nstates = {
    phone: phoneHMMs[phone]["means"].shape[0]
    for phone in phones
}

stateList = [
    phone + "_" + str(i)
    for phone in phones
    for i in range(nstates[phone])
]

def process_dataset(datapath, outfilename):
    data = []

    for root, dirs, files in os.walk(datapath):
        for file in files:
            if file.endswith(".wav"):
                filename = os.path.join(root, file)
                print("processing:", filename)

                samples, samplingrate = loadAudio(filename)

                lmfcc = mfcc(samples)
                mspec_feat = mspec(samples)

                gender, speaker, digits, repetition = path2info(filename)

                wordTrans = list(digits)
                phoneTrans = words2phones(wordTrans, prondict)

                targets_str = forcedAlignment(lmfcc, phoneHMMs, phoneTrans)

                targets = np.array([
                    stateList.index(s)
                    for s in targets_str
                ])

                data.append({
                    "filename": filename,
                    "gender": gender,
                    "speaker": speaker,
                    "digits": digits,
                    "repetition": repetition,
                    "lmfcc": lmfcc,
                    "mspec": mspec_feat,
                    "targets": targets
                })

    np.savez(outfilename, data=data, stateList=stateList)
    print("saved:", outfilename)


process_dataset(
    "tidigits/disc_4.1.1/tidigits/train",
    "traindata.npz"
)

process_dataset(
    "tidigits/disc_4.2.1/tidigits/test",
    "testdata.npz"
)




#train = np.load("traindata.npz", allow_pickle=True)

#print(train["data"].shape)