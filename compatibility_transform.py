import os
import numpy as np
try:
    from tqdm import tqdm
except:
    def tqdm(x): return x

data = np.load("results/mfcc_3dim.npz")
data_delta = np.load("results/mfcc_delta_3dim.npz")
data_delta_delta = np.load("results/mfcc_delta_delta_3dim.npz")
phoneme = np.load("results/phoneme_label.npz")
word = np.load("results/word_label.npz")
sil = np.load("results/silent_feature.npz")
f0dd = np.load("results/f0_delta_delta.npz")

if not os.path.exists("results/DATA"):
    os.mkdir("results/DATA")
if not os.path.exists("results/LABEL"):
    os.mkdir("results/LABEL")

for key in tqdm(data.keys()):
    np.savetxt("results/DATA/" + key + ".txt", data[key])
    np.savetxt("results/DATA/" + key + "_d.txt", data_delta[key])
    np.savetxt("results/DATA/" + key + "_dd.txt", data_delta_delta[key])
    np.savetxt("results/DATA/" + key + "_sil.txt", sil[key])
    np.savetxt("results/DATA/" + key + "_f0dd.txt", f0dd[key])
    np.savetxt("results/LABEL/" + key + ".lab", phoneme[key])
    np.savetxt("results/LABEL/" + key + ".lab2", word[key])
