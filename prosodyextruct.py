import os
import re
import pyreaper
import numpy as np
import matplotlib.pyplot as plt
from python_speech_features import delta as delta_mfcc
from speech_feature_extraction import Extractor
from speech_feature_extraction.util import WavLoader
from scipy.io import wavfile
try:
    from tqdm import tqdm
except:
    def tqdm(x): return x

def get_names(dataset_dir):
    names = np.loadtxt("data/" + dataset_dir + "files.txt", dtype=str)
    np.savetxt("results/files.txt", names, fmt="%s")
    return names

def load_data(name, dataset_dir):
    file = "data/" + dataset_dir + name + ".wav"
    fs, x = wavfile.read(file)
    if x.ndim == 2:
        x = x[:,0].copy(order='C')
        #x = x[:,1].copy(order='C')
        #x = x.mean(axis=0)
    return (x,fs)

def load_lab_conv(name, length, fs, pdict, wdict, dataset_dir, wrddist):
    phn = np.loadtxt("data/" + dataset_dir + name + ".lab", dtype=[('col1', 'f16'), ('col2', 'f16'), ('col3', 'S10')])
    wrd = np.loadtxt("data/" + dataset_dir + name + ".lab2", dtype=[('col1', 'f16'), ('col2', 'f16'), ('col3', 'S10')])
    phn_frm = np.empty(length)
    lab_len = len(phn)
    adj = length / (phn[-1][1] * fs)
    prev = 0
    for i in range(lab_len):
        if i == lab_len - 1:
            end = length
        else:
            end = int(phn[i][1] * fs * adj)
        if phn[i][2] in pdict:
            phn_frm[prev:end] = pdict[phn[i][2]]
        else:
            pdict[phn[i][2]] = len(pdict)
            phn_frm[prev:end] = pdict[phn[i][2]]
        prev = end
    wrd_frm = np.empty(length)
    lab_len = len(wrd)
    adj = length / (wrd[-1][1] * fs)
    prev = 0
    for i in range(len(wrd)):
        if i == lab_len - 1:
            end = length
        else:
            end = int(wrd[i][1] * fs * adj)
        if wrd[i][2] in wdict:
            wrddist[wdict[wrd[i][2]]]+=1
            wrd_frm[prev:end] = wdict[wrd[i][2]]
        else:
            wdict[wrd[i][2]] = len(wdict)
            wrd_frm[prev:end] = wdict[wrd[i][2]]
            wrddist[wdict[wrd[i][2]]]+=1
        prev = end + 1
    return (phn_frm, wrd_frm, pdict, wdict)

def plotfig(name,x,fs,f0,sil):
    time = len(x)/fs
    reaper_time = np.linspace(0, time, len(f0))
    plt.clf()
    plt.figure(figsize=(16, 9), dpi=100)
    ax1 = plt.subplot2grid((5,1), (0,0))
    ax2 = plt.subplot2grid((5,1), (1,0))
    ax3 = plt.subplot2grid((5,1), (2,0))
    ax4 = plt.subplot2grid((5,1), (3,0))
    ax5 = plt.subplot2grid((5,1), (4,0))
    ax1.set_title('spectrogram')
    ax1.set_ylabel('frequency')
    pxx, freqs, bins, im = ax1.specgram(x, Fs=fs)
    ax2.set_title('f0 contour')
    ax2.set_ylabel('frequency')
    ax2.set_xlim(0, np.max(time))
    ax2.plot(reaper_time, f0[:,0], linewidth=1)
    ax2.set_ylim(0, )
    ax3.set_title('f0 delta')
    ax3.set_ylabel('f0 delta')
    ax3.set_xlim(0, np.max(time))
    ax3.plot(reaper_time, f0[:,1], linewidth=1)
    ax4.set_title('f0 delta delta')
    ax4.set_ylabel('f0 delta delta')
    ax4.set_xlim(0, np.max(time))
    ax4.plot(reaper_time, f0[:,2], linewidth=1)
    ax5.set_title('silent interval')
    ax5.set_xlabel('time [sec]')
    ax5.set_ylabel('length [msec]')
    ax5.set_xlim(0, np.max(time))
    ax5.plot(reaper_time, sil, linewidth=1)
    ax2.set_ylim(0, )
    plt.tight_layout()
    plt.savefig("results/figures/" + name + ".png")
    plt.close()

def delta(sdata, window = 1, order = 1):
    data = np.pad(sdata, (window, window), mode='constant', constant_values=-1)
    #data = np.pad(sdata, (window, window), mode='edge')
    difdata = np.zeros(len(sdata))
    for i in range(len(sdata)):
        difdata[i] = np.dot(np.arange(-window, window+1), data[i : i+2*window+1]) / (2 * sum([j**2 for j in range(1, window+1)]))
    if order > 1:
        difdata = np.vstack((difdata, delta(sdata=difdata, window=window, order=order-1)))
    return difdata

def sil_cut(sdata, phn, wrd, fs, sil_len = 0.2, sil_thr = -16, sil_edg = 0.01):
    data_len = len(sdata)
    sil_feature = np.zeros(data_len)
    sil_len = int(sil_len * fs)
    if sil_len > data_len or sil_len < sil_edg:
        return (sdata, sil_feature, phn, wrd)
    if sil_thr != None:
        sil_thr = (10 ** (sil_thr/10)) * sdata.max()
    else:
        print(sdata.min(), (10 ** (-16/10)) * sdata.max())
        sil_thr = 10
    sil_det = np.where(sdata <= sil_thr)
    if not sil_det:
        return (sdata, sil_feature, phn, wrd)
    sil_int = []
    start = sil_det[0][0]
    prev = sil_det[0][0]
    cont = 0
    sil_det_len = len(sil_det[0])
    for i in range(sil_det_len):
        if sil_det[0][i] - prev != 1 or i == sil_det_len - 1:
            if cont == 1:
                sil_int.insert(0, [start, sil_det[0][i]])
            cont = 0
            start = sil_det[0][i]
        elif cont == 0 and (sil_det[0][i] - start) >= sil_len:
            cont = 1
        prev = sil_det[0][i]
    if not sil_int:
        return (sdata, sil_feature, phn, wrd)
    sil_edg = int(sil_edg * fs)
    data = sdata
    for i, j in sil_int:
        if i != 0:
            i += sil_edg
        data = np.delete(data, range(i,j+1))
        sil_feature = np.delete(sil_feature, range(i,j+1))
        phn = np.delete(phn, range(i,j+1))
        wrd = np.delete(wrd, range(i,j+1))
        if i != 0:
            sil_feature[i - 1] = (j+1 - i) / fs
    sil_feature[-1] = 0
    return (data, sil_feature, phn, wrd)

def silent_fit(silent, fs, frame_period=0.01, window_len=0.025):
    window_len *= fs
    frame_period *= fs
    silent_fit = []
    if int(frame_period - ((len(silent) - window_len) % frame_period)) != frame_period:
        silent = np.pad(silent, (0, int(frame_period - ((len(silent) - window_len) % frame_period))), mode='constant', constant_values=0)
    for i in range(int((len(silent) - window_len + frame_period) / frame_period)):
        silent_fit = np.append(silent_fit, np.sum(silent[int(i * frame_period):int(i * frame_period + window_len - 1)]))
    return silent_fit

def label_fit(phn, wrd, fs, frame_period=0.01, window_len=0.025):
    window_len *= fs
    frame_period *= fs
    if int(frame_period - ((len(phn) - window_len) % frame_period)) != frame_period:
        phn = np.pad(phn, (0, int(frame_period - ((len(phn) - window_len) % frame_period))), mode='edge')
        wrd = np.pad(wrd, (0, int(frame_period - ((len(wrd) - window_len) % frame_period))), mode='edge')
    phn_fit = []
    wrd_fit = []
    for i in range(int((len(phn) - window_len + frame_period) / frame_period)):
        phn_fit = np.append(phn_fit, phn[int(i * frame_period + (window_len / 2))])
        wrd_fit = np.append(wrd_fit, wrd[int(i * frame_period + (window_len / 2))])
    return phn_fit, wrd_fit

if not os.path.exists("results"):
    os.mkdir("results")
# if not os.path.exists("results/WAVE"):
#     os.mkdir("results/WAVE")
# if not os.path.exists("results/figures"):
#     os.mkdir("results/figures")

dataset_dir = "aioi_dataset/"

extractor = Extractor(WavLoader)
names = get_names(dataset_dir)
pdict = {}
wdict = {}
mfcc = {}
mfccd = {}
mfccdd = {}
f0dd = {}
silent = {}
phn_lab = {}
wrd_lab = {}
f0dd_max = 0
sil_max = 0
wrddist = np.zeros(50)
for name in tqdm(names):
    y,fs = load_data(name, dataset_dir)
    phn, wrd, pdict, wdict = load_lab_conv(name, len(y), fs, pdict, wdict, dataset_dir, wrddist)
    x, sil, phn, wrd = sil_cut(y, phn, wrd, fs, sil_len=0.01, sil_thr=-8, sil_edg=0) #aioi_dataset
    #x, sil, phn, wrd = sil_cut(y, phn, wrd, fs, sil_len=0.03, sil_thr=-24, sil_edg=0) #murakami_dataset
    #x, sil, phn, wrd = sil_cut(y, phn, wrd, fs, sil_len=0.35, sil_thr=-10, sil_edg=0.15) #murakami_dataset
    sil = silent_fit(sil, fs, frame_period=0.01, window_len=0.025)
    phn, wrd = label_fit(phn, wrd, fs, frame_period=0.01, window_len=0.025)
    pm_times, pm, f0_times, f0, corr = pyreaper.reaper(x, fs, minf0=40.0, maxf0=300.0, frame_period=0.01)
    f0 = np.pad(f0, (0, len(sil)-len(f0)), 'constant')
    f0_delta = delta(sdata = f0, window = 2, order=2)
    s = extractor._mfcc_cord(x, fs)
    if f0dd_max < f0_delta[1].max():
        f0dd_max = f0_delta[1].max()
    if sil_max < sil.max():
        sil_max = sil.max()
    d = delta_mfcc(s, 2)
    dd = delta_mfcc(d, 2)
    mfcc[name] = s
    mfccd[name] = d
    mfccdd[name] = dd
    phn_lab[name] = phn
    wrd_lab[name] = wrd
    silent[name] = sil
    f0dd[name] = f0_delta[1]
    check = s.shape[0]
    if check != d.shape[0] or check != dd.shape[0] or check != phn.shape[0] or check != wrd.shape[0] or check != sil.shape[0] or check != f0_delta[1].shape[0]:
        print(name, s.shape, d.shape, dd.shape, phn.shape, wrd.shape, sil.shape, f0_delta[1].shape)
        assert 0
    # wavfile.write("results/WAVE/" + name + ".wav", fs, x)
    # plotfig(name, x, fs, np.vstack((f0, f0_delta)).T, sil)
print(pdict, wdict, wrddist)
for key in names:
    f0dd[key][np.where(f0dd[key] < 0)] = 0
    if f0dd_max > 0:
        f0dd[key] /= f0dd_max
    if sil_max > 0:
        silent[key] /= sil_max
    silent[key][-1] = 1
np.savez("results/mfcc_12dim.npz", **mfcc)
np.savez("results/mfcc_delta_12dim.npz", **mfccd)
np.savez("results/mfcc_delta_delta_12dim.npz", **mfccdd)
np.savez("results/phoneme_label.npz", **phn_lab)
np.savez("results/word_label.npz", **wrd_lab)
np.savez("results/silent_feature.npz", **silent)
np.savez("results/f0_delta_delta.npz", **f0dd)
