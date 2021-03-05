#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from tqdm import trange, tqdm
from sklearn.metrics import adjusted_rand_score

#%%
def get_names(dataset_dir):
    return np.loadtxt(dataset_dir + "files.txt", dtype=str)

def get_labels(dataset_dir):
    letter_label = np.loadtxt(dataset_dir + "LABEL/9_3.lab")
    word_label = np.loadtxt(dataset_dir + "LABEL/9_3.lab2")
    return letter_label, word_label

def get_datas_and_length(dataset_dir):
    data = np.loadtxt(dataset_dir + "DATA/9_3.txt")
    f0 = np.loadtxt(dataset_dir + "DATA/9_3_f0dd.txt")
    sil = np.loadtxt(dataset_dir + "DATA/9_3_sil.txt")
    length = len(data)
    return data, length, f0, sil

def get_results(names, length):
    letter_results = [np.loadtxt("../../experimental_results/tcds/murakami/pdaa_f0/" + name + "/results/9_3_l.txt").reshape((-1, length)) for name in names]
    word_results = [np.loadtxt("../../experimental_results/tcds/murakami/pdaa_f0/" + name + "/results/9_3_s.txt").reshape((-1, length)) for name in names]
    dur_results = [np.loadtxt("../../experimental_results/tcds/murakami/pdaa_f0/" + name + "/results/9_3_d.txt").reshape((-1, length)) for name in names]
    return letter_results, word_results, dur_results

def _plot_discreate_sequence(true_data, title, sample_data, plotopts = {}, cmap = None, cmap2 = None):
        ax = plt.subplot2grid((10, 1), (1, 0))
        plt.sca(ax)
        ax.matshow([true_data], aspect = 'auto', cmap=cmap)
        ax.axes.yaxis.set_ticks([])
        plt.ylabel('Truth Label\n')
        #label matrix
        ax = plt.subplot2grid((10, 1), (2, 0), rowspan = 8)
        plt.suptitle(title)
        plt.sca(ax)
        if cmap2 is not None:
            cmap = cmap2
        ax.matshow(sample_data, aspect = 'auto', **plotopts, cmap=cmap)
        #write x&y label
        plt.xlabel('Frame')
        plt.ylabel('Iteration')
        plt.xticks(())

#%%
if not os.path.exists("figures"):
    os.mkdir("figures")

#%%
dataset_dir = "murakami_dataset/"

#%%
names = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']
data, length, f0, sil = get_datas_and_length(dataset_dir)
l_label, w_label = get_labels(dataset_dir)

l_results, w_results, d_results = get_results(names, length)

print("Done!")

L = 10
S = 10

#%%
lcolors = ListedColormap([cm.tab20(float(i)/L) for i in range(L)])
wcolors = ListedColormap([cm.tab20(float(i)/S) for i in range(S)])

#%%
print("Plot results...")
for i, name in enumerate(tqdm(names)):
    plt.clf()
    _plot_discreate_sequence(l_label, name + "_l", l_results[i], cmap=lcolors)
    plt.savefig("figures/" + name + "_l.png")
    plt.clf()
    _plot_discreate_sequence(w_label, name + "_s", w_results[i], cmap=wcolors)
    plt.savefig("figures/" + name + "_s.png")
    plt.clf()
    _plot_discreate_sequence(w_label, name + "_d", d_results[i], cmap=wcolors, cmap2=cm.binary)
    plt.savefig("figures/" + name + "_d.png")
print("Done!")
