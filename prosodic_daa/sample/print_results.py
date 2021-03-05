import numpy as np
from scipy import stats

data = np.load("results_pl/npbdaa/summary_files/letter_ARI.npy")
data2 = np.load("results_pl/npbdaa/summary_files/word_ARI.npy")
l_mean_1 = data.mean(axis=0)[-1]
l_std_1 = data.std(axis=0)[-1]
w_mean_1 = data2.mean(axis=0)[-1]
w_std_1 = data2.std(axis=0)[-1]
print("npbdaa:letter_ARI", l_mean_1, l_std_1)
print("npbdaa:word_ARI  ", w_mean_1, w_std_1)
data = np.load("results_pl/pdaa/summary_files/letter_ARI.npy")
data2 = np.load("results_pl/pdaa/summary_files/word_ARI.npy")
l_mean_2 = data.mean(axis=0)[-1]
l_std_2 = data.std(axis=0)[-1]
w_mean_2 = data2.mean(axis=0)[-1]
w_std_2 = data2.std(axis=0)[-1]
print("pdaa:letter_ARI", l_mean_2, l_std_2)
print("pdaa:word_ARI  ", w_mean_2, w_std_2)
data = np.load("results_pl/pdaa_f0/summary_files/letter_ARI.npy")
data2 = np.load("results_pl/pdaa_f0/summary_files/word_ARI.npy")
l_mean_3 = data.mean(axis=0)[-1]
l_std_3 = data.std(axis=0)[-1]
w_mean_3 = data2.mean(axis=0)[-1]
w_std_3 = data2.std(axis=0)[-1]
print("f0dd:letter_ARI", l_mean_3, l_std_3)
print("f0dd:word_ARI  ", w_mean_3, w_std_3)
data = np.load("results_pl/pdaa_sil/summary_files/letter_ARI.npy")
data2 = np.load("results_pl/pdaa_sil/summary_files/word_ARI.npy")
l_mean_4 = data.mean(axis=0)[-1]
l_std_4 = data.std(axis=0)[-1]
w_mean_4 = data2.mean(axis=0)[-1]
w_std_4 = data2.std(axis=0)[-1]
print("silent:letter_ARI", l_mean_4, l_std_4)
print("silent:word_ARI  ", w_mean_4, w_std_4)
print("_________________________________________________________________________________________________________")
print("pl_letter_ARI npbdaa: f0dd ", stats.ttest_ind_from_stats(l_mean_1, l_std_1, 20, l_mean_3, l_std_3, 20, False))
print("pl_word_ARI   npbdaa: f0dd ", stats.ttest_ind_from_stats(w_mean_1, w_std_1, 20, w_mean_3, w_std_3, 20, False))
print("pl_letter_ARI npbdaa:silent", stats.ttest_ind_from_stats(l_mean_1, l_std_1, 20, l_mean_4, l_std_4, 20, False))
print("pl_word_ARI   npbdaa:silent", stats.ttest_ind_from_stats(w_mean_1, w_std_1, 20, w_mean_4, w_std_4, 20, False))
print("pl_letter_ARI  f0dd :silent", stats.ttest_ind_from_stats(l_mean_3, l_std_3, 20, l_mean_4, l_std_4, 20, False))
print("pl_word_ARI    f0dd :silent", stats.ttest_ind_from_stats(w_mean_3, w_std_3, 20, w_mean_4, w_std_4, 20, False))
print("pl_letter_ARI  f0dd : pdaa ", stats.ttest_ind_from_stats(l_mean_3, l_std_3, 20, l_mean_2, l_std_2, 20, False))
print("pl_word_ARI    f0dd : pdaa ", stats.ttest_ind_from_stats(w_mean_3, w_std_3, 20, w_mean_2, w_std_2, 20, False))
print("pl_letter_ARI silent: pdaa ", stats.ttest_ind_from_stats(l_mean_4, l_std_4, 20, l_mean_2, l_std_2, 20, False))
print("pl_word_ARI   silent: pdaa ", stats.ttest_ind_from_stats(w_mean_4, w_std_4, 20, w_mean_2, w_std_2, 20, False))
print("pl_letter_ARI  pdaa :npbdaa", stats.ttest_ind_from_stats(l_mean_2, l_std_2, 20, l_mean_1, l_std_1, 20, False))
print("pl_word_ARI    pdaa :npbdaa", stats.ttest_ind_from_stats(w_mean_2, w_std_2, 20, w_mean_1, w_std_1, 20, False))
print("_________________________________________________________________________________________________________")

# a =np.arange(1,10)
# b =np.arange(10,31)
# for i in a:
#     print(i,np.loadtxt("sample_results_power_law/pdaa_sil/0"+str(i)+"/summary_files/Word_ARI.txt")[-1])
# for i in b:
#     print(i,np.loadtxt("sample_results_power_law/pdaa_sil/"+str(i)+"/summary_files/Word_ARI.txt")[-1])
# for i in a:
#     print(i,np.loadtxt("experiment21-01-26/pdaa_sil/0"+str(i)+"/summary_files/Word_ARI.txt")[-1])
# for i in b:
#     print(i,np.loadtxt("experiment21-01-26/pdaa_sil/"+str(i)+"/summary_files/Word_ARI.txt")[-1])
