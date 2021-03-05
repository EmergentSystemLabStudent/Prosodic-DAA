import numpy as np
import tensorflow as tf
from DSAE_PBHL import AE, SAE, SAE_PBHL
from DSAE_PBHL import DSAE, DSAE_PBHL
from DSAE_PBHL.util import Builder

def packing(np_objs):
    return np.concatenate(np_objs, axis=0)

def packing_pb(np_objs, lengths, speaker_N, hot_val=1, cold_val=0):
    T = sum(lengths)
    cumsum_lens = np.concatenate(([0], np.cumsum(lengths)))
    pb = np.ones((T, speaker_N)) * cold_val
    for i, id in enumerate(np_objs):
        pb[cumsum_lens[i]:cumsum_lens[i+1], id] = hot_val
    return pb

def unpacking(np_obj, lengths):
    cumsum_lens = np.concatenate(([0], np.cumsum(lengths)))
    N = len(lengths)
    return [np_obj[cumsum_lens[i]:cumsum_lens[i+1]] for i in range(N)]

def normalize(data, lower=0.0, upper=1.0, axis=None):
    min_v = data.min(axis=axis)
    max_v = data.max(axis=axis)
    data = (data - min_v) / (max_v - min_v) # normalize to [0, 1]
    data = data * (upper - lower) + lower # normalize to [lower, upper]
    return data

print("loading data...")
# static = np.load("feature/hicut_logpowspec.npz")
static = np.load("results/mfcc_12dim.npz")
# static = np.load("results/mfcc_delta_12dim.npz")
# static = np.load("results/mfcc_delta_delta_12dim.npz")
# speaker_id = np.load("feature/speaker_id.npz")

keys = list(static.keys())
lengths = [static[key].shape[0] for key in keys]
T = sum(lengths)

print("packing data...")
static_packed = packing([static[key] for key in keys])

print("defining networks...")
structure = [12, 8, 5, 3]
# structure = [50, 32, 16, 8, 5, 3]
L = len(structure)
builder = Builder(structure[0])
for dim in structure[1:]:
    builder.stack(SAE, dim)
builder.print_recipe()

with tf.variable_scope("dsae"):
    dsae = builder.build()

print("normalizing data...")
static_packed = normalize(static_packed, lower=-1, upper=1, axis=0)

train_flag = np.random.rand(T) < 1.1

data_all = static_packed
data_train = data_all[train_flag]
data_cross = data_all[~train_flag]

print("training networks...")
epoch = 10
# threshold = 2.0E-6
# threshold = 1.0E-12
threshold = 1.0E-60
with tf.Session() as sess:
    initializer = tf.global_variables_initializer()
    sess.run(initializer)
    global_step = 0
    for i in range(L-1):
        print(f"Training {i+1} th network (all:{L-1})")
        summary_writer = tf.summary.FileWriter(f"graph/{i+1}_th_network_train", sess.graph)
        global_step += dsae.fit_until(sess, i, data_train, epoch, threshold, summary_writer=summary_writer, global_step=global_step)
        # summary_writer       = tf.summary.FileWriter(f"graph/{i+1}_th_network_train", sess.graph)
        # cross_summary_writer = tf.summary.FileWriter(f"graph/{i+1}_th_network_cross", sess.graph)
        # last_loss = dsae.losses_with_eval(sess, data_train)[i]
        # while True:
        #     loss, c_loss = dsae.fit_with_cross(sess, i, data_train, data_cross, epoch,
        #         summary_writer=summary_writer, cross_summary_writer=cross_summary_writer)
        #     if abs(last_loss - loss) < threshold:
        #         break
        #     last_loss = loss
    compressed = dsae.hidden_layers_with_eval(sess, data_all)[-1]

print("max value:{}".format(compressed.max(axis=0)))
print("min value:{}".format(compressed.min(axis=0)))

print("unpacing data...")
unpacked = unpacking(compressed, lengths)

print("making feature dict...")
compressed = {}
for i, key in enumerate(keys):
    compressed[key] = unpacked[i]

print("saving data...")
np.savez("results/mfcc_3dim.npz", **compressed)
# np.savez("results/mfcc_delta_3dim.npz", **compressed)
# np.savez("results/mfcc_delta_delta_3dim.npz", **compressed)

print("Finished!!")
