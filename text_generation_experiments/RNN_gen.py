import tensorflow as tf
import numpy as np
import os
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from string import punctuation

data = open("text_generation_experiments/nltk_brown.txt", "r").read()
tf.compat.v1.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None
)
# Apparently eager execution was turned off on my machine, turned it on
seq_len = 100
batch_size = 512
epochs = 30
text = data.lower().translate(str.maketrans("", "", punctuation))

n_chars = len(text)
vocab = sorted(set(text))
n_unique = len(vocab)

char2int = {c:i for i,c in enumerate(vocab)}
int2char = {i:c for i,c in enumerate(vocab)}

encoded_text = np.array([char2int[x] for x in text])
char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)

sequences = char_dataset.batch(2*seq_len + 1, drop_remainder=True)

for seq in sequences.take(2):
    print(''.join([int2char[x] for x in seq.numpy()]))

def split_sample(sample):
    #Splits a single sample into multiple
    ds = tf.data.Dataset.from_tensors((sample[:seq_len], sample[seq_len]))
    #print(sample)
    for i in range(1, (sample.shape[0]-1)//2):
        inp = sample[i:i+seq_len] # sequence starting at i
        target = sample[i+seq_len] #value at end of sequence
        other = tf.data.Dataset.from_tensors((inp, target))
        ds = ds.concatenate(other)
    return ds

dataset = sequences.flat_map(split_sample)

def one_hot(inp, target):
    return tf.one_hot(inp, n_unique), tf.one_hot(target, n_unique)

dataset = dataset.map(one_hot)

#print first two
for element in dataset.take(2):
    print("Input:", ''.join([int2char[np.argmax(char_vector)] for char_vector in element[0].numpy()]))
    print("Target:", int2char[np.argmax(element[1].numpy())])
    print("Input shape:", element[0].shape)
    print("Target shape:", element[1].shape)
    print("="*50, "\n")

shuffleset = dataset.repeat().shuffle(1024).batch(batch_size, drop_remainder=True)

model = Sequential([
    LSTM(256, input_shape=(seq_len, n_unique), return_sequences=True),
    Dropout(0.3),
    LSTM(256), 
    Dense(n_unique, activation="softmax"),
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="categorical_crossentropy")

# make results folder if does not exist yet
if not os.path.isdir("results"):
    os.mkdir("results")
# train the model
model.fit(shuffleset, steps_per_epoch=(len(encoded_text) - seq_len) // batch_size, epochs=epochs)
# save the model
model.save(f"RNNtextgenmodel.h5")