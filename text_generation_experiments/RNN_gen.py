"""
THIS NETWORK WAS TRAINED USING THE KAGGLE NOTEBOOKS
It's kinda broken right now and I'm busy with other projects so I might come back to this later
However, it's being put on hold for now. 
"""


# GET DATA
import requests as rq
content = rq.get("http://www.gutenberg.org/cache/epub/98/pg98.txt")
open("to2c.txt", "w", encoding="utf-8").write(content)

# -------------------------------------------------------

# TRAIN MODEL

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from string import punctuation
#import nltk
import os.path
import time
import pickle
import re

starttime = time.time()
#data = " ".join(nltk.corpus.brown.words())
data = open("../input/to2ctextdata/to2c.txt", encoding="utf-8").read()
text = re.sub(" +", " ", data.lower().translate(str.maketrans("", "", punctuation)).replace("\n", " ").replace("project gutenbergtm", ""))

tf.compat.v1.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None
)
# Apparently eager execution was turned off on my machine, turned it on
seq_len = 100
batch_size = 512
epochs = 5

n_chars = len(text)
vocab = sorted(set(text))
n_unique = len(vocab)

char2int = {c:i for i,c in enumerate(vocab)}
int2char = {i:c for i,c in enumerate(vocab)}
# save these dictionaries for later generation
pickle.dump(char2int, open(f"char2int.pickle", "wb"))
pickle.dump(int2char, open(f"int2char.pickle", "wb"))

encoded_text = np.array([char2int[x] for x in text])
char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)

sequences = char_dataset.batch(2*seq_len + 1, drop_remainder=True)

#for seq in sequences.take(2):
#    print(''.join([int2char[x] for x in seq.numpy()]))

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
for element in dataset.take(5):
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

uploadnum = 2 #used to denote which dataset backup i'm using since i'm downloading and uploading model's h5 everytime i close computer

if(os.path.exists("RNNTO2Ctextgenmodel.h5")):
    model.load_weights(f"RNNTO2Ctextgenmodel.h5")
elif(os.path.exists(f"../input/to2ch5-{uploadnum}/RNNTO2Ctextgenmodel.h5")):
    model.load_weights(f"../input/to2ch5-{uploadnum}/RNNTO2Ctextgenmodel.h5")

model.compile(optimizer=tf.keras.optimizers.Adam(), loss="categorical_crossentropy")

for epoch in range(epochs):
    print(f"Currently on Epoch {epoch+1} out of total {epochs} epochs | {(epoch/epochs)*100}% complete")
    # train the model
    model.fit(shuffleset, steps_per_epoch=(len(encoded_text) - seq_len) // batch_size, epochs=1)
    # save the model
    model.save(f"RNNTO2Ctextgenmodel.h5")

endtime = time.time()
print("time elapsed (minutes): ", (endtime-starttime)//60)

# ------------------------------------------------------------------------------------------------- #

# TEST MODEL / GENERATE TEXT

import numpy as np
import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation
import pickle

sequence_length = 100
seed = "yesterday i killed a man "
char2int = pickle.load(open(f"char2int.pickle", "rb"))
int2char = pickle.load(open(f"int2char.pickle", "rb"))
vocab_size = len(char2int)

model = Sequential([
    LSTM(256, input_shape=(sequence_length, vocab_size), return_sequences=True),
    Dropout(0.3), 
    LSTM(256), 
    Dense(vocab_size, activation="softmax"),
])
model.load_weights(f"RNNTO2Ctextgenmodel.h5")
s = seed
n_chars = 1000 #generate 1000 characters
generated = ""
for i in tqdm.tqdm(range(n_chars), "Generating text"):
    # make input seq
    X = np.zeros((1, sequence_length, vocab_size))
    for t, char in enumerate(seed):
        X[0, (sequence_length-len(seed))+t, char2int[char]] = 1
    #predict
    predicted = model.predict(X, verbose=0)[0]
    #convert vec2int
    next_ind = np.argmax(predicted)
    #convert int2char
    next_char = int2char[next_ind]
    #add char to results
    generated += next_char
    #shift seed
    seed = seed[1:] + next_char
    
print("Seed: ", s)
print("Generated text: ")
print(generated)