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
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding, Flatten
from string import punctuation
from nltk.tokenize import TweetTokenizer
import os.path
import time
import pickle
import re

starttime = time.time()

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


def one_hot(inp, target):
    return tf.one_hot(inp, n_unique), tf.one_hot(target, n_unique)
#"""
# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# instantiating the model in the strategy scope creates the model on the TPU
with tpu_strategy.scope():

    #model = tf.keras.Sequential( … ) # define your model normally
    #model.compile( … )

    # train model normally
    # model.fit(training_dataset, epochs=EPOCHS, steps_per_epoch=…)
    batch_size = 16 * tpu_strategy.num_replicas_in_sync
    #"""
    #batch_size=256
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
    #data = " ".join(nltk.corpus.brown.words())
    data = open("../input/to2ctextdata/to2c.txt", encoding="utf-8").read()
    text = re.sub(" +", " ", data.lower()).replace("\n", " ").replace("project gutenbergtm", "")
    # text = re.sub(r"[,.;@#?!&$-]+\ *", " ", text)
    text = re.sub(r"[@#&$-]+\ *", " ", text)

    tk = TweetTokenizer()

    textList = tk.tokenize(text)
    print(textList[0:10])

    tf.compat.v1.enable_eager_execution(
        config=None, device_policy=None, execution_mode=None
    )
    # Apparently eager execution was turned off on my machine, turned it on
    seq_len = 30
    #batch_size = 512
    epochs = 13
    #tokenized_text = tknzr.tokenize(" ".join(text))
    n_chars = len(textList)
    vocab = sorted(set(textList))
    n_unique = len(vocab)
    print("Number of words in vocab: ", n_unique)
    #print(vocab)

    word2num = {w:(n) for n,w in enumerate(vocab)}
    num2word = {n:w for n,w in enumerate(vocab)}
    # save these dictionaries for later generation
    pickle.dump(word2num, open(f"word2num.pickle", "wb"))
    pickle.dump(num2word, open(f"num2word.pickle", "wb"))

    encoded_text = np.array([word2num[x] for x in textList])
    char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)

    sequences = char_dataset.batch(2*seq_len + 1, drop_remainder=True)

    dataset = sequences.flat_map(split_sample)

    #for seq in sequences.take(2):
    #    print(''.join([int2char[x] for x in seq.numpy()]))

    dataset = dataset.map(one_hot)

    #print first two
    #for element in dataset.take(5):
    #    print("Input:", ''.join([num2word[np.argmax(char_vector)] for char_vector in element[0].numpy()]))
    #    print("Target:", num2word[np.argmax(element[1].numpy())])
    #    print("Input shape:", element[0].shape)
    #    print("Target shape:", element[1].shape)
    #    print("="*50, "\n")

    shuffleset = dataset.repeat().shuffle(1024).batch(batch_size, drop_remainder=True)


    model = Sequential([
        LSTM(512, input_shape=(seq_len, n_unique), return_sequences=True),
        Dropout(0.3),
        LSTM(512),
        #Embedding(32, 32), 
        #Flatten(), 
        Dense(n_unique, activation="softmax"),
    ])

    if(os.path.exists("RNNTO2Ctextgenmodel_W_PUNCT_TPU.h5")):
        model.load_weights(f"RNNTO2Ctextgenmodel_W_PUNCT_TPU.h5")
    else:
        model.load_weights(f"../input/cccccc/RNNTO2Ctextgenmodel_W_PUNCT.h5")

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="categorical_crossentropy")

    for epoch in range(epochs):
        print(f"Currently on Epoch {epoch+1} out of total {epochs} epochs")
        # train the model
        model.fit(shuffleset, steps_per_epoch=(len(encoded_text) - seq_len) // batch_size, epochs=1)
        # save the model
        model.save(f"./RNNTO2Ctextgenmodel_W_PUNCT_TPU.h5")

endtime = time.time()
print("time elapsed (minutes): ", (endtime-starttime)//60)
# ------------------------------------------------------------------------------------------------- #

# TEST MODEL / GENERATE TEXT

import numpy as np
import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation
import pickle
from nltk.tokenize import TweetTokenizer

# detect and init the TPU
#tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
#tf.config.experimental_connect_to_cluster(tpu)
#tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
#tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# instantiating the model in the strategy scope creates the model on the TPU
#with tpu_strategy.scope():
tk = TweetTokenizer()
sequence_length = 30
seed = "after seeing death, i"
word2num = pickle.load(open(f"word2num.pickle", "rb"))
num2word = pickle.load(open(f"num2word.pickle", "rb"))
vocab_size = len(word2num.keys())
#print(char2int)

model = Sequential([
    LSTM(512, input_shape=(seq_len, n_unique), return_sequences=True),
    Dropout(0.3),
    LSTM(512),
    #Embedding(32, 32), 
    #Flatten(), 
    Dense(n_unique, activation="softmax"),
])

model.load_weights(f"./RNNTO2Ctextgenmodel_W_PUNCT_TPU.h5")
s = tk.tokenize(seed)
#s = seed
n_chars = 1000 #generate 2000 characters
generated = []

for i in tqdm.tqdm(range(n_chars), "Generating text"):
    # make input seq
    X = np.zeros((1, sequence_length, vocab_size))
    for t, char in enumerate(s):
        X[0, (sequence_length-len(s))+t, word2num[char]] = 1
    #predict
    predicted = model.predict(X, verbose=0)[0]
    #convert vec2int
    next_ind = np.argmax(predicted)
    #convert int2char
    next_char = num2word[next_ind]
    #add char to results
    generated.append(next_char)
    #shift seed
    s = s[1:] + [next_char]

print("Seed: ", seed)
print("Generated text: ")
print(" ".join(generated))