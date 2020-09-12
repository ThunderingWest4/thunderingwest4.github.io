import numpy as np
import re
import nltk
import os

#Problems with this model: basically restating different parts of the training text and can jump to different parts and not make enough sense. In addition, can't really do much when it runs into a n-gram that it hasn't seen before
# In addition, longer grams help the program retain more context and make more sense whereas shorter grams get closer and closer to randomly getting words

data = open('text_generation_experiments/nltk_brown.txt', encoding='utf-8').read()
corpus = nltk.word_tokenize(data)

ngrams = {}
#n = 2 #bigram model
n = 3 #trigram model
#n = 4 #quadgram? four-gram? whatever you call it
filename = "text_generation_experiments/nltk_brown_"+str(n)+"grams.txt"
if(os.path.exists(filename)):
    ngrams = eval(open(filename, "r", encoding="utf-8").read())
else:
    
    for i in range(len(corpus)-n):
        ngram = " ".join(corpus[i:i+n])
        if not (ngram in ngrams.keys()):
            ngrams[ngram] = []
        ngrams[ngram].append(corpus[i+n]) # saying that this n-gram is followed by this specific word

    #keys = list(ngrams.keys())

    #for i in range(10):
    #    print(keys[i], ngrams[keys[i]]) # displaying firs 10 ngrams and word
    open(filename, "a").write(str(ngrams))

n_words = 1000

seq_start = np.random.choice([i for i in range(len(corpus)-n)])
#print(seq_start)
seq = " ".join(corpus[seq_start:seq_start+n]) #starting gram
out = seq
for i in range(n_words):
    if(seq not in ngrams.keys()):
        break
    poss = ngrams[seq]
    nxt = np.random.choice(poss)
    out += " " + nxt
    seq_words = nltk.word_tokenize(out)
    seq = ' '.join(seq_words[len(seq_words)-n:len(seq_words)])

print(out)