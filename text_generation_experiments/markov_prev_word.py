"""

To run this on your own computer: 
-> Just download the text file (nltk_brown.txt) and this file
-> Run this file. Simple!

"""

import numpy as np
import re

data = open('markov_chain_textgen_code/nltk_brown.txt', encoding='utf-8').read()
corpus = data.split(" ")

#print("Number of words (includes punctuation like periods/exclamation points and quotation marks): " + str(len(corpus)))

wordpairs = []
for i in range(len(corpus)-1):
    wordpairs.append((corpus[i], corpus[i+1]))

wdict = {}

for w1, w2 in wordpairs:
    if w1 in wdict.keys():
        wdict[w1].append(w2)
    else:
        wdict[w1] = [w2]

wstart = np.random.choice(corpus)
while wstart[0].islower() and ("".join(re.findall("[a-zA-Z]+", wstart)).lower()==wstart.lower()):
    wstart = np.random.choice(corpus)

chain = [wstart]

len_text = 1000

for i in range(len_text):
    chain.append(np.random.choice(wdict[chain[-1]]))


print("---------------")
print(" ".join(chain))