from sklearn import datasets
import random
import tensorflow as tf
from tensorflow import keras
import numpy as np

#Thanks to https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/

iris = datasets.load_iris()
irisdat = iris.data
#print(irisdat)
numTypes = 3
#total of 150 different things in the iris dataset
#4 attributes
#first 50 are setosa, second 50 are versicolour, last 50 are virginica
val = []

for i in range(len(irisdat)):

    u = irisdat[i]
    if(i<=50): 
        val.append([u, [1, 0, 0]])
    elif(50 < i and i <= 100):
        val.append([u, [0, 1, 0]])
    elif(100 < i and i <= 150):
        val.append([u, [0, 0, 1]])

random.shuffle(val)

training = val[0:99]
trainX, trainy = np.array([np.array(training[i][0]) for i in range(len(training))]), np.array([np.array(training[i][1]) for i in range(len(training))])
testing = val[100:]
testX, testy = np.array([np.array(testing[i][0]) for i in range(len(testing))]), np.array([np.array(testing[i][1]) for i in range(len(testing))])

iters = 3000
alpha = 0.1

model = keras.Sequential([

    keras.layers.Dense(units=5, activation='sigmoid'), 
    keras.layers.Dense(units=6, activation='sigmoid'),
    keras.layers.Dense(units=3, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.SGD(lr=alpha, momentum=0.9), metrics=['accuracy'])
print("----------- about to start fitting model ------------")
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=iters, verbose=0)
print("--------- about to test predictions ------------")
test_eval = model.evaluate(testX, testy, verbose=0)
print("Accuracy on testing set: " + str(test_eval))
print("Accuracy on training set: " + str(model.evaluate(trainX, trainy, verbose=0)))