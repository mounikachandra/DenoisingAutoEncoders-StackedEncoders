import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from load_mnist import mnist
import numpy as np

digit_range = [0,1,2,3,4,5,6,7,8,9]

train_data, train_label, val_data, val_label, test_data, test_label = \
            mnist(noTrSamples=50, noValSamples=0, noTsSamples=1000,\
            digit_range=digit_range,\
            noTrPerClass=5, noValPerClass=0, noTsPerClass=100)
            
train_data = np.transpose(train_data)
train_label = np.transpose(train_label)
train_label = np.squeeze(train_label, axis=1)
test_data = np.transpose(test_data)
test_label = np.transpose(test_label)
test_label = np.squeeze(test_label, axis=1)

classifier = svm.SVC(C=5, gamma=0.05)

classifier.fit(train_data, train_label)

predicted = classifier.predict(test_data)
accuracy = metrics.accuracy_score(test_label, predicted)