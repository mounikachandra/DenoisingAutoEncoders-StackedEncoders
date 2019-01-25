import numpy as np
import os
import pdb
import matplotlib.pyplot as plt

#datasets_dir = '/Users/cubic/hemanth/S2018/cse591/miniProjects/data/'
datasets_dir = './data/'

def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


def mnist(noTrSamples=1000, noValSamples = 400,noTsSamples=100, \
                        digit_range=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], \
                        noTrPerClass=100, noValPerClass=10 ,noTsPerClass=10):
    assert noTrSamples==noTrPerClass*len(digit_range), 'noTrSamples and noTrPerClass mismatch'
    assert noTsSamples==noTsPerClass*len(digit_range), 'noTrSamples and noTrPerClass mismatch'
    data_dir = os.path.join(datasets_dir, 'mnist/')
    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trData = loaded[16:].reshape((60000, 28*28)).astype(float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trLabels = loaded[8:].reshape((60000)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsData = loaded[16:].reshape((10000, 28*28)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsLabels = loaded[8:].reshape((10000)).astype(float)

    trData = trData/255.
    tsData = tsData/255.

    tsX = np.zeros((noTsSamples, 28*28))
    trX = np.zeros((noTrSamples-noValSamples, 28*28))
    valX= np.zeros((noValSamples, 28*28))
    tsY = np.zeros(noTsSamples)
    trY = np.zeros(noTrSamples-noValSamples)
    valY = np.zeros(noValSamples)

    count = 0
    for ll in digit_range:
        # Train data
        idl = np.where(trLabels == ll)
        #print(idl)
        idl1 = idl[0][: (noTrPerClass-noValPerClass)]
        idl2 = idl[0][(noTrPerClass-noValPerClass):noTrPerClass]
        #print(idl1)
        #print(idl2)
        idx1 = list(range(count*(noTrPerClass-noValPerClass), (count+1)*(noTrPerClass-noValPerClass)))
        idx2 = list(range(count*(noValPerClass), (count+1)*noValPerClass))
        #print(idx1)
        #print(idx2)
        trX[idx1, :] = trData[idl1, :]
        trY[idx1] = trLabels[idl1]
        #print(trY)
        # Val data
        valX[idx2, :] = trData[idl2, :]
        valY[idx2] = trLabels[idl2]
        # Test data
        idl = np.where(tsLabels == ll)
        idl = idl[0][: noTsPerClass]
        idx = list(range(count*noTsPerClass, (count+1)*noTsPerClass))
        tsX[idx, :] = tsData[idl, :]
        tsY[idx] = tsLabels[idl]
        count += 1
    
    np.random.seed(1)
    test_idx = np.random.permutation(tsX.shape[0])
    tsX = tsX[test_idx,:]
    tsY = tsY[test_idx]

    trX = trX.T
    tsX = tsX.T
    valX = valX.T
    trY = trY.reshape(1, -1)
    valY = valY.reshape(1, -1)
    tsY = tsY.reshape(1, -1)
    return trX, trY, valX, valY, tsX, tsY


def main():
    trX, trY, valX, valY, tsX, tsY = mnist(noTrSamples=21,noValSamples=6,
                               noTsSamples=9, digit_range=[0, 5, 8],
                               noTrPerClass=7,noValPerClass=2, noTsPerClass=3)

    plt.imshow(trX[:,5].reshape(28, -1))
    trY[0,5]
    
if __name__ == "__main__":
    main()
