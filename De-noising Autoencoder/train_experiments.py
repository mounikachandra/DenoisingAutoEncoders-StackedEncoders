import numpy as np
from load_data import myDataset
from model import Model, Noise
import matplotlib.pyplot as plt
import pdb
import argparse

def train(X, X_val, net_dims, noise_level, epochs=2000, learningRate=0.1,costEstimate="MSE", decayRate = 0.5):
    learningRate=float(args.learningRate)
    # learningRate = lr
    epochs=int(args.epochs)
    decayRate = float(args.decayRate)
    batchSize=int(args.batchSize)
    n_in, n_h, n_fin = net_dims
    data=myDataset(args)
    model=Model()
    noise = Noise()
    parameters = model.initialize_2layer_weights(n_in, n_h, n_fin)

    costs = []
    costs_ = []
    
    for ii in range(epochs):
        noBatches = int(X.shape[0]/batchSize)
        learningRate = learningRate*(1/(1+decayRate*ii))
        print(learningRate)
        print("Epoch: ",ii )
        for jj in range(noBatches):

            XTrBatch= data.getTrMiniBatch()
            

            ## change this noise level to a constant when experimenting with a different parameter
            noisyXTrBatch = noise.GaussianNoise(XTrBatch, sd=noise_level)
            
            W1,b1 = parameters["W1"],parameters["b1"]
            W2,b2 = parameters["W2"],parameters["b2"]

            A1,cache1= model.layer_forward(noisyXTrBatch, W1, b1, "relu")
            A2,cache2 = model.layer_forward(A1, W2, b2, "sigmoid")

            if costEstimate == "MSE":
                cost = model.MSE(A2, XTrBatch)
                dA= (A2-XTrBatch)/(XTrBatch.shape[1])
                # dA = np.mean(2* (A2-XTrBatch))
            else:
                cost = model.crossEntropy(A2, XTrBatch)
                dA= ((A2 - X)/ (A2*(1.0 - A2)))/XTrBatch.shape[1]

            dA_prev2, dW2, db2 = model.layer_backward(dA, cache2, W2, b2, "sigmoid")
            dA_prev1, dW1, db1 = model.layer_backward(dA_prev2, cache1, W1, b1, "relu")

            parameters['W2']+=-learningRate*dW2
            parameters['b2']+=-learningRate*db2
            parameters['W1']+=-learningRate*dW1
            parameters['b1']+=-learningRate*db1

            if jj % 10 == 0:
                print("Cost at iteration %i is: %f" %(jj*5, cost))
            if jj % 20 == 0:
                XValBatch = data.getValMiniBatch()
                noisyXValBatch = noise.GaussianNoise(XValBatch, sd=noise_level)
                costs.append(cost)
                A1_,cache1_= model.layer_forward(noisyXValBatch, W1, b1, "relu")
                A2_,cache2_ = model.layer_forward(A1_, W2, b2, "sigmoid")
                if costEstimate == "MSE":
                    cost_ = model.MSE(A2_, XValBatch)
                else:
                    cost_ = model.crossEntropy(A2_, XValBatch)
                costs_.append(cost_)

    return costs,costs_, parameters

def predict(parameters):
    data=myDataset(args)
    model=Model()
    noise = Noise()
    XTsBatch= data.getTsMiniBatch()
    noisyXTsBatch = noise.GaussianNoise(XTsBatch, sd=0.3)

    W1,b1 = parameters["W1"],parameters["b1"]
    W2,b2 = parameters["W2"],parameters["b2"]

    A1,cache1= model.layer_forward(noisyXTsBatch, W1, b1, "relu")
    A2,cache2 = model.layer_forward(A1, W2, b2, "sigmoid")

    fig1=plt.figure(figsize=(8, 8))
    columns = 4
    rows = 4
    for i in range(1, columns*rows +1):
        pixels1 = A2[:,i]
        img = pixels1.reshape((28, 28))
        fig1.add_subplot(rows, columns, i)
        plt.imshow(img)
    
    fig2=plt.figure(figsize=(8, 8))
    columns = 4
    rows = 4
    for i in range(1, columns*rows +1):
        pixels2 = XTsBatch[:,i]
        img = pixels2.reshape((28, 28))
        fig2.add_subplot(rows, columns, i)
        plt.imshow(img)

    fig3=plt.figure(figsize=(8, 8))
    columns = 4
    rows = 4
    for i in range(1, columns*rows +1):
        pixels3 = noisyXTsBatch[:,i]
        img = pixels3.reshape((28, 28))
        fig3.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()


def main(args):
    data = myDataset(args)
    train_data, val_data, test_data = data.getTrData(), data.getValData(), data.getTsData()

    m,n_in = train_data.shape
    n_fin = n_in
    n_h = 1000
    net_dims = [n_in, n_h, n_fin]

    epochs = int(args.epochs)
    plot_costs=[]
    plot_costs_=[]
    test_acc=[]
    net_dims = [n_in, n_h, n_fin]

    ## Following list store the train, val costs and parameters

    trainingCost_list = []
    validationCost_list = []
    parametersDict_list = []
    # xAxis = list(range(epochs))

    ## This noise_level parameter can be changed to any other parameter and can conduct experiments
    ## Change the for loop to a list of other variables
    ## Ex: for noise_name in ['gaussian', 's&p', 'masking']
    ## tested learning rated [0.5, 0.1, 0.05, 0.005, 0.0005]

    # learning_rates = [1, 0.5, 0.1, 0.05, 0.01, 0.005]
    noise_ranges = list(np.arange(0.1,1,0.1))

    for noise_level in noise_ranges:
        costs,costs_, parameters = train(train_data, val_data, net_dims, noise_level,\
            epochs=epochs, learningRate=args.learningRate, costEstimate=args.costEstimate, decayRate = args.decayRate)
        trainingCost_list.append(costs)
        validationCost_list.append(costs_)
        parametersDict_list.append(parameters)
        
    # predict(parameters)

    ## The plots of training curve vs parameter
    xAxis = list(range(len(trainingCost_list[0])))
   
    for i in range(len(noise_ranges)):
        plt.plot(xAxis, trainingCost_list[i], label=('train' + str(noise_ranges[i])))
        plt.title('Train_cost at different noise levels')
        # plt.plot(it, costs_, label='val')
        # plt.plot(xAxis, validationCost_list[i], label=('val' + str(learning_rates[i])))
        # plt.title('Val_cost at different noise levels')
        # plt.xlabel('iterations')
        # plt.ylabel('Cost')
    plt.legend( loc=1)
    plt.figure()
    for i in range(len(noise_ranges)):
        plt.plot(xAxis, validationCost_list[i], label=('val' + str(noise_ranges[i])))
        plt.title('Val_cost at different noise levels')
        # plt.xlabel('iterations')
        # plt.ylabel('Cost')
    plt.legend( loc=1)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataPath', default='./data',
                        help="data path")
    parser.add_argument('--noTraining', default='60000',
                        help="number of training examples")
    parser.add_argument('--noValidation', default='10000',
                        help="number of validation examples")
    parser.add_argument('--noTesting', default='10000',
                        help="number of testing examples")
    parser.add_argument('--batchSize', default='64',
                        help="batch size")
    parser.add_argument('--learningRate', default='0.0001',
                        help="initial learning rate")
    parser.add_argument('--labelRange', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        help="all the labels in the output")
    parser.add_argument('--noTrPerClass', default='6000',
                        help="number of training examples per class")
    parser.add_argument('--noValPerClass', default='1000',
                        help="number of validation examples per class")
    parser.add_argument('--noTsPerClass', default='1000',
                        help="number of testing example per class")
    parser.add_argument('--epochs', default=10,
                        help="number of epochs")
    parser.add_argument('--costEstimate', default='MSE',
                        help="Loss function")
    parser.add_argument('--decayRate', default='0.5',
                        help="decay rate for learning rate")

    args = parser.parse_args()

    # args.id += '--batchSize' + str(args.batchSize)
    # args.id += '--learningRate' + str(args.learningRate)
    # args.id += '--noTraining' + str(args.noTraining)
    # args.id += '--noTesting' + str(args.noTesting)
    # args.id += '--cost' + str(args.cost)
    # args.id += '--dataPath' + str(args.dataPath)
    # args.id += '--noValidation' + str(args.noValidation)
    
    main(args)


## Run first comment to test code is exectued without errors
## python train.py --noTraining 60 --noValidation 10 --noTesting 10 --learningRate 0.5 --decayRate 0.1 --noTrPerClass 6 --noValPerClass 1 --noTsPerClass 1 --batchSize 16 --epochs 9

## After above run the the below line with required parameters 
## python train.py --learningRate 0.5 --decayRate 0.5 --batchSize 128
