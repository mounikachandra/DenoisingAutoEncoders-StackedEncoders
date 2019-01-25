import numpy as np
import matplotlib.pyplot as plt
import pdb


class Model():

    def softmax_cross_entropy_loss(self, Z, Y=np.array([])):
        cache={}
        #print(Y.shape) 1*m 
        n,m = Y.shape
        #print(Y)
        mask = range(m)
        A = np.exp(Z-np.max(Z))/(np.sum(np.exp(Z-np.max(Z)),axis=0)).reshape(1,m)  ## total is n,m
        cache["A"]=A
        loss = -np.log(A[Y.astype(int),mask])
        loss= np.sum(loss)/m 
        return A, cache, loss

    def softmax_cross_entropy_loss_der(self, Y, cache):
        n,m = Y.shape
        mask = range(m)
        dZ= cache["A"]
        dZ[Y.astype(int),mask]-=1
        dZ/=m
        return dZ

    def initialize_2layer_weights(self,n_in, n_h, n_fin):

        parameters = {}
        parameters["W1"] = (np.random.randn(n_in,n_h)*(np.sqrt(2.0/(n_in*n_h)))).T
        parameters["b1"] = np.zeros(n_h).reshape((n_h,1))
        parameters["W2"] = (np.random.randn(n_h,n_fin)*(np.sqrt(2.0/(n_h*n_fin)))).T
        parameters["b2"] = np.zeros(n_fin).reshape((n_fin,1))

        return parameters


    def initialize_multilayer_weights(self, net_dims):
        numLayers = len(net_dims)
        parameters = {}
        for l in range(numLayers-1):
            parameters["W"+str(l+1)] = (np.random.randn(net_dims[l], net_dims[l+1])*(np.sqrt(2.0/(net_dims[l] * net_dims[l+1])))).T
            #print(parameters["W"+str(l+1)].shape)
            parameters["b"+str(l+1)] = np.zeros(net_dims[l+1]).reshape((net_dims[l+1],1))
        return parameters

    def tanh(self,Z):
        
        A = np.tanh(Z)
        cache = {}
        cache["Z"] = Z
        return A, cache

    def tanh_der(self,dA, cache):
        
        Z = cache["Z"]
        dZ = dA * (1-np.tanh(Z)**2)
        return dZ

    def sigmoid(self,Z):
        
        A = 1/(1+np.exp(-Z))
        cache = {}
        cache["Z"] = Z
        return A, cache

    def sigmoid_der(self,dA, cache):
        

        Z = cache["Z"]
        A_, cache_= self.sigmoid(Z)
        dZ= dA * A_ *(1-A_)

        return dZ

    def linear(self,Z):
        A = Z
        cache = {}
        return A, cache

    def linear_der(self, dA, cache):
        dZ = np.array(dA, copy=True)
        return dZ

    def relu(self,Z):
        A = np.maximum(0,Z)
        cache = {}
        cache["Z"] = Z
        return A, cache

    def relu_der(self,dA, cache):
        dZ = np.array(dA, copy=True)
        Z = cache["Z"]
        dZ[Z<0] = 0
        return dZ

    def linear_forward(self,A, W, b):
        Z=W.dot(A)+b
        cache = {}
        cache["A"] = A
        return Z, cache

    def layer_forward(self,A_prev, W, b, activation):
        Z, lin_cache = self.linear_forward(A_prev, W, b)
        if activation == "sigmoid":
            A, act_cache = self.sigmoid(Z)
        elif activation == "tanh":
            A, act_cache = self.tanh(Z)
        elif activation == "relu":
            A, act_cache = self.relu(Z)
        elif activation == "linear":
            A, act_cache = self.linear(Z)
        
        cache = {}
        cache["lin_cache"] = lin_cache
        cache["act_cache"] = act_cache

        return A, cache

    def multi_layer_forward(self, X, parameters):
        L = len(parameters)//2
        A = X
        caches = []
        for l in range(1,L):  # since there is no W0 and b0
            A, cache = self.layer_forward(A, parameters["W"+str(l)], parameters["b"+str(l)], "sigmoid")
            caches.append(cache)

        AL, cache = self.layer_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], "linear")
        caches.append(cache)
        return AL, caches

    def crossEntropy(self,A2,X):

        obs1,obs2 = X.shape
        pred = A2

        cost= -X*np.log(pred) - (1-X)*np.log(1-pred)
        cost= cost.sum()/(obs1*obs2)

        return cost

    def MSE(self,A2,X):
        # obs = X.shape[1]
        # cost= (A2-X)*(A2-X)
        # cost= cost.sum()/obs
        cost= 0.5*np.mean((A2-X)**2)

        return cost

    def linear_backward(self,dZ, cache, W, b):
        A =cache["A"]
        dA_prev = np.dot(W.T,dZ)
        dW = np.dot(dZ,A.T)
        db = np.sum(dZ,axis=1,keepdims= True)
        return dA_prev, dW, db

    def layer_backward(self,dA, cache, W, b, activation):
        lin_cache = cache["lin_cache"]
        act_cache = cache["act_cache"]

        if activation == "sigmoid":
            dZ = self.sigmoid_der(dA, act_cache)
        elif activation == "tanh":
            dZ = self.tanh_der(dA, act_cache)
        elif activation == "relu":
            dZ = self.relu_der(dA, act_cache)
        elif activation == "linear":
            dZ = self.linear_der(dA, act_cache)
        dA_prev, dW, db = self.linear_backward(dZ, lin_cache, W, b)
        return dA_prev, dW, db

    def multi_layer_backward(self,dAL, caches, parameters):
        L = len(caches)  # with one hidden layer, L = 2
        gradients = {}
        dA = dAL
        activation = "linear"
        for l in reversed(range(1,L+1)):
            dA, gradients["dW"+str(l)], gradients["db"+str(l)] = \
                        self.layer_backward(dA, caches[l-1], \
                        parameters["W"+str(l)],parameters["b"+str(l)],\
                        activation)
            activation = "sigmoid"
        return gradients

    def classify(self, X, parameters):
    
        ALast, cache = self.multi_layer_forward(X.T, parameters)
        Ypred = np.argmax(ALast, axis=0)

        return Ypred

    def update_parameters(self,parameters, gradients, epoch, learning_rate, decay_rate=0.0):
        #alpha = learning_rate*(1/(1+decay_rate*epoch))
        L = len(parameters)//2
        ### CODE HERE
        for l in range(L-1):
            parameters["W"+str(l+1)]+=-learning_rate*gradients["dW"+str(l+1)]
            parameters["b"+str(l+1)]+=-learning_rate*gradients["db"+str(l+1)]
        return parameters, learning_rate

    def accuracy(self,predicted_labels, actual_labels):
        diff = predicted_labels - actual_labels
        return (1.0 - (float(np.count_nonzero(diff)) / len(diff)))*100

class Noise():
    def SaltAndPepper(self, image, rate=0.3):
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
          
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1

        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out
        
    def GaussianNoise(self, X, sd=0.5):
        X += np.random.normal(0, sd, X.shape)
        return X
        
    def MaskingNoise(self, X, rate=0.5):
        mask = (np.random.uniform(0,1, X.shape)<rate).astype("i4")
        X = mask*X
        return X

class Optimizers():
    def __init__(self, args):
        self.learningRate = args.learningRate
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.epsilon = args.epsilon
    def adam(self, x, dx, config = None):
        if config == None:
            config = {}
        config['m'] = np.zeros_like(x)
        config['v'] = np.zeros_like(x)
        config['t'] = 0 
  
        next_x = x
        config["t"] += 1.0
        config["m"] = (self.beta1 * config["m"]) + ((1 - self.beta1) * dx)
        config["v"] = (self.beta2 * config["v"]) + ((1 - self.beta2)*(dx**2))
        mt = config["m"] / (1 - self.beta1**config["t"])
        vt = config["v"] / (1 - self.beta2**config["t"])
        next_x -= self.learningRate * mt / (np.sqrt(vt) + self.epsilon)        
   
        return next_x, config