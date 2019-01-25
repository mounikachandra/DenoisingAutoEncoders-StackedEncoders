import numpy as np
import matplotlib.pyplot as plt
import pdb

np.random.seed(1)

class Model():


    def initialize_2layer_weights(self,n_in, n_h, n_fin):

        parameters = {}
        parameters["W1"] = (np.random.randn(n_in,n_h)*(np.sqrt(2.0/(n_in*n_h)))).T
        parameters["b1"] = np.zeros(n_h).reshape((n_h,1))
        parameters["W2"] = (np.random.randn(n_h,n_fin)*(np.sqrt(2.0/(n_h*n_fin)))).T
        parameters["b2"] = np.zeros(n_fin).reshape((n_fin,1))

        return parameters

    def tanh(self,Z):
        '''
        computes tanh activation of Z

        Inputs: 
            Z is a numpy.ndarray (n, m)

        Returns: 
            A is activation. numpy.ndarray (n, m)
            cache is a dictionary with {"Z", Z}
        '''
        A = self.np.tanh(Z)
        cache = {}
        cache["Z"] = Z
        return A, cache

    def tanh_der(self,dA, cache):
        '''
        computes derivative of tanh activation

        Inputs: 
            dA is the derivative from subsequent layer. numpy.ndarray (n, m)
            cache is a dictionary with {"Z", Z}, where Z was the input 
            to the activation layer during forward propagation

        Returns: 
            dZ is the derivative. numpy.ndarray (n,m)
        '''
        ### CODE HERE
        Z = cache["Z"]
        dZ = dA * (1-self.np.tanh(Z)**2)
        return dZ

    def sigmoid(self,Z):
        '''
        computes sigmoid activation of Z

        Inputs: 
            Z is a numpy.ndarray (n, m)

        Returns: 
            A is activation. numpy.ndarray (n, m)
            cache is a dictionary with {"Z", Z}
        '''
        A = 1/(1+np.exp(-Z))
        cache = {}
        cache["Z"] = Z
        return A, cache

    def sigmoid_der(self,dA, cache):
        '''
        computes derivative of sigmoid activation

        Inputs: 
            dA is the derivative from subsequent layer. numpy.ndarray (n, m)
            cache is a dictionary with {"Z", Z}, where Z was the input 
            to the activation layer during forward propagation

        Returns: 
            dZ is the derivative. numpy.ndarray (n,m)
        '''
        ### CODE HERE

        Z = cache["Z"]
        A_, cache_= self.sigmoid(Z)
        dZ= dA * A_ *(1-A_)

        return dZ

    def relu(self,Z):
        '''
        computes relu activation of Z

        Inputs: 
            Z is a numpy.ndarray (n, m)

        Returns: 
            A is activation. numpy.ndarray (n, m)
            cache is a dictionary with {"Z", Z}
        '''

        A = np.maximum(0,Z)
        cache = {}
        cache["Z"] = Z
        return A, cache

    def relu_der(self,dA, cache):
        '''
        computes derivative of relu activation

        Inputs: 
            dA is the derivative from subsequent layer. numpy.ndarray (n, m)
            cache is a dictionary with {"Z", Z}, where Z was the input 
            to the activation layer during forward propagation

        Returns: 
            dZ is the derivative. numpy.ndarray (n,m)
        '''
        dZ = np.array(dA, copy=True)
        Z = cache["Z"]
        dZ[Z<0] = 0
        return dZ

    def linear_forward(self,A, W, b):
        '''
        Input A propagates through the layer 
        Z = WA + b is the output of this layer. 

        Inputs: 
            A - numpy.ndarray (n,m) the input to the layer
            W - numpy.ndarray (n_out, n) the weights of the layer
            b - numpy.ndarray (n_out, 1) the bias of the layer

        Returns:
            Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
            cache - a dictionary containing the inputs A, W and b
            to be used for derivative
        '''
        ### CODE HERE
        Z=W.dot(A)+b
        cache = {}
        cache["A"] = A
        return Z, cache

    def layer_forward(self,A_prev, W, b, activation):
        '''
        Input A_prev propagates through the layer and the activation

        Inputs: 
            A_prev - numpy.ndarray (n,m) the input to the layer
            W - numpy.ndarray (n_out, n) the weights of the layer
            b - numpy.ndarray (n_out, 1) the bias of the layer
            activation - is the string that specifies the activation function

        Returns:
            A = g(Z), where Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
            g is the activation function
            cache - a dictionary containing the cache from the linear and the nonlinear propagation
            to be used for derivative
        '''
        Z, lin_cache = self.linear_forward(A_prev, W, b)
        if activation == "sigmoid":
            A, act_cache = self.sigmoid(Z)
        elif activation == "tanh":
            A, act_cache = self.tanh(Z)
        elif activation == "relu":
            A, act_cache = self.relu(Z)
        cache = {}
        cache["lin_cache"] = lin_cache
        cache["act_cache"] = act_cache

        return A, cache

    def crossEntropy(self,A2,X):

        obs = X.shape[1]
        pred = A2

        cost= -X*np.log(pred) - (1-X)*np.log(1-pred)
        cost= cost.sum()/obs

        return cost

    def MSE(self,A2,X):

        # obs = X.shape[1]

        # cost= (1/2)*(A2-X)*(A2-X)
        # cost= cost.sum()/obs
        cost= np.mean((X - A2)**2)

        return cost

    def linear_backward(self,dZ, cache, W, b):
        '''
        Backward propagation through the linear layer

        Inputs:
            dZ - numpy.ndarray (n,m) derivative dL/dz 
            cache - a dictionary containing the inputs A
                where Z = WA + b,    
                Z is (n,m); W is (n,p); A is (p,m); b is (n,1)
            W - numpy.ndarray (n,p)  
            b - numpy.ndarray (n, 1)

        Returns:
            dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
            dW - numpy.ndarray (n,p) the gradient of W 
            db - numpy.ndarray (n, 1) the gradient of b
        '''
        # CODE HERE
        A =cache["A"]
        dA_prev = np.dot(W.T,dZ)
        dW = np.dot(dZ,A.T)
        db = np.sum(dZ,axis=1,keepdims= True)
        return dA_prev, dW, db

    def layer_backward(self,dA, cache, W, b, activation):
        '''
        Backward propagation through the activation and linear layer

        Inputs:
            dA - numpy.ndarray (n,m) the derivative to the previous layer
            cache - dictionary containing the linear_cache and the activation_cache
            W - numpy.ndarray (n,p)  
            b - numpy.ndarray (n, 1)
        
        Returns:
            dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
            dW - numpy.ndarray (n,p) the gradient of W 
            db - numpy.ndarray (n, 1) the gradient of b
        '''
        lin_cache = cache["lin_cache"]
        act_cache = cache["act_cache"]

        if activation == "sigmoid":
            dZ = self.sigmoid_der(dA, act_cache)
        elif activation == "tanh":
            dZ = self.tanh_der(dA, act_cache)
        elif activation == "relu":
            dZ = self.relu_der(dA, act_cache)
        dA_prev, dW, db = self.linear_backward(dZ, lin_cache, W, b)
        return dA_prev, dW, db
    
class Noise():
    def SaltAndPepper(self, image, rate=0.3):
        row,col = image.shape
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
	    mean = 0
	    var = sd
	    sigma = var**0.5
	    gauss = np.random.normal(mean,sigma,(X.shape))
	    gauss = gauss.reshape(X.shape)
	    noisy = X + gauss
	    return noisy
	        
    def MaskingNoise(self, X, rate=0.5):
        mask = (np.random.uniform(0,1, X.shape)<rate).astype("i4")
        X = mask*X
        return X