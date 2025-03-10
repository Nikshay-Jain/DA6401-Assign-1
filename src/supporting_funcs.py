# Contains function for activation functions, loss functions and their derivatives

import numpy as np

def load_dataset():
    with np.load("fashion-mnist.npz") as data:
        x_train, y_train = data["x_train"], data["y_train"]
        x_test, y_test = data["x_test"], data["y_test"]
    return (x_train, y_train), (x_test, y_test)

def one_hot(inp): 
    outarray = np.zeros((inp.size, 10))
    outarray[np.arange(inp.size), inp] = 1
    return outarray

def Preprocess(X,y):
    #checks if same dim in X and y
    assert(X.shape[0]==y.shape[0]),"Inputs must contain same number of examples, stored in rows"
    
    X_processed = (np.reshape(X,(X.shape[0],784))/255.0).T       #reshaping & normalizing
    y_processed = one_hot(y).T                                   #one hotting y
    return np.array(X_processed),y_processed

def train_val_split(X, y, splits=0.1):
    i = int((1 - splits) * X.shape[0])         
    index = np.random.permutation(X.shape[0])

    Xtrain, Xval = np.split(np.take(X,index,axis=0), [i])
    ytrain, yval = np.split(np.take(y,index), [i])
    return Xtrain, Xval, ytrain, yval

def get_activation(activation):
    def sigmoid(x):
        return np.where(x >= 0, 
                        1 / (1 + np.exp(-x)), 
                        np.exp(x) / (1 + np.exp(x)))
    def softmax(x):
        z = x-np.max(x,axis=0)
        return np.exp(z)/np.sum(np.exp(z),axis=0)
    
    def relu(x):
        return np.maximum(0, x)
    
    if activation=='sigmoid':
        return sigmoid
    elif activation=='softmax':
        return softmax
    elif activation== 'tanh':
        return np.tanh
    elif activation== 'relu':
        return relu

def diff_activation(activation):
    def sigmoid_d(x):
        sig= np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
        return sig*(1-sig)
    
    def softmax_d(x):
        z=x-np.max(x,axis=0)
        soft=np.exp(z)/np.sum(np.exp(z),axis=0)
        return soft*(1-soft)
    
    def tanh_d(x):
        return 1-np.tanh(x)**2
    
    def relu_d(x):
        return np.where(x >= 0, 
                            1, 
                            0)
    
    if activation=='sigmoid':
        return sigmoid_d
    elif activation=='softmax':
        return softmax_d
    elif activation=='tanh':
        return tanh_d
    elif activation=='relu':
        return relu_d
    assert(activation=='relu'or activation=='tanh'or activation=='sigmoid' or activation=='softmax'),\
    'Must be \'relu\'or \'tanh\' or \'sigmoid\' or \'softmax\' '

def get_loss(loss='cross_entropy'):
    safety=1e-30

    def crossentropy(P,Q):
        assert(P.shape==Q.shape), "Inputs must be of same shape"
        return np.sum([-np.dot(P[:,i],np.log2(Q[:,i]+safety)) for i in range(P.shape[1])])
    
    def SE(P,Q):
        assert(P.shape==Q.shape), "Inputs must be of same shape"
        return np.sum(np.square(P-Q))
    
    if loss=="mean_squared_error":
        return SE
    
    return crossentropy

def get_loss_derivative(loss):
    def SE_d(y_in,y_pred_in):
        def indicator(i,j):
                if i==j:
                    return 1
                return 0

        assert(y_in.shape[0]==y_pred_in.shape[0]),"Inputs must contain same number of examples"

        y=y_in.ravel()
        y_pred=y_pred_in.ravel()

        return np.array([
            [2*np.sum([(y_pred[i]-y[i])*y[i]*(indicator(i,j) - y_pred[j]) for i in range(y.shape[0])])]
            for j in range(len(y))
        ])    
   
    def crossentropy_d(y,y_pred):
        return -(y-y_pred)
    
    if loss=="cross_entropy":
        return crossentropy_d
    return SE_d