import numpy as np

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