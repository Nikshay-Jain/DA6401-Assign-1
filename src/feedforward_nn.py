# Question 2:

import numpy as np
import wandb
from supporting_funcs import *
from wandb_setup import setup_wandb

# Initialize Weights & Biases
wandb_run = setup_wandb(run_name="Q2-feedforward-nn")
config = wandb.config
config.update({
    "epochs": 10,
    "learning_rate": 0.01,
    "batch_size": 64,
    "hidden_layers": [128, 64]
})

# Activation functions
def relu(x):
    return np.maximum(0, x)

def diff_relu(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Initialize params for nn
def init_nn(input_size, hidden_layers, output_size):
    layers = [input_size] + hidden_layers + [output_size]
    weights = [np.random.randn(layers[i], layers[i+1]) * 0.01 for i in range(len(layers) - 1)]
    biases = [np.zeros((1, layers[i+1])) for i in range(len(layers) - 1)]
    return {"weights": weights, "biases": biases}

# Forward propagation
def forward_pass(x, params):
    a = [x.reshape(x.shape[0], -1) / 255.0]    # Flatten and normalize input
    z = []
    for w, b in zip(params["weights"][:-1], params["biases"][:-1]):
        z.append(np.dot(a[-1], w) + b)
        a.append(relu(z[-1]))
    z.append(np.dot(a[-1], params["weights"][-1]) + params["biases"][-1])
    a.append(softmax(z[-1]))
    return a, z

# Backward propagation
def backward_pass(y_true, a, z, params, eta):
    m = y_true.shape[0]
    y_one_hot = np.eye(10)[y_true]            # Convert to one-hot encoding
    dz = a[-1] - y_one_hot                    # Softmax derivative
    
    for i in range(len(params["weights"]) - 1, -1, -1):
        dw = np.dot(a[i].T, dz) / m
        db = np.sum(dz, axis=0, keepdims=True) / m
        
        if i > 0:
            dz = np.dot(dz, params["weights"][i].T) * diff_relu(z[i-1])
        
        params["weights"][i] -= eta * dw
        params["biases"][i] -= eta * db

# Training function
def train_nn(x_train, y_train, params, epochs, eta, batch_size):
    for epoch in range(epochs):
        indices = np.random.permutation(len(x_train))
        x_train, y_train = x_train[indices], y_train[indices]
        
        for i in range(0, len(x_train), batch_size):
            x_batch, y_batch = x_train[i:i+batch_size], y_train[i:i+batch_size]
            a, z = forward_pass(x_batch, params)
            backward_pass(y_batch, a, z, params, eta)
        
        preds = predict(x_train, params)
        acc = np.mean(preds == y_train)
        wandb.log({"epoch": epoch+1, "train_accuracy": acc})
        print(f"Epoch {epoch+1}/{epochs}, Accuracy: {acc:.4f}")

# Prediction function
def predict(x, params):
    a, _ = forward_pass(x, params)
    return np.argmax(a[-1], axis=1)

# Load data
(x_train, y_train), (x_test, y_test) = load_dataset()

# Initialize params and train nn
params = init_nn(input_size=28*28, hidden_layers=[128, 64], output_size=10)
train_nn(x_train, y_train, params, epochs=config.epochs, eta=config.learning_rate, batch_size=config.batch_size)

# Evaluate on test data
y_pred = predict(x_test, params)
test_acc = np.mean(y_pred == y_test)
wandb.log({"test_accuracy": test_acc})
print(f"Test Accuracy: {test_acc:.4f}")



class optimizers_legacy:
    def __init__(self,X_size,Y_size,hidden_layer_sizes,hidden_layer_activations,hidden_layer_initializations,
                 loss='cross_entropy',optimizer='adam',lamdba=0,batch_size=1,epochs=10,eta=1e-3,ES=True,log=True):

        self.batch_size=batch_size
        self.epochs=epochs
        
        self.const_hidden_layer_size=const_hidden_layer_size
        self.const_hidden_layer_activation=const_hidden_layer_activation
        self.const_hidden_layer_initializations=const_hidden_layer_initializations

        self.lamdba=lamdba
        
        self.model=Model(X_size,Y_size,hidden_layer_sizes,hidden_layer_activations,
                         hidden_layer_initializations,loss,lamdba_m=lamdba/self.batch_size,batch_size=self.batch_size)
        
        self.learning_rate=eta
        self.optimizer=optimizer
        
        if self.optimizer=='sgd':
            self.batch_size=1
            
        self.train_loss=[]
        self.train_acc=[]
        
        self.val_loss=[]
        self.val_acc=[]
        
        self.log=log
        self.ES=ES
        if self.ES:
            self.ES_best_val_loss=1e30
            self.ES_paitence=5
            self.ES_model=None
            self.ES_epoch=-1
        

    def wandb_logger(self,t,valtrue=True):
        
        if valtrue:
        
            wandb.log({'train_loss':self.train_loss[-1],'val_loss':self.val_loss[-1],\
                       'train_acc':self.train_acc[-1],'val_acc':self.val_acc[-1],'epoch':t})
            
        else:
            wandb.log({'train_loss':self.train_loss[-1],\
                       'train_acc':self.train_acc[-1],'epoch':t})

    def iterate(self,updator,X,Y,testdat):
        reminder=X.shape[1]%self.batch_size #uneven batch size
        
        for t in tqdm(range(self.epochs)):
            for i in range(0,np.shape(X)[1]-self.batch_size,self.batch_size):
                x=X[:,i:i+self.batch_size]
                y=Y[:,i:i+self.batch_size]
                y_pred=self.model.forward(x)
    
    
                self.model.backward(x,y,y_pred)
                updator(t)

            if reminder:

                x=np.hstack((X[:,i+self.batch_size:],X[:,:reminder]))
                y=np.hstack((Y[:,i+self.batch_size:],Y[:,:reminder]))
                y_pred=self.model.forward(x)
                self.model.backward(x,y,y_pred)
                updator(t)
            
            if testdat:
                Xval,Yval=testdat
                self.loss_calc(X,Y,Xval,Yval)
                valtrue=True
            else:
                self.loss_calc_fit(X,Y)
                valtrue=False
                self.ES=False    

            if self.ES:
                if self.ES_best_val_loss>self.val_loss[-1]:
                    self.ES_best_val_loss=self.val_loss[-1]
                    self.ES_model=copy.deepcopy(self.model)
                    self.patience=5
                    self.ES_epoch=t
                    if self.log:
                        self.wandb_logger(t,valtrue)

                else:
                    self.patience-=1
                    if not self.patience:
                        print('Early stopping at epoch: ',t, "reverting to epoch ", self.ES_epoch)
                        self.loss_calc(X,Y,Xval,Yval)
                        if self.log:
                            self.wandb_logger(t,valtrue)
                            
                        self.model=self.ES_model
                        self.loss_calc(X,Y,Xval,Yval)
                        if self.log:
                            self.wandb_logger(t+1,valtrue)

                        return
            elif self.log:
                        self.wandb_logger(t,valtrue)

        if self.ES: #return best model if epochs are over       
            self.model=self.ES_model                
                   
            
    def accuracy_check(self,Y,Ypred):
        return np.sum(np.argmax(Ypred,axis=0)==np.argmax(Y,axis=0))/Y.shape[1]        
    
    def loss_calc(self,X,Y,Xval,Yval):
            regularization=1/2*self.model.lamdba_m*np.sum([np.sum(layer.W**2) for layer in self.model.layers])
            
            Ypred=self.model.predict(X)
            Yvalpred=self.model.predict(Xval)
            
            self.train_loss.append((self.model.loss(Y,Ypred)+regularization)/X.shape[1])
            self.val_loss.append(self.model.loss(Yval,Yvalpred)/Xval.shape[1])
            self.train_acc.append(self.accuracy_check(Y,Ypred))
            self.val_acc.append(self.accuracy_check(Yval,Yvalpred))
            
    def loss_calc_fit(self,X,Y):
        regularization=1/2*self.model.lamdba_m*np.sum([np.sum(layer.W**2) for layer in self.model.layers])
        Ypred=self.model.predict(X)
        self.train_loss.append((self.model.loss(Y,Ypred)+regularization)/X.shape[1])
        self.train_acc.append(self.accuracy_check(Y,Ypred))

    def batch_gradient_descent(self,traindat,testdat):
        X,Y=traindat
        def update_batch(_):
            for layer in self.model.layers:
                layer.W=layer.W-self.learning_rate*layer.d_W
                layer.b=layer.b-self.learning_rate*layer.d_b
        updator=update_batch
        self.iterate(updator,X,Y,testdat)

    def momentum(self,traindat,testdat,beta=0.9):
        X,Y=traindat
        u_W=[np.zeros(np.shape(layer.d_W)) for layer in self.model.layers]
        u_b=[np.zeros(np.shape(layer.d_b)) for layer in self.model.layers]
        
        def update_mom(_):
            for i in range(len(self.model.layers)):
                layer=self.model.layers[i]
                u_W[i]=beta*u_W[i]+layer.d_W
                u_b[i]=beta*u_b[i]+layer.d_b
                layer.W=layer.W-self.learning_rate*u_W[i]
                layer.b=layer.b-self.learning_rate*u_b[i]
        
        updator=update_mom
        self.iterate(updator,X,Y,testdat)
            

    def rmsprop(self,traindat,testdat,beta=0.9,epsilon=1e-10):
        X,Y=traindat
        v_W=[np.zeros(np.shape(layer.d_W)) for layer in self.model.layers]
        v_b=[np.zeros(np.shape(layer.d_b)) for layer in self.model.layers]
        
        def update_rms(_):
                for i in range(len(self.model.layers)):                 
                    layer=self.model.layers[i]
                    v_W[i]=beta*v_W[i]+(1-beta)*layer.d_W**2
                    v_b[i]=beta*v_b[i]+(1-beta)*layer.d_b**2
                    layer.W=layer.W-(self.learning_rate/np.sqrt(v_W[i]+epsilon))*layer.d_W
                    layer.b=layer.b-(self.learning_rate/np.sqrt(v_b[i]+epsilon))*layer.d_b

        updator=update_rms
        self.iterate(updator,X,Y,testdat)
            
    def Adam(self,traindat,testdat,beta1=0.9, beta2=0.999,epsilon=1e-10):
        X,Y=traindat

        m_W=[np.zeros(np.shape(layer.d_W)) for layer in self.model.layers]
        v_W=[np.zeros(np.shape(layer.d_W)) for layer in self.model.layers]
        m_b=[np.zeros(np.shape(layer.d_b)) for layer in self.model.layers]
        v_b=[np.zeros(np.shape(layer.d_b)) for layer in self.model.layers]
        
        def update_adam(t):
            for i in range(len(self.model.layers)):
                layer=self.model.layers[i]
                #updating momentum, velocity
                m_W[i]=beta1*m_W[i]+(1-beta1)*layer.d_W
                m_b[i]=beta1*m_b[i]+(1-beta1)*layer.d_b

                v_W[i]=beta2*v_W[i]+(1-beta2)*layer.d_W**2
                v_b[i]=beta2*v_b[i]+(1-beta2)*layer.d_b**2

                m_W_hat=m_W[i]/(1-np.power(beta1,t+1))
                m_b_hat=m_b[i]/(1-np.power(beta1,t+1))
                v_W_hat=v_W[i]/(1-np.power(beta2,t+1))
                v_b_hat=v_b[i]/(1-np.power(beta2,t+1))

                layer.W=layer.W-(self.learning_rate*m_W_hat)/(np.sqrt(v_W_hat)+epsilon)
                layer.b=layer.b-(self.learning_rate*m_b_hat)/(np.sqrt(v_b_hat)+epsilon)

        updator=update_adam
        self.iterate(updator,X,Y,testdat)
    
    def NAG(self,traindat,testdat,beta=0.9):
        X,Y=traindat
        m_W=[np.zeros(np.shape(layer.d_W)) for layer in self.model.layers]
        m_b=[np.zeros(np.shape(layer.d_b)) for layer in self.model.layers]
        def update_nag(_):
            for i in range(len(self.model.layers)):
                layer=self.model.layers[i]
                m_W[i]=beta*m_W[i]+self.learning_rate*layer.d_W
                m_b[i]=beta*m_b[i]+self.learning_rate*layer.d_b


                layer.W=layer.W-(beta*m_W[i]+self.learning_rate*layer.d_W[i])
                layer.b=layer.b-(beta*m_b[i]+self.learning_rate*layer.d_b[i])
            
        updator=update_nag
        self.iterate(updator,X,Y,testdat)
    
    def NAdam(self,traindat,testdat,beta1=0.9, beta2=0.999,epsilon=1e-10):
        X,Y=traindat
        m_W=[np.zeros(np.shape(layer.d_W)) for layer in self.model.layers]
        v_W=[np.zeros(np.shape(layer.d_W)) for layer in self.model.layers]
        m_b=[np.zeros(np.shape(layer.d_b)) for layer in self.model.layers]
        v_b=[np.zeros(np.shape(layer.d_b)) for layer in self.model.layers]
        
        def update_nadam(t):
            for i in range(len(self.model.layers)):
                layer=self.model.layers[i]
                #updating momentum, velocity
                m_W[i]=beta1*m_W[i]+(1-beta1)*layer.d_W
                m_b[i]=beta1*m_b[i]+(1-beta1)*layer.d_b

                v_W[i]=beta2*v_W[i]+(1-beta2)*layer.d_W**2
                v_b[i]=beta2*v_b[i]+(1-beta2)*layer.d_b**2

                m_W_hat=m_W[i]/(1-np.power(beta1,t+1))
                m_b_hat=m_b[i]/(1-np.power(beta1,t+1))
                v_W_hat=v_W[i]/(1-np.power(beta2,t+1))
                v_b_hat=v_b[i]/(1-np.power(beta2,t+1))



                layer.W=layer.W-(self.learning_rate/(np.sqrt(v_W_hat)+epsilon))*\
                (beta1*m_W_hat+((1-beta1)/(1-np.power(beta1,t+1)))*layer.d_W)
                layer.b=layer.b-(self.learning_rate/(np.sqrt(v_b_hat)+epsilon))*\
                (beta1*m_b_hat+((1-beta1)/(1-np.power(beta1,t+1)))*layer.d_b)            

        updator=update_nadam
        self.iterate(updator,X,Y,testdat)

    def run(self,traindat,testdat=None,momentum=0.9,beta=0.9,beta1=0.9, beta2=0.999,epsilon=1e-10):
        
        if self.optimizer=="batch":
            self.batch_gradient_descent(traindat,testdat)
            
        elif self.optimizer=="sgd":
            assert(self.batch_size==1), "Batch size should be 1 for stochastic gradient descent"
            self.batch_gradient_descent(traindat,testdat)
            
        elif self.optimizer=="momentum":
            self.momentum(traindat,testdat,beta)
            
            
        elif self.optimizer=="nesterov":
            self.NAG(traindat,testdat,beta)
            
        elif self.optimizer=="rmsprop":
            self.rmsprop(traindat,testdat,beta=0.9,epsilon=1e-10)
            
        elif self.optimizer=="adam":
            self.Adam(traindat,testdat,beta1=0.9, beta2=0.999,epsilon=1e-10)
            
        elif self.optimizer=="nadam":
            self.NAdam(traindat,testdat,beta1=0.9, beta2=0.999,epsilon=1e-10)
            
        else:
            print("Invalid optimizer name "+ self.optimizer)
            return(0)