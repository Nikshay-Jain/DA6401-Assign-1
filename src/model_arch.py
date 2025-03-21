import numpy as np
from tqdm import tqdm
import wandb, copy, datetime
from supporting_funcs import *
import matplotlib.pyplot as plt

'''
Structure Block:-
Includes:-
-layer class
-Model class
-Opimizer class
        -- Has model object, implements ealry stopping and returns the best model class trough Optimizer.model
        -- uses a single layer size, activation, and initialization for each layer in  Optimizer.model
'''

# Layer class
class layer:
    '''
    Layer Class:-
    Each layer can be initialized to different sizes, activations, initializations (He/Xavier/Random)
    
    Args:
        input_size,output_size,
        activation: activation function, (default:'sigmoid'),
        batch_size: fixed size of batches, used for broadcasting, (default:2)
        type_: initialization type, (default:'random')
    '''
    
    def __init__(self, input_size, output_size, activation='sigmoid', batch_size=2, type_='random'):
        type_.lower()
        assert(type_=='random'or type_=='xavier'or type_=='glorot' or type_=='he' or type_=='kaiming' ),\
        'Must be \'random\'or \'xavier\' or \'glorot\' or \'He\' or \'kaiming\' '
        
        if type_=='random':
            scale = 0.01
            self.W = np.random.randn(output_size,input_size)*scale     #size ixj
            
        elif type_=='xavier' or type_=='glorot':
            # Xavier Uniform
            r = np.sqrt(6/(input_size+output_size))
            self.W = np.random.uniform(-r,r,(output_size,input_size))

        else:
            self.W= np.random.randn(output_size,input_size)*np.sqrt(2/input_size)

        self.b=np.zeros((output_size,1))
        
        self.a=np.zeros((output_size,batch_size))
        self.h=np.zeros((output_size,batch_size))
        self.g=get_activation(activation)
        
        self.d_a=np.zeros((output_size,batch_size))
        self.d_h=np.zeros((output_size,batch_size))
        self.d_W=np.zeros((output_size,input_size))
        self.d_b=np.zeros((output_size,1))
        self.d_g=diff_activation(activation)
        
    def forward(self, inputs):
        '''forward pass in layer'''
        self.a = self.b+np.matmul(self.W,inputs)
        self.h = self.g(self.a)
        return self.h
        
    def hard_set(self,W,b):
        '''hardsets the weight. useful for debugging'''
        self.W = W
        self.b = b

# Model Class
class Model:
    '''
    Model Class:- it is a complete model.      
    Contains list of layers, loss metric, derivatives of loss mertic, regularization parameter lambda, batch size
    Args:
        X_size:number of features of X (int)
        Y_size:number of features of Y(int)
        hidden_layer_sizes: sizes of various hidden layers (list(ints))
        hidden_layer_activations: sizes of various hidden layers (list(ints))
        hidden_layer_initializations: sizes of various hidden layers (list(ints))
        loss: loss choice 'mean_squared_error'/'cross_entropy' (str)
        lamdba_m: regularization parameter
        batch_size: batch size
    '''   
        
    def __init__(self,X_size,Y_size,hidden_layer_sizes,hidden_layer_activations,hidden_layer_initializations,loss,lamdba_m,batch_size):
        self.input_size=X_size
        self.output_size=Y_size
        self.hidden_layer_sizes=hidden_layer_sizes
        self.layers=[]
        self.batch_size=batch_size
        
        #building layer class
        prev_size=self.input_size

        assert(len(hidden_layer_sizes)==len(hidden_layer_activations)==len(hidden_layer_initializations)),\
        'lengths of layer sizes, activations and initializations dont match'
        for size,activation,inits in zip(hidden_layer_sizes,hidden_layer_activations,hidden_layer_initializations):

            self.layers.append(layer(prev_size,size,activation,batch_size,inits))
            prev_size=size
        self.layers.append(layer(size,self.output_size,'softmax',batch_size,'xavier'))#output layer
        #layer class built
        
        self.loss=get_loss(loss)#without regularization term. We add in optimizer class
        self.loss_d=get_diff_loss(loss)
        self.lamdba_m=lamdba_m #we shall pass lambda/m to this, where m is patch size
        
    def forward(self,x):
        '''Model forward pass through layers'''
        output=x
        
        for layer in  self.layers:
            output=layer.forward(output)  
        return output
    
    def backward(self,x,y,y_pred):
        '''Model backward pass through layers'''
        
        # self.layers[-1].d_h is not needed as d_h is used to calculate d_a and self.layers[-1].h is softmax
        self.layers[-1].d_a=self.loss_d(y,y_pred)
        
        for idx in range(len(self.layers)-1,0,-1): #goes from L->2, for l=1 we do outside
            #compute gradient wrt parameters
            self.layers[idx].d_W=np.dot(self.layers[idx].d_a,np.transpose(self.layers[idx-1].h))+self.lamdba_m*self.layers[idx].W
            self.layers[idx].d_b=np.sum(self.layers[idx].d_a,axis=1,keepdims=True)
            
            #compute gradient wrt layer below -- will help in next layer iter
            self.layers[idx-1].d_h=np.matmul(np.transpose(self.layers[idx].W),self.layers[idx].d_a)
            
            #compute gradient -- element wise multiplivation, derivative of the activation function of layer idx-1
            self.layers[idx-1].d_a=self.layers[idx-1].d_h*self.layers[idx-1].d_g(self.layers[idx-1].a)
        assert(idx-1==0)
                        
        self.layers[0].d_W=np.dot(self.layers[0].d_a,np.transpose(x))+self.lamdba_m*self.layers[0].W
        self.layers[0].d_b=np.sum(self.layers[0].d_a,axis=1,keepdims=True)
        
    def predict(self,Xtest,probab=True):
        if probab:
            return self.forward(Xtest)
        return np.argmax(self.forward(Xtest),axis=0)
    
    def compute_accuracy(self, X_test, Y_test):
        '''Compute accuracy of the best model on test data'''
        Y_pred = self.forward(X_test)
        Y_pred_labels = np.argmax(Y_pred, axis=0)
        Y_true_labels = np.argmax(Y_test, axis=0)
        accuracy = np.mean(Y_pred_labels == Y_true_labels) * 100
        return accuracy

    def plot_confusion_matrix(self, X_test, Y_test):
        '''Compute and plot confusion matrix'''
        Y_pred = self.forward(X_test)
        Y_pred_labels = np.argmax(Y_pred, axis=0)
        Y_true_labels = np.argmax(Y_test, axis=0)

        num_classes = Y_test.shape[0]
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

        # Populate confusion matrix
        for true, pred in zip(Y_true_labels, Y_pred_labels):
            confusion_matrix[true, pred] += 1

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(confusion_matrix, cmap='Blues', interpolation='nearest')
        plt.colorbar()

        # Adding text labels
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, str(confusion_matrix[i, j]), ha='center', va='center', color='red')

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.xticks(np.arange(num_classes))
        plt.yticks(np.arange(num_classes))

        # Save figure
        plt.savefig("confusion_matrix.png")

        # Log confusion matrix image to wandb
        wandb.log({"Confusion Matrix": wandb.Image("confusion_matrix.png")})

    def cross_entropy_loss(self, y_true, y_pred):
        '''Compute cross entropy loss'''
        epsilon = 1e-10  # Avoid log(0) errors
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[1]

    def squared_error_loss(self, y_true, y_pred):
        '''Compute squared error loss'''
        return np.mean(np.sum((y_true - y_pred) ** 2, axis=0))

    def compare_losses(self, X_test, Y_test):
        '''Compute and compare both loss functions, then plot the results'''
        Y_pred = self.forward(X_test)
        ce_loss = self.cross_entropy_loss(Y_test, Y_pred)
        se_loss = self.squared_error_loss(Y_test, Y_pred)

        # Log losses in wandb
        wandb.log({
            "Cross Entropy Loss": ce_loss,
            "Squared Error Loss": se_loss
        })

        # Plot the comparison
        plt.figure(figsize=(8, 5))
        plt.bar(["Cross Entropy Loss", "Squared Error Loss"], [ce_loss, se_loss], color=['blue', 'orange'])
        plt.ylabel("Loss Value")
        plt.title("Comparison of Loss Functions")
        
        # Save and log the loss comparison image
        plt.savefig("loss_comparison.png")
        wandb.log({"Loss Comparison": wandb.Image("loss_comparison.png")})
        
class optimizers:
    ''' 
    Optimizer class
    Args:
    methods:
        __init__: initialzies the optimizer class
        wandb_logger: logs to wandb
        iterate:
    '''

    def __init__(self, X_size, Y_size, num_layers=3, const_hidden_layer_size=32, const_hidden_layer_activation='relu',
             const_hidden_layer_initializations='He', loss='cross_entropy', optimizer='adam', lamdba=0,
             batch_size=1, epochs=10, learning_rate=1e-3, ES=True, log=True, momentum=0.9, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-10):

        self.batch_size = batch_size
        self.epochs = epochs
        self.num_layers = num_layers
        self.const_hidden_layer_size = const_hidden_layer_size
        self.const_hidden_layer_activation = const_hidden_layer_activation
        self.const_hidden_layer_initializations = const_hidden_layer_initializations
        self.lamdba = lamdba
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.momentum = momentum
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Initialize hidden layer sizes, activations, and initializations
        hidden_layer_sizes = [const_hidden_layer_size] * num_layers
        hidden_layer_activations = [const_hidden_layer_activation] * num_layers
        hidden_layer_initializations = [const_hidden_layer_initializations] * num_layers

        self.model = Model(X_size, Y_size, hidden_layer_sizes, hidden_layer_activations,
                        hidden_layer_initializations, loss, lamdba_m=lamdba / self.batch_size, batch_size=self.batch_size)

        if self.optimizer == 'sgd':
            self.batch_size = 1

        self.train_loss = []
        self.train_accuracy = []
        self.val_loss = []
        self.val_accuracy = []
        self.log = log
        self.ES = ES

        if self.ES:
            self.ES_best_val_loss = 1e30
            self.ES_paitence = 5
            self.ES_model = None
            self.ES_epoch = -1
        
    def wandb_logger(self,t,valtrue=True):
        '''logs to wandb, 2 forms depending on validation data existance'''
        
        if valtrue:
            wandb.log({'train_loss':self.train_loss[-1],'val_loss':self.val_loss[-1],\
                       'train_accuracy':self.train_accuracy[-1],'val_accuracy':self.val_accuracy[-1],'epoch':t})
            
        else:
            wandb.log({'train_loss':self.train_loss[-1],\
                       'train_accuracy':self.train_accuracy[-1],'epoch':t})
        
    def iterate(self, updator, X, Y, testdat=None):
        '''
        Iterator method
        iterates over epochs, performs forward, backprop
        Calls updator, loss_calc methods 
        '''
        reminder = X.shape[1] % self.batch_size  # uneven batch size
        self.ES_best_val_loss = float('inf')

        for t in tqdm(range(self.epochs)):
            for i in range(0, np.shape(X)[1] - self.batch_size, self.batch_size):
                x = X[:, i:i + self.batch_size]
                y = Y[:, i:i + self.batch_size]
                y_pred = self.model.forward(x)

                self.model.backward(x, y, y_pred)
                updator(t)

            if reminder:
                x = np.hstack((X[:, i + self.batch_size:], X[:, :reminder]))
                y = np.hstack((Y[:, i + self.batch_size:], Y[:, :reminder]))
                y_pred = self.model.forward(x)
                self.model.backward(x, y, y_pred)
                updator(t)

            if testdat is not None:  # Check if testdat exists
                Xval, Yval = testdat
                self.loss_calc(X, Y, Xval, Yval)
                valtrue = True
            else:
                self.loss_calc_fit(X, Y)
                valtrue = False
                self.ES = False

            if self.ES:
                if self.ES_best_val_loss > self.val_loss[-1]:
                    self.ES_best_val_loss = self.val_loss[-1]
                    self.ES_model = copy.deepcopy(self.model)
                    self.patience = 5
                    self.ES_epoch = t
                    if self.log:
                        self.wandb_logger(t, valtrue)
                else:
                    self.patience -= 1
                    if not self.patience:
                        print('Early stopping at epoch: ', t, "reverting to epoch ", self.ES_epoch)
                        self.loss_calc(X, Y, Xval, Yval)
                        if self.log:
                            self.wandb_logger(t, valtrue)
                        self.model = self.ES_model
                        self.loss_calc(X, Y, Xval, Yval)
                        if self.log:
                            self.wandb_logger(t + 1, valtrue)
                        return
            elif self.log:
                self.wandb_logger(t, valtrue)

            # Log epoch-level metrics
            wandb.log({
                "epoch": t + 1,
                "train_loss": self.train_loss[-1],
                "train_accuracy": self.train_accuracy[-1],
                "val_loss": self.val_loss[-1] if valtrue else None,
                "val_accuracy": self.val_accuracy[-1] if valtrue else None,
                "test_accuracy": self.model.compute_accuracy(Xval, Yval) if valtrue else None,
                "learning_rate": self.learning_rate,
                "num_layers": self.num_layers,
                "hidden_size": self.const_hidden_layer_size,
                "activation": self.const_hidden_layer_activation,
                "optimizer": self.optimizer,
                "batch_size": self.batch_size,
                "weight_decay": self.lamdba,
                "momentum": self.momentum,
                "beta": self.beta,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "_timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  
            })

        if self.ES:  # return best model if epochs are over       
            self.model = self.ES_model

        return self.train_loss[-1], self.train_accuracy[-1]
    
    def accuracy_check(self,Y,Ypred):
        return np.sum(np.argmax(Ypred,axis=0)==np.argmax(Y,axis=0))/Y.shape[1]        
    
    def loss_calc(self,X,Y,Xval,Yval):
        regularization=1/2*self.model.lamdba_m*np.sum([np.sum(layer.W**2) for layer in self.model.layers])
        
        Ypred=self.model.predict(X)
        Yvalpred=self.model.predict(Xval)
        
        self.train_loss.append((self.model.loss(Y,Ypred)+regularization)/X.shape[1])
        self.val_loss.append(self.model.loss(Yval,Yvalpred)/Xval.shape[1])
        self.train_accuracy.append(self.accuracy_check(Y,Ypred))
        self.val_accuracy.append(self.accuracy_check(Yval,Yvalpred))
            
    def loss_calc_fit(self,X,Y):
        regularization=1/2*self.model.lamdba_m*np.sum([np.sum(layer.W**2) for layer in self.model.layers])
        Ypred=self.model.predict(X)
        self.train_loss.append((self.model.loss(Y,Ypred)+regularization)/X.shape[1])
        self.train_accuracy.append(self.accuracy_check(Y,Ypred)) 

    def batch_gradient_descent(self,traindat,testdat):
        X,Y=traindat
        def update_batch(_):
            for layer in self.model.layers:
                layer.W=layer.W-self.learning_rate*layer.d_W
                layer.b=layer.b-self.learning_rate*layer.d_b
        updator=update_batch
        self.iterate(updator,X,Y,testdat)
            
    def momentum(self,traindat,testdat,momentum=0.9):
        X,Y=traindat
        
        u_W=[np.zeros(np.shape(layer.d_W)) for layer in self.model.layers]
        u_b=[np.zeros(np.shape(layer.d_b)) for layer in self.model.layers]
        
        def update_mom(_):
            for i in range(len(self.model.layers)):
                layer=self.model.layers[i]
                u_W[i]=momentum*u_W[i]+layer.d_W
                u_b[i]=momentum*u_b[i]+layer.d_b
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
                layer = self.model.layers[i]
                # Update momentum and velocity
                m_W[i] = beta1 * m_W[i] + (1 - beta1) * layer.d_W
                m_b[i] = beta1 * m_b[i] + (1 - beta1) * layer.d_b

                v_W[i] = beta2 * v_W[i] + (1 - beta2) * layer.d_W**2
                v_b[i] = beta2 * v_b[i] + (1 - beta2) * layer.d_b**2

                # Bias correction
                m_W_hat = m_W[i] / (1 - np.power(beta1, t + 1))
                m_b_hat = m_b[i] / (1 - np.power(beta1, t + 1))
                v_W_hat = v_W[i] / (1 - np.power(beta2, t + 1))
                v_b_hat = v_b[i] / (1 - np.power(beta2, t + 1))

                # Update parameters
                layer.W -= (self.learning_rate * m_W_hat) / (np.sqrt(v_W_hat) + epsilon)
                layer.b -= (self.learning_rate * m_b_hat) / (np.sqrt(v_b_hat) + epsilon)

        updator = update_adam
        self.iterate(updator, X, Y, testdat)
    
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
        
    def run(self, traindat, testdat=None, momentum=0.9, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-10):
        if self.optimizer == "sgd":
            self.batch_gradient_descent(traindat, testdat)
        elif self.optimizer == "momentum":
            self.momentum(traindat, testdat, momentum=momentum)
        elif self.optimizer == "nesterov":
            self.NAG(traindat, testdat, beta=beta)
        elif self.optimizer == "rmsprop":
            self.rmsprop(traindat, testdat, beta=beta, epsilon=epsilon)
        elif self.optimizer == "adam":
            self.Adam(traindat, testdat, beta1=beta1, beta2=beta2, epsilon=epsilon)
        elif self.optimizer == "nadam":
            self.NAdam(traindat, testdat, beta1=beta1, beta2=beta2, epsilon=epsilon)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")