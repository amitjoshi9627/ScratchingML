from utils import *

class LogisticRegression:

    def __init__(self,max_iter=10000):
        self.max_iter = max_iter
    
    def _initialize_weights(self):
        self.weights = np.array([0.0]*self.nfeatures)

    def _initialize_bias(self):
        self.bias = 0.0

    def _update_weights(self,features,targets,weights,bias,lr):
        
        weights = weights.astype("float")
        N = len(features)
        predictions = sigmoid(np.dot(features,weights))
        gradient_descent = np.dot(features.T,predictions - targets) / N
        weights -= lr * gradient_descent

        return weights
    
    def _cost_function(self,features,targets,weights):
        
        predictions = sigmoid(np.dot(features,weights))
        return log_loss(predictions,targets)

    def fit(self,X_train,y_train,learning_rate=0.01):
        
        X_train,y_train = np.array(X_train),np.array(y_train)
        self.nfeatures = X_train.shape[1]
        self._initialize_weights()
        self._initialize_bias()

        for i in range(self.max_iter):
            self.weights = self._update_weights(X_train,y_train,self.weights,self.bias,learning_rate)
            cost = self._cost_function(X_train,y_train,self.weights)
            if (i+1) % 50 == 0:
                print("iter={:d}    error={:.2}".format(i+1,cost))

    def predict(self,X_test):
        X_test = np.array(X_test)
        predictions = np.dot(X_test,self.weights) 
        return np.round(sigmoid(predictions))

