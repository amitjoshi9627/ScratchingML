from utils import *

class LinearRegression:
    
    def __init__(self,max_iter=1000):
        self.max_iter = max_iter

    
    def _initialize_weights(self):
        self.weights = np.array([0.0]*self.nfeatures)

    def _initialize_bias(self):
        self.bias = 0.0

    def _update_weights(self,features,targets,weights,bias,lr):
        
        predictions = np.dot(features,weights) + bias
        weights = weights.astype('float')        
        error  = targets - predictions

        weights_derivative = -features * (error.reshape(error.shape[0],1))
        bias_derivative = -1 * (error.reshape(error.shape[0],1))

        weights = weights -  2 * lr * np.mean(weights_derivative,axis=0)
        bias = bias - 2 * lr * np.mean(bias_derivative)

        return weights,bias

    def _cost_function(self,features,targets,weights,bias):
        
        predictions = np.dot(features,weights) + bias
        sq_error = np.square(targets - predictions)
        return np.mean(sq_error)

    def fit(self,X_train,y_train,learning_rate = 0.01):
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        self.nfeatures = X_train.shape[1]
        self._initialize_weights()
        self._initialize_bias()

        for i in range(self.max_iter):
            self.weights,self.bias = self._update_weights(X_train,y_train,self.weights,self.bias,learning_rate)
            cost = self._cost_function(X_train,y_train,self.weights,self.bias)
            if (i+1) % 50 == 0:
                print("iter={:d}    error={:.2}".format(i+1, cost))

    def predict(self,X_test):
        X_test = np.array(X_test)
        predictions = np.dot(X_test,self.weights) + self.bias
        return predictions

