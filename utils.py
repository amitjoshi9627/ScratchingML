import numpy as np


def euclidean_distance(data, y):

    sq_distance = 0
    for ind in range(0, data.shape[0]):
        sq_distance += np.square((data[ind] - y[ind]))

    return np.sqrt(sq_distance)


def MinMaxScaler(data):

    data = np.array(data)
    for ind in range(0, data.shape[1]):

        dataMin = np.amin(data[:, ind])
        dataMax = np.amax(data[:, ind])
        data[:, ind] = (data[:, ind] - dataMin) / (dataMax - dataMin)

    return data


def StandardScaler(data):

    data = np.array(data)
    for ind in range(0, data.shape[1]):
        n = data.shape[0]
        dataMean = np.mean(data[:, ind])

        res = np.sum(np.square((data[:, ind] - dataMean)))
        stddev = np.sqrt(res / n)

        if stddev:
            data[:, ind] = (data[:, ind] - dataMean) / stddev

    return data

def train_test_split(X,y,test_size=0.3,train_size=None,random_state = None):

    if random_state is not None:
        np.random.seed(seed = random_state)
    if train_size:
        test_size = 1.0 - train_size
    X = np.array(X)
    y = np.array(y)
    size = X.shape[0]
    indx = np.random.choice(size,int(size * test_size))

    return X[~indx],X[indx],y[~indx],y[indx]

def accuracy_score(data1,data2):
    return np.mean(np.array(data1) == np.array(data2))

def mean_squared_error(data1,data2):
    
    return np.mean(np.square(data1 - data2))

def mean_squared_log_error(h,y):

    return np.mean(np.square(np.log(h+1)-np.log(y+1)))

def sigmoid(x):

    return 1.0 / (1+np.exp(-x))

def log_loss(h,y):

    cost1 = -1 * y * np.log(h)
    cost2 = (1-y) * np.log(1-h) 
    return np.mean(cost1 - cost2)

def softmax(scores):

    return np.exp(scores) / np.sum(np.exp(scores),axis=0)