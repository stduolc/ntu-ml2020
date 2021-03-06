import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

class Normalizer():
    def __init__(self, num_inputs):
        self.n = torch.zeros(num_inputs)
        self.mean = torch.zeros(num_inputs)
        self.mean_diff = torch.zeros(num_inputs)
        self.var = torch.zeros(num_inputs)

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.clone()
        self.mean += (x-self.mean)/self.n
        self.mean_diff += (x-last_mean)*(x-self.mean)
        self.var = torch.clamp(self.mean_diff/self.n, min=1e-2)

    def normalize(self, inputs):
        obs_std = torch.sqrt(self.var)
        return (inputs - self.mean)/obs_std

class LinearRegression(torch.nn.Module):

    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(510, 1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

X_train_fpath = sys.argv[1]
Y_train_fpath = sys.argv[2]
X_test_fpath = sys.argv[3]
output_fpath = sys.argv[4]

print('# read training data')
X_train = np.genfromtxt(X_train_fpath, delimiter=',', skip_header=1)
Y_train = np.genfromtxt(Y_train_fpath, delimiter=',', skip_header=1)
X_train = np.delete(X_train, 0, axis=1)
Y_train = np.delete(Y_train, 0, axis=1)

print("#X_train={} Y_train={}".format(X_train.shape, Y_train.shape))


# These are the columns that I want to normalize
def _normalize_column_normal(X, train=True, specified_column = None, X_mean=None, X_std=None):
    # The output of the function will make the specified column number to
    # become a Normal distribution
    # When processing testing data, we need to normalize by the value
    # we used for processing training, so we must save the mean value and
    # the variance of the training data
    if train:
        if specified_column == None:
            specified_column = np.arange(X.shape[1])
        length = len(specified_column)
        X_mean = np.reshape(np.mean(X[:, specified_column],0), (1, length))
        X_std  = np.reshape(np.std(X[:, specified_column], 0), (1, length))

    X[:,specified_column] = np.divide(np.subtract(X[:,specified_column],X_mean), X_std)

    return X, X_mean, X_std

col = [0,1,3,4,5,7,10,12,25,26,27,28]
X_train, X_mean, X_std = _normalize_column_normal(X_train, specified_column=col)

X_data = Variable(torch.Tensor(X_train).cuda())
Y_data = Variable(torch.Tensor(Y_train).cuda())


def _normalize_column_0_1(X, train=True, specified_column = None, X_min = None, X_max=None):
    # The output of the function will make the specified column of the training data
    # from 0 to 1
    # When processing testing data, we need to normalize by the value
    # we used for processing training, so we must save the max value of the
    # training data
    if train:
        if specified_column == None:
            specified_column = np.arange(X.shape[1])
        length = len(specified_column)
        X_max = np.reshape(np.max(X[:, specified_column], 0), (1, length))
        X_min = np.reshape(np.min(X[:, specified_column], 0), (1, length))

    X[:, specified_column] = np.divide(np.subtract(X[:, specified_column], X_min), np.subtract(X_max, X_min))

    return X, X_max, X_min


def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def train_dev_split(X, y, dev_size=0.25):
    train_len = int(round(len(X)*(1-dev_size)))
    return X[0:train_len], y[0:train_len], X[train_len:None], y[train_len:None]


def _sigmoid(z):
    # sigmoid function can be used to output probability
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-6, 1-1e-6)

def get_prob(X, w, b):
    # the probability to output 1
    print("#shape X:{} w:{} b:{}".format(X.shape, w.shape, b.shape))
    return _sigmoid(np.matmul(X, w) + b)

def infer(X, w, b):
    # use round to infer the result
    return np.round(get_prob(X, w, b))

def _cross_entropy(y_pred, Y_label):
    # compute the cross entropy
    print("#shape y_pred={} Y_label={}".format(y_pred.shape, Y_label.shape))
    cross_entropy = -np.dot(Y_label, np.log(y_pred))-np.dot((1-Y_label), np.log(1-y_pred))
    return cross_entropy

def _gradient(X, Y_label, w, b):
    # return the mean of the graident
    y_pred = get_prob(X, w, b)
    pred_error = Y_label - y_pred
#    w_grad = -np.mean(np.multiply(pred_error, X.T), 1)
    w_grad = -np.mean(pred_error.T * X.T, 1, keepdims=True)
    b_grad = -np.mean(pred_error, keepdims=True)
    return w_grad, b_grad

def _gradient_regularization(X, Y_label, w, b, lamda):
    # return the mean of the graident
    #shape X: (32, 511) Y: (32, 2) w: (511, 2) b: (1,2)
    y_pred = get_prob(X, w, b)
    pred_error = Y_label - y_pred
    #shape: pred_error:(32, 2) X:(32, 511) w:(511, 2)
    w_grad = -np.mean(np.multiply(pred_error.T, X.T), 1)+lamda*w
    b_grad = -np.mean(pred_error)
    return w_grad, b_grad

def _loss(y_pred, Y_label, lamda, w):
    return _cross_entropy(y_pred, Y_label) + lamda * np.sum(np.square(w))

def accuracy(Y_pred, Y_label):
    acc = np.sum(Y_pred == Y_label)/len(Y_pred)
    return acc


def train(X_train, Y_train):
    # split a validation set

    dev_size = 0.1155
    X_train, Y_train, X_dev, Y_dev = train_dev_split(X_train, Y_train, dev_size = dev_size)
    # X_train.shape: (47989, 511) Y_train.shape: (47989, 2) X_dev.shape: (6267, 511) Y_dev.shape: (6267, 2)

    # Use 0 + 0*x1 + 0*x2 + ... for weight initialization
    regularize = True
    if regularize:
        lamda = 0.001
    else:
        lamda = 0

    max_iter = 4000 # max iteration number
    batch_size = 32 # number to feed in the model for average to avoid bias
    learning_rate = 0.1  # how much the model learn for each step
    step =1

    loss_train = []
    loss_validation = []
    train_acc = []
    dev_acc = []



    model = LinearRegression()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(max_iter):
        # Random shuffle for each epoch
        y_pred = model(X_data)
        loss = criterion(y_pred, Y_data)
        print('**epoch:{} loss:{}'.format(epoch, loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model

# return loss is to plot the result
model = train(X_train, Y_train)

X_test = np.genfromtxt(X_test_fpath, delimiter=',', skip_header=1)
X_test = np.delete(X_test, 0, axis=1)
col = [0,1,3,4,5,7,10,12,25,26,27,28]
X_test, X_mean, X_std = _normalize_column_normal(X_test, specified_column=col)

X_test = Variable(torch.Tensor(X_test).cuda())

# Do the same data process to the test data
#X_test, _, _= _normalize_column_normal(X_test, train=False, specified_column = col, X_mean=X_mean, X_std=X_std)
result = model(X_test)

with open(output_fpath, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(result):
            f.write('%d,%d\n' %(i, v))
