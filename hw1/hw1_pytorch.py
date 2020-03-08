print("# import packages")

import sys
import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
import csv

print("# read in training set")
raw_data = np.genfromtxt(sys.argv[1], delimiter=',') ## train.csv
data = raw_data[1:,3:]
where_are_NaNs = np.isnan(data)
data[where_are_NaNs] = 0

month_to_data = {}  ## Dictionary (key:month , value:data)

for month in range(12):
    sample = np.empty(shape = (18 , 480))
    for day in range(20):
        for hour in range(24):
            sample[:,day * 24 + hour] = data[18 * (month * 20 + day): 18 * (month * 20 + day + 1),hour]
    month_to_data[month] = sample

print("# preprocess")
x = np.empty(shape = (12 * 471 , 18 * 9),dtype = float)
y = np.empty(shape = (12 * 471 , 1),dtype = float)

for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour,:] = month_to_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1,-1)
            y[month * 471 + day * 24 + hour,0] = month_to_data[month][9 ,day * 24 + hour + 9]


print("# normalization")
#mean = np.mean(x, axis = 0)
#std = np.std(x, axis = 0)
#for i in range(x.shape[0]):
#    for j in range(x.shape[1]):
#        if not std[j] == 0 :
#            x[i][j] = (x[i][j]- mean[j]) / std[j]

cuda0 = torch.device('cuda:0')
#x_data = Variable(torch.Tensor(x, device=cuda0))
x_data = torch.from_numpy(x).float().to(cuda0)
#y_data = Variable(torch.Tensor(y, device=cuda0))
y_data = torch.from_numpy(y).float().to(cuda0)

print('x_shape: {}'.format(x_data.shape))
print('y_shape: {}'.format(y_data.shape))

print("# training")
#dim = x.shape[1] + 1
#w = np.zeros(shape = (dim, 1 ))
#x = np.concatenate((np.ones((x.shape[0], 1 )), x) , axis = 1).astype(float)
#learning_rate = np.array([[200]] * dim)
#adagrad_sum = np.zeros(shape = (dim, 1 ))

class LinearRegressionModel(torch.nn.Module):

    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(162, 1)  # One in and one out

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

our_model = LinearRegressionModel()

if torch.cuda.is_available():
    print('#use cuda')
    our_model.cuda()

criterion = torch.nn.MSELoss(size_average = False)
#optimizer = torch.optim.Adam(our_model.parameters(), lr = 0.01)
optimizer = torch.optim.Adam(our_model.parameters(), lr = 0.01)

for epoch in range(10000):

    # Forward pass: Compute predicted y by passing
    # x to the model
    pred_y = our_model(x_data)

    # Compute and print loss
    loss = criterion(pred_y, y_data)

    # Zero gradients, perform a backward pass,
    # and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('epoch {}, loss {}'.format(epoch, loss)) if not epoch % 1000 else False

print("# read in testing set")
test_raw_data = np.genfromtxt(sys.argv[2], delimiter=',')   ## test.csv
test_data = test_raw_data[:, 2: ]
where_are_NaNs = np.isnan(test_data)
test_data[where_are_NaNs] = 0

print("# predict")
test_x = np.empty(shape = (240, 18 * 9),dtype = float)

for i in range(240):
    test_x[i,:] = test_data[18 * i : 18 * (i+1),:].reshape(1,-1)

x_test = torch.from_numpy(test_x).float().to(cuda0)

answer = our_model(x_test)


print("#write file")
f = open(sys.argv[3],"w")
w = csv.writer(f)
title = ['id','value']
w.writerow(title)
for i in range(240):
    content = ['id_'+str(i),answer[i][0].item()]
    w.writerow(content)
