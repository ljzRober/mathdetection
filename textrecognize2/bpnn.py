import os
import sys
import numpy as np
from PIL import Image
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt


input_layer_size  = 2025 ;
hidden_layer_size1 = 1000 ;
hidden_layer_size2 = 500 ;
num_labels = 15 ;
x = np.ones((6000,2026)); #创建输入层
y = np.ones((6000,1)); #创建标签
count = 0;

path = "dataset2"
dirs = os.listdir(path)
for i in dirs:
    print(i)
    path = path+"/"+i
    files = os.listdir(path)
    for j in np.arange(0,400):
        path = path + "/" + files[j]
        im = Image.open(path)
        data = im.getdata()
        data = np.matrix(data)
        x[count][1:2026] = data

        y[count] = int(count/400)
        count += 1

        # if j == 0:
        #     q = data.reshape(45,45)
        #     reim  = Image.fromarray(q)
        #     plt.imshow(reim)
        #     plt.show()

        # if j == 0:
        #     print(files[j])
        #
        #     #二值化
        #     threshold = 185
        #     table = []
        #     for i in range(256):
        #         if i < threshold:
        #             table.append(0)
        #         else:
        #             table.append(1)
        #     # convert to binary image by the table
        #     bim = im.point(table, '1')
        #     plt.imshow(bim, cmap="binary")
        #     plt.show()
        path = "dataset2/" + i


    path = "dataset2"
print(x.shape)
print(y.shape)

def randInitializeWeights(L_in,L_out):
    epsilon_init = 0.12
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
    return W

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size1)
initial_Theta2 = randInitializeWeights(hidden_layer_size1, hidden_layer_size2)
initial_Theta3 = randInitializeWeights(hidden_layer_size2, num_labels)

def serialize(a, b, c): #序列化
    return np.concatenate((np.ravel(a), np.ravel(b), np.ravel(c)))

params = serialize(initial_Theta1, initial_Theta2, initial_Theta3)
params = params.reshape(params.shape[0], 1)
print(params.shape)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

def deserialize(seq): #转化回来
    return seq[:1000 * 2026].reshape(1000 , 2026), seq[1000 * 2026:1000 * 2026 + 500 * 1001].reshape(500 , 1001),\
           seq[1000 * 2026 + 500 * 1001:].reshape(15 , 501)

def bpfunction(params, X, y, lambda1):
    J = 0
    m1 = X.shape[0]
    m = X.shape[1]
    theta1, theta2, theta3 = deserialize(params)

    z2 = np.dot(X , theta1.T)
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m1), axis=1)
    z3 = np.dot(a2 , theta2.T)
    a3 = np.insert(sigmoid(z3), 0, values=np.ones(m1), axis=1)
    z4 = np.dot(a3 , theta3.T)
    a4 = sigmoid(z4)

    temp_y = np.zeros((m1, theta3.shape[0])) #分组
    for c in np.arange(theta3.shape[0]):
        temp_y[:, c] = np.reshape((y == c), 6000, 1)

    d4 = a4 - temp_y
    d3 = np.multiply(np.dot(d4 , theta3), sigmoid_gradient(a3))
    d3 = d3[:, 1: theta3.shape[1]]
    d2 = np.multiply(np.dot(d3 , theta2), sigmoid_gradient(a2))
    d2 = d2[:, 1: theta2.shape[1]]

    # compute the cost
    first_term = np.multiply(-temp_y, np.log(a4))
    second_term = np.multiply((1 - temp_y), np.log(1 - a4))
    J = np.sum(first_term - second_term)

    J = J / m

    # add the cost regularization term
    J += (float(lambda1) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)) + np.sum(np.power(theta3[:, 1:], 2)))

    Theta1_grad = 1 / m * (np.dot(d2.T, X) + np.insert(lambda1 * theta1[:, 1:], 0, values=np.ones(theta1.shape[0]), axis=1))
    Theta2_grad = 1 / m * (np.dot(d3.T, a2) + np.insert(lambda1 * theta2[:, 1:], 0, values=np.ones(theta2.shape[0]), axis=1))
    Theta3_grad = 1 / m * (np.dot(d4.T, a3) + np.insert(lambda1 * theta3[:, 1:], 0, values=np.ones(theta3.shape[0]), axis=1))

    params = serialize(Theta1_grad, Theta2_grad, Theta3_grad)

    return J, params

if __name__ == "__main__":
    # minimize the objective function
    fmin = minimize(fun=bpfunction, x0=params, args=(x, y, 0.6),
                method='TNC', jac=True, options={'maxiter': 50})

    print(fmin)
    #print(theta.shape)

    X = np.matrix(x)
    theta1, theta2, theta3 = deserialize(fmin.x)
    theta1_scv = pd.DataFrame(theta1)
    theta2_scv = pd.DataFrame(theta2)
    theta3_scv = pd.DataFrame(theta3)
    theta1_scv.to_csv('theta1.csv')
    theta2_scv.to_csv('theta2.csv')
    theta3_scv.to_csv('theta3.csv')
    z2 = np.dot(X, theta1.T)
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(X.shape[0]), axis=1)
    z3 = np.dot(a2, theta2.T)
    a3 = np.insert(sigmoid(z3), 0, values=np.ones(X.shape[0]), axis=1)
    z4 = np.dot(a3, theta3.T)
    a4 = sigmoid(z4)
    y_pred = np.array(np.argmax(a4, axis=1) + 1)
    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print('accuracy = {0}%'.format(accuracy * 100))
