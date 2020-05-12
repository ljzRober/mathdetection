import numpy as np
from bpnn import serialize

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

    theta1[:, 0] = 0
    theta2[:, 0] = 0
    theta3[:, 0] = 0
    Theta1_grad = 1 / m * (np.dot(d2.T , X) + lambda1 * theta1)
    Theta2_grad = 1 / m * (np.dot(d3.T , a2) + lambda1 * theta2)
    Theta3_grad = 1 / m * (np.dot(d4.T , a3) + lambda1 * theta3)

    theta = serialize(Theta1_grad, Theta2_grad, Theta3_grad)

    return J, theta



