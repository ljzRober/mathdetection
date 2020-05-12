import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

theta1 = pd.read_csv('theta1.csv')
theta2 = pd.read_csv('theta2.csv')
theta3 = pd.read_csv('theta3.csv')

theta1 = theta1.values[:, 1:]
theta2 = theta2.values[:, 1:]
theta3 = theta3.values[:, 1:]

#show picture
path = "dataset2/5/5_62057.jpg"
img = Image.open(path)
data = img.getdata()
data = np.matrix(data)
q = data.reshape(45,45)
print(q)
reim  = Image.fromarray(q)
plt.imshow(reim)
plt.show()

#recognize
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
q = q.reshape(1, 2025)
q = np.insert(q, 0, values=np.ones(q.shape[0]), axis=1)
z2 = np.dot(q, theta1.T)
a2 = np.insert(sigmoid(z2), 0, values=np.ones(q.shape[0]), axis=1)
z3 = np.dot(a2, theta2.T)
a3 = np.insert(sigmoid(z3), 0, values=np.ones(q.shape[0]), axis=1)
z4 = np.dot(a3, theta3.T)
a4 = sigmoid(z4)
print(a4)
words = ["+", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "=", "div", "x"]
result1  = np.argmax(a4)
print("数字为："+words[result1])