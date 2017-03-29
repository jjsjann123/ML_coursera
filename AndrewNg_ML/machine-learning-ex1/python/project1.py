import numpy as np
import matplotlib.pyplot as plt

def loaddata(file):
    return np.loadtxt(file, delimiter=',')

def linearRegression(x, y, theta, alpha):
    m = y.size
    theta = theta - alpha / m * np.dot(np.transpose(x), np.dot(x, theta) - y)
    return theta
    

def computeCostFunction(x, y, theta):
    m = y.size
    J = np.sum((np.dot(x, theta) - y)**2)/2/m
    return J

data = loaddata('../ex1/ex1data1.txt')
profits = data[:,0]
populations = data[:,1]

plt.plot(profits, populations, 'ro')
plt.ylabel('profit')
plt.xlabel('population')

data = np.insert(data, 0, 1, axis=1)
x = data[:, 0:2]
y = data[:, 2]
theta = np.array([10 ,0])
t = int(max(x[:,1]))+2

for i in range(0, 1500):
    print theta
    J = computeCostFunction(x, y, theta)
    print J
    theta = linearRegression(x, y, theta, 0.01)
    if i % 100 == 0:
        plt.plot([0, t], [theta[0], theta[0]+t*theta[1]]) 



plt.show();
