import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def loaddata(file):
    return np.loadtxt(file, delimiter=',')

def normalizeDataRange(data):
    avg = np.zeros(data.shape[1]);
    scl = np.ones(data.shape[1]);
    for i in range(0, data.shape[1]):
        #avg[i] = np.average(data[:,i])
        avg[i] = (np.max(data[:,i])+np.min(data[:,i]))/2.0
        scl[i] = 1.0/(np.max(data[:,i])-np.min(data[:,i]))
    return (data[:] - avg)*scl

def normalizeData(data):
    avg = np.zeros(data.shape[1]);
    std = np.ones(data.shape[1]);
    for i in range(0, data.shape[1]-1):
        avg[i] = np.average(data[:,i])
        std[i] = 1.0/np.std(data[:,i])
    return {'data':(data[:] - avg)*std, 'avg': avg, 'std': std}

def normalEquation(x, y):
    return np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)

def linearRegression(x, y, theta, alpha):
    m = y.size
    #theta = theta - alpha / m * np.dot(np.transpose(x), np.dot(x, theta) - y)
    theta = theta - alpha / m * x.T.dot(x.dot(theta)-y)
    return theta

def computeCostFunction(x, y, theta):
    m = y.size
    J = np.sum((np.dot(x, theta) - y)**2)/2/m
    return J

def computeJV(X, Y, theta):
    dif = (X.dot(theta)-Y) 
    return dif.T.dot(dif) / 2.0 / Y.size

data = loaddata('../ex1/ex1data2.txt')
n_data = normalizeData(data)

mat = n_data['data']
X = np.insert(mat[:, 0:mat.shape[1]-1], 0, 1, axis=1)
Y = mat[:, mat.shape[1]-1]
theta = np.zeros(X.shape[1])

iterations = np.arange(0, 1500)
J = np.zeros(iterations.size)

for i in iterations:
    J[i] = computeJV(X, Y, theta)
    theta = linearRegression(X, Y, theta, 0.01)

plt.plot(iterations, J, 'ro')
plt.show()

print computeJV(X, Y, theta)
theta = normalEquation(X,Y)
print computeJV(X, Y, theta)

'''
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

x_cord = np.arange(-10,10,0.025)
y_cord = np.arange(-1, 4, 0.025)
X, Y = np.meshgrid(x_cord, y_cord)
J = np.zeros(X.shape)
for i in range(1, X.shape[0]):
    for j in range(1, X.shape[1]):
        t0 = X[i][j]
        t1 = Y[i][j]
        t = np.array([t0, t1])
        J[i][j] = computeCostFunction(x, y, t)

#plt.figure().gca(projection='3d').plot_surface(X, Y, J)
v = np.array([1, 2.0, 3.0, 5, 10, 50, 100, 200, 300, 400])
plt.figure().gca(projection='3d').contour(X, Y, J, v)

plt.show();
'''
