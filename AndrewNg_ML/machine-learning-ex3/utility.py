import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy import optimize
import scipy.io

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

def processData(file_name):
    data = normalizeData(loaddata(file_name))
    mat = data['data']
    X = np.insert(mat[:, 0:mat.shape[1]-1], 0, 1, axis=1)
    Y = mat[:, mat.shape[1]-1]
    return (data, X, Y)

def sigmoid(x):
    try:
        ret = 1.0/(1+math.exp(-x))
    except OverflowError:
        ret = 0.0
    return ret

def computeHV(X, theta):
    sg = np.vectorize(sigmoid)
    return sg(X.dot(theta))

def computeJV(X, Y, theta):
    m = Y.size
    h_theta = computeHV(X, theta)
    return ( -np.log(h_theta.T).dot(Y) - np.log(1-h_theta.T).dot(1-Y))/m

def computeGV(X, Y, theta):
    m = Y.size
    h_theta = computeHV(X, theta)
    return X.T.dot(h_theta-Y)/m

def draw(data):
    plt.scatter(data[:,0],data[:,1],data[:,2]*25,c='r',marker='o')
    plt.scatter(data[:,0],data[:,1],(1-data[:,2])*25,c='b',marker='x')

def computeJRV(X, Y, theta, l):
    m = Y.size
    h_theta = computeHV(X, theta)
    l_theta = np.copy(theta)
    l_theta[0] = 0;
    return (l/2.0*sum(l_theta**2) - np.log(h_theta.T).dot(Y) - np.log(1-h_theta.T).dot(1-Y))/m

def computeGRV(X, Y, theta, l):
    m = Y.size
    h_theta = computeHV(X, theta)
    l_theta = np.copy(theta)
    l_theta[0] = 0;
    return (X.T.dot(h_theta-Y)+l*l_theta)/m

def expandData(X, order):
    for cur in np.arange(2, order+1):
        for iter in np.arange(0, cur+1):
            X = np.insert(X, X.shape[1], (X[:,1]**iter)*(X[:,2]**(cur-iter)), axis=1)
    return X

def expandArray(X, order):
    for cur in np.arange(2, order+1):
        for iter in np.arange(0, cur+1):
            X = np.append(X, (X[1]**iter)*(X[2]**(cur-iter)))
    return X
