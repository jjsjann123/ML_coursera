from utility import *

data = scipy.io.loadmat('ex3data1.mat')
X = np.insert(data['X'], 0, 1, axis=1)
y = data['y']
l = 1.0
#l = 0
theta = np.zeros((10, 401))

def obj(theta):
    global X
    global Y
    global l
    return computeJRV(X, Y, theta, l)

def negder(theta):
    global X
    global Y
    global l
    return computeGRV(X, Y, theta, l)

def der(theta):
    global X
    global Y
    global l
    return computeGRV(X, Y, theta, l)

for i in range(0, 10):
    Y = np.where(y[:,0]-i-1, 0, 1)
    res = optimize.minimize(obj, theta[i,:], method='BFGS', jac=negder)
    theta[i,:] = res.x
    prediction = np.where(computeHV(X, res.x)>0.5, 1, 0)
    hit = np.where(prediction==Y,1,0)
    print i, "result: ", float(sum(hit))/hit.size

res = np.argmax(computeHV(X, theta.T), axis=1)
hit = np.where(res+1==y[:,0], 1, 0)
print "overall result: ", float(sum(hit))/hit.size

