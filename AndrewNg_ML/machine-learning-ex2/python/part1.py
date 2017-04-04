from utility import *

old_data = loaddata('ex2data1.txt')
data, X, Y = processData('ex2data1.txt')
theta = np.zeros(X.shape[1]);
sg = np.vectorize(sigmoid)

def obj(theta):
    global X
    global Y
    return computeJV(X, Y, theta)

def negder(theta):
    global X
    global Y
    return computeGV(X, Y, theta)

res = optimize.minimize(obj, theta, method='BFGS', jac=negder)
draw(data['data'])
plt.plot([-2.5, -(res.x[0]+res.x[2]*(-2.5))/res.x[1]],
        [-(res.x[0]+res.x[1]*(-2.5))/res.x[2], -2.5])
plt.show()
