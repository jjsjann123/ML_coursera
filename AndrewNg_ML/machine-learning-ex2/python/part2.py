from utility import *

old_data = loaddata('ex2data2.txt')
data, X, Y = processData('ex2data2.txt')
X = expandData(X, 6)
theta = np.zeros(X.shape[1]);
sg = np.vectorize(sigmoid)
#Z = np.arange(0,6).reshape(3,2)

#l = 0.1
l = 01

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

print computeJV(X, Y, theta)
res = optimize.minimize(obj, theta, method='BFGS', jac=negder)
draw(data['data'])

x_cord = np.arange(-2, 2, 0.1)
y_cord = np.arange(-2, 2, 0.1)
x, y = np.meshgrid(x_cord, y_cord)
J = np.zeros(x.shape)

for i in range(0, x.shape[0]):
    for j in range(0, x.shape[1]):
        features = expandArray(np.array([1, x[i][j], y[i][j]]), 6)
        J[i][j] = computeHV(features, 0, res.x)
        
plt.contour(x, y, J, [0.5])
plt.show()
