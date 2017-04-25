from nn_utility import *

data = scipy.io.loadmat('ex3data1.mat')
X = data['X']
y = data['y']

weight = scipy.io.loadmat('ex3weights.mat')
theta = [weight['Theta1'], weight['Theta2']]

res = forwardPropagation(X, theta)
hit = prediction(res, y)
print "overall result: ", float(sum(hit))/hit.size

