from utility import *
import sys

def loadData(filename):
    data = scipy.io.loadmat(filename)
    pos = np.insert(data['pos_examples_nobias'], 2, 1, axis=1)
    neg = np.insert(data['neg_examples_nobias'], 2, 1, axis=1)
    w = data['w_init'][:,0]
    return (data, pos, neg, w)

def train_perceptron(pos, neg, w):
    old_w = np.copy(w)
    for i in range(0,pos.shape[0]):
        if old_w.dot(pos[i])<0:
            w += pos[i]
    for i in range(0,neg.shape[0]):
        if old_w.dot(neg[i])>=0:
            w -= neg[i]

def error(pos, neg, w):
    err = 0
    for i in range(0,pos.shape[0]):
        if w.dot(pos[i])<0:
            err+=1
    for i in range(0,neg.shape[0]):
        if w.dot(neg[i])>=0:
            err+=1
    return err

def draw(pos, neg, w):
    plt.plot(pos[:,0], pos[:,1], 'ro')
    plt.plot(neg[:,0], neg[:,1], 'bx')
    x = (w[0]-w[2])/w[1]
    y = -(w[0]+w[2])/w[1]
    plt.plot([-1, 1], [x, y])

def main():
    if len(sys.argv) >= 2:
        filename = sys.argv[1]
    else:
        filename = 'data/dataset1.mat'
    (data, pos, neg, w) = loadData(filename)
    cond = True
    while (cond):
        cmd = raw_input('q to quit')
        if cmd == 'q':
            cond = False
        else:
            draw(pos, neg, w)
            print ('error: ', error(pos, neg, w))
            print ('weight: ', w)
            plt.show()
            train_perceptron(pos, neg, w)

if __name__ == "__main__":
    main()

