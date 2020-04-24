import random as rand
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
def errorplot(err):
    plt.figure(figsize=(8,3))
    plt.plot(err, '|')
    plt.ylim(0.5,1.5)
    plt.xlabel('Iterations')
    plt.ylabel('Misclassified')
    plt.show()
    return
def draw(X,Y,W): #works only for 2D datapoints
    plt.figure(figsize=(10,10))
    x = X[:,1]
    y = -(W[1]/W[2])*x -(W[0]/W[2])
    closest,margin = float('inf'),float('inf')
    for i,p in enumerate(X):
        d = abs(np.dot(W,p))
        if d < closest:
            closest = d
            margin = abs(y[i] - X[i][2])
    y_down = y - margin
    y_up = y + margin
    plt.scatter(X[:,1],X[:,2],c=Y)
    plt.plot(x, y, 'k', label='seperating hyperplane',c='b')
    plt.plot(x, y_down, ':',c='gray')
    plt.plot(x, y_up, ':',c='gray')
    plt.title('Visualization of a maximum-margin hyperplane')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()
    return

def train(X,Y,T = 10000):
    C = 1
    maxfval = np.amax(X)
    theta,step,m = np.zeros(X.shape[1]),25,len(X)
    weights,errors = [],[]
    for t in range(1,T):
        error = 0
        wt = (step*(1/t)) * theta
        i = t%m
        if Y[i]*np.dot(wt,X[i]) < 1:
            theta += C*Y[i]*X[i]
            error = 1
        weights.append(wt)
        errors.append(error)
    wbar = np.mean(weights,axis=0)
    return weights[-1],errors

def test(X,Y,W):
    misclassified = 0
    for i,x in enumerate(X):
        if Y[i]*np.dot(W,x) < 1:
            misclassified += 1
    f = open('README.txt',"w")
    print('Total number of test data points : ',X.shape[0],file = f)
    print('Total number of misclassified data points : ',misclassified,file=f)
    return

if __name__=="__main__":
    X0,Y = make_blobs(n_samples=100, n_features = 2, centers=2, cluster_std=1.05, random_state=10)
    X1 = np.c_[np.ones((X0.shape[0])), X0] # add one to the x-values to incorporate bias
    Y[Y==0]= -1
    train_test_split = 0.8
    tsamples = int(train_test_split*X1.shape[0])
    trainX,trainY,testX,testY = X1[:tsamples],Y[:tsamples],X1[tsamples:],Y[tsamples:]
    W,err = train(trainX,trainY)
    draw(trainX,trainY,W)
    test(testX,testY,W)
