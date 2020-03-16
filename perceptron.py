import argparse
import numpy as np
import matplotlib.pyplot as plt
def visualize_2D(X,W): #works only for 2D datapoints
    plt.figure(figsize=(10,10))
    xmin,xmax = (min(X[:,0]),max(X[:,0]))
    xrange = xmax - xmin
    x = np.linspace(xmin - xrange*0.1, xmax + xrange*0.1,100,0.01)
    y = -(W[1]/W[2])*x -(W[0]/W[2])
    plt.scatter(X[X[:,2] == 1][:,0],X[X[:,2] == 1][:,1],c='purple')
    plt.scatter(X[X[:,2] == 0][:,0],X[X[:,2] == 0][:,1],c='green')
    plt.plot(x, y, '-r', label='decision boundary',c='black')
    plt.title('Scatter Plot with Linear Boundary')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()
    return

#Empirical Risk Minimization
def erm(X,Y):
    W,F = perceptron(X,Y)
    m = len(Y)
    error = [1 for i in range(m) if Y[i] != F[i]]
    error = sum(error)/m
    print("Weight Vector : ",W)
    print('Error of prediction : ',error)
    return error

#perceptron Algorithm
def perceptron(X,Y,WT=[],test=False):
    m = X.shape[0]
    B = np.ones(m)
    X = np.vstack((B,X.T)) # add a row for bias
    if len(WT) > 0 and test:
        W = WT
    else:
        W = np.random.uniform(0,1,X.shape[0])
    convergence = False
    iterations,epsilon,F = (0,0.000000001,[])
    while not convergence:
        d = np.dot(W,X)
        F = [1 if i > 0 else 0 for i in d]
        if test: return (W,F)
        delta = np.dot((Y - F),X.T)
        if abs(np.sum(delta)) < epsilon:
            convergence = True
        W += delta
        iterations += 1
        if iterations > 1000:
            #print("Convergence not possible : Inseperable")
            return (W,F)
    #print("Convergence took ",iterations,"iterations")
    return (W,F)

def cross_validation(X,Y,k):
    m = X.shape[0]
    if k > m:
        return "Error: fold size cannot exceed sample size"
    it,fold, size = (1,0,m//k)
    errors = []
    while(fold < m):
        test_X,test_Y = X[fold: fold + size],Y[fold: fold + size]
        train_X,train_Y = np.concatenate([X[:fold],X[fold+size:]]),np.concatenate([Y[:fold],Y[fold+size:]])
        W,_ = perceptron(train_X,train_Y)
        NW,F = perceptron(test_X,test_Y,WT=W,test=True)
        err = [1 for i in range(len(F)) if test_Y[i] != F[i]]
        err = sum(err)/len(F)
        errors.append(err)
        it += 1
        fold += (size)
    print("Weight Vector : ",W)
    print("Individual Errors : ",errors)
    print("Mean Error across 10 folds : ",np.mean(errors))
    return (W,errors,np.mean(errors))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Here\'s how to run a perceptron!')
    parser.add_argument('--dataset',default='linearly-separable-dataset.csv')
    parser.add_argument('--mode',default='erm')
    args = parser.parse_args()
    dt = np.genfromtxt(args.dataset, delimiter=',', skip_header = 1)    # Assumes the given file is a csv file.
    X,Y = np.hsplit(dt, np.array([dt.shape[1]-1]))
    if args.mode == 'erm':
        err = erm(X,Y.T[0])
    elif args.mode == 'cv':
        W,E,AE = cross_validation(X,Y.T[0],10)
    else:
        print("Invalid option - Try Again!")
