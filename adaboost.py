import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
def erm(X,rounds):
    m,d = X.shape
    d = d - 1
    hs = np.zeros(m)
    WT,JT,TT = adaboost(X,rounds)
    kRoundErrors = []
    for j in range(rounds):
        h = [-1 if X[i][JT[j]] > TT[j] else 1 for i in range(m)]
        hs = np.add(hs,[h[i]*WT[j] for i in range(m)])
        tillJRoundErr = sum(1 for i in range(m) if hs[i]*X[i][d] < 0 )/m
        kRoundErrors.append(tillJRoundErr)
    error = sum(1 for i in range(m) if hs[i]*X[i][d] < 0 )/m
    print('Weight Vector : ',WT)
    print('Empirical Risk : ',error)
    return kRoundErrors

def cross_validation(X,rounds):
    m,d = X.shape
    d = d - 1
    k = 10
    it,fold, size = (0,0,m//k)
    errors = []
    _10FoldKroundErrors = [[] for i in range(10)]
    while(it < k):
        test = X[fold: fold + size]
        train = np.concatenate([X[:fold],X[fold+size:]])
        if it == (k-1):
            test = X[fold:]
            train = X[:fold]
        WT,JT,TT = adaboost(train,rounds)
        hs = np.zeros(len(test))
        for i in range(rounds):
            Jstar,Tstar,wt = JT[i],TT[i],WT[i]
            h = [-1 if test[i][Jstar] > Tstar else 1 for i in range(len(test))]
            hs = np.add(hs,[h[i]*wt for i in range(len(test))])
            err_till_i = sum(1 for i in range(len(test)) if hs[i]*test[i][d] < 0 )/len(test)
            _10FoldKroundErrors[it].append(err_till_i)
        err = sum(1 for i in range(len(test)) if hs[i]*test[i][d] < 0 )/len(test)
        errors.append(err)
        it += 1
        fold += (size)
    print('Weight Vector : ',WT)
    print('Individual Errors across 10 folds : ',errors)
    print('Average Validation Error across 10 folds : ',np.mean(errors))
    return np.mean(_10FoldKroundErrors,axis=0)

def plot(err,mode='erm'):
    plt.figure(figsize=(8,5))
    x = [i+1 for i in range(len(err))]
    plt.title('Plot of Empirical Risk with T rounds of Adaboost')
    plt.xlabel('Number of iterations')
    plt.ylabel('Training Error')
    if mode == 'cv':
        plt.title('Plot of Mean Validation Error with T rounds of Adaboost')
        plt.ylabel('Average Validation Error over 10 folds')
    plt.plot(x, err, '-r',c='blue')
    plt.grid()
    plt.show()
    return

#Input Parameters : X: numpy dataset, D: probability distribution
def weakLearner(X,D):# decision stumps
    #Last attribute represents label
    m,d = X.shape
    d = d - 1
    Fstar,Tstar,Jstar = np.inf,None,None
    for j in range(d):
        # Prepare dataset with attributes nthdimension,label,probability distribution
        S = np.vstack([X[:,j],X[:,d],D]).T
        S = S[S[:,0].argsort()]
        S = np.append(S,[[S[-1][0]+1,-1,1]],axis=0) #add extra element
        F = sum(S[i][2] for i in range(m) if S[i][1] == 1)
        if F < Fstar:
            Fstar,Tstar,Jstar = F,(S[0][0] - 1),j
        for i in range(m):
            F -= S[i][2]*S[i][1] # Distribution * actual_label
            if F < Fstar and S[i][0] != S[i+1][0]: # update Fstar
                Fstar,Tstar,Jstar = F,(S[i][0] + S[i+1][0])/2,j
    return (Jstar,Tstar)

#Input Parameters: X: Dataset, T: number of rounds
def adaboost(X,T):
    m,d = X.shape
    d = d - 1
    D,hs,WT,err,JT,TT = np.full(m, (1 / m)),np.zeros(m),[],[],[],[]
    for t in range(T):
        Jstar,Tstar = weakLearner(X,D)
        h = [-1 if X[i][Jstar] > Tstar else 1 for i in range(m)]
        epsilon = sum(D[i] for i in range(m) if X[i][d]*h[i] < 0)
        wt = 0.5*(math.log((1/(epsilon + 1e-10)) - 1))
        WT.append(wt)
        JT.append(Jstar)
        TT.append(Tstar)
        for j in range(m):
            D[j] *= np.exp((-1)*wt*h[j]*X[j][d])
        D /= np.sum(D)
    return (WT,JT,TT)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Here\'s how to run Adaboost!')
    parser.add_argument('--dataset',default='Breast_cancer_data.csv')
    parser.add_argument('--mode',default='erm')
    args = parser.parse_args()
    dt = np.genfromtxt(args.dataset, delimiter=',', skip_header = 1)    # Assumes the given file is a csv file.
    rounds = 100
    print("Number of Rounds of Adaboost = ",rounds)
    for i in range(dt.shape[0]):
        if dt[i][dt.shape[1]-1] == 0:
            dt[i][dt.shape[1]-1] = -1
    if args.mode == 'erm':
        err = erm(dt,rounds)
        #plot(err)
    elif args.mode == 'cv':
        err = cross_validation(dt,rounds)
        #plot(err,'cv')
    else:
        print("Invalid option - Try Again!")
