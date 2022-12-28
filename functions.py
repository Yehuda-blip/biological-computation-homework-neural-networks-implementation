import numpy as np

EPS = 1e-6

def sigmoid(x):
     return 1 / (1 + np.exp(-x))
def dfSigmoid(sigVal):
    return sigmoid(sigVal) * (1 - sigmoid(sigVal))

relu = lambda x : (x > 0) * x
reluNeg = lambda x : (x > 0) * x - 1
dfRelu = lambda x : (x > 0) * 1

MSE = lambda val , target : ((target - val) ** 2) / 2
dfMSE = lambda val , target : (target - val)

def binaryCrossEntropy(val, target):
    val = np.clip(val, EPS, 1 - EPS)
    bce = -(target * np.log(val) + (1 - target) * np.log(1 - val))
    return np.full(len(target), np.mean(bce))

def dfBinaryCrossEntropy(val, target):
    val = np.clip(val, EPS, 1 - EPS)
    return (target / val) - (1 - target) / (1 - val)
    

lossCombinations = [(MSE, dfMSE, 'Mean Squared Err')]#, (binaryCrossEntropy, dfBinaryCrossEntropy, 'Binary Cross Entropy')]


activationCombinations = [(sigmoid, dfSigmoid, 'Sigmoid')]#, (relu, dfRelu, 'ReLU'), (reluNeg, dfRelu, 'Negative ReLU')]