import numpy as np
from Layer import Layer as Layer
import functions as fn
import copy

class Network:
    def __init__(self, sizes, layerConnectivity = 'dense'):
        self.sizes = sizes
        self.network = []
        for i in range(len(sizes) - 1):
            if layerConnectivity == 'dense' or layerConnectivity[i] == 'dense':
                self.network.append(Layer(sizes[i], sizes[i+1]))
            else:
                self.network.append(Layer(sizes[i], sizes[i+1], layerConnectivity[i], dense=False))
        self.network.append(Layer(sizes[-1], 0))
        self.inputLayer = self.network[0]
        self.outputLayer = self.network[-1]
        self.networkBackup = copy.deepcopy(self.network)

        
    def feedForward(self, inputValues, activation):
        self.inputLayer.nodeVals = inputValues

        for i in range(len(self.network) - 1):
            self.network[i].nodeVals[0] = 1 # bias node
            self.network[i+1].beforeActivation = np.matmul(self.network[i].edges, self.network[i].nodeVals)
            self.network[i+1].nodeVals = activation(self.network[i+1].beforeActivation)

    def backPropogation(self, target, learningRate, loss, dfLoss, activation, dfActivation):
        self.outputLayer.dfActivation = dfActivation(self.outputLayer.beforeActivation)
        dfL = dfLoss(self.outputLayer.nodeVals, target)
        self.outputLayer.backDelta = dfL * self.outputLayer.dfActivation

        for i in range(len(self.network) - 2, -1, -1):
            currLayer = self.network[i]
            nextLayer = self.network[i+1]
            currLayer.dfActivation = dfActivation(currLayer.beforeActivation)
            for j in range(currLayer.size):
                currDelta = 0
                for k in range(nextLayer.size):
                    currDelta += currLayer.edges[k, j] * nextLayer.backDelta[k] * currLayer.dfActivation[j]
                currLayer.backDelta[j] = currDelta

        for i in range(len(self.network) - 2, -1, -1):
            currLayer = self.network[i]
            nextLayer = self.network[i+1]
            for j in range(currLayer.size):
                for k in range(nextLayer.size):
                    if currLayer.isDense or (j, k) in currLayer.activeEdges:
                        addval = learningRate * nextLayer.backDelta[k] * currLayer.nodeVals[j]
                        currLayer.edges[k, j] += addval
                        
        # for layer in self.network:
        #     print (layer.edges)
        # print()

    def fit(self, trainFeatures, trainTargets, testFeatures, testTargets, loss, dfLoss, activation, dfActivation, epochs, learningRate):
        epochDetails = []

        if len(trainFeatures[0]) != self.inputLayer.size:
            raise Exception()
        if len(trainTargets[0]) != self.outputLayer.size:
            raise Exception()
        if len(testFeatures[0]) != self.inputLayer.size:
            raise Exception()
        if len(testTargets[0]) != self.outputLayer.size:
            raise Exception()
        
        for epoch in range(epochs):

            trainingLosses = []
            for i in range(len(trainFeatures)):
                self.feedForward(trainFeatures[i], activation)
                # print(self.outputLayer.nodeVals)
                # print(trainTargets[i])
                trainingLosses.append(sum(loss(self.outputLayer.nodeVals, trainTargets[i])))
                self.backPropogation(trainTargets[i], learningRate, loss, dfLoss, activation, dfActivation)

            testingLosses = []
            correctClassifications = []
            for i in range(len(testFeatures)):
                self.feedForward(testFeatures[i], activation)
                testingLosses.append(sum(loss(self.outputLayer.nodeVals, testTargets[i])))
                if np.argmax(self.outputLayer.nodeVals) == np.argmax(testTargets[i]):
                    correctClassifications.append(True)
                else:
                    correctClassifications.append(False)

            trainingLossesMean = np.mean(trainingLosses)
            testingLossesMean = np.mean(testingLosses)
            correctness = correctClassifications.count(True) / len(correctClassifications)

            print('end of epoch ' + str(epoch) + ':')
            print(' mean of training loss: ' + str(trainingLossesMean))
            print(' mean of testing loss: ' + str(testingLossesMean))
            print(' correctness: {:.1%}'.format(correctness))
            print()
                    
            epochDetails.append((trainingLossesMean, testingLossesMean, correctness, trainingLosses, testingLosses, correctClassifications))

        return epochDetails

    def resetToInitialState(self):
        self.network = copy.deepcopy(self.networkBackup)
        self.inputLayer = self.network[0]
        self.outputLayer = self.network[-1]