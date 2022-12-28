import numpy as np
import pandas as pd
import copy
import random
import Network
import functions as fn
import matplotlib.pyplot as plt

VIRGINICA = 'Iris-virginica'
VERSICOLOR = 'Iris-versicolor'
SETOSA = 'Iris-setosa'
targetIndices = {
    VIRGINICA: 0,
    VERSICOLOR: 1,
    SETOSA: 2
}

df = pd.read_csv('Iris.csv')
df.head()
features = df.loc[:, 'SepalLengthCm':'PetalWidthCm'].values
strTargets = df.loc[:, 'Species'].values
targets = []
for target in strTargets:
    t = np.zeros(3)
    t[targetIndices[target]] = 1.0
    targets.append(t)

targets = np.array(targets)

def trainTestSplit(featuresSrc, targetsSrc, testSize):
    _features = copy.deepcopy(featuresSrc)
    _targets = copy.deepcopy(targetsSrc)

    trainFeatures = []
    trainTargets = []
    testFeatures = []
    testTargets = []

    randomEnd = len(_features)
    for i in range(testSize):
        testSample = random.randrange(randomEnd)
        testFeatures.append(copy.deepcopy(_features[testSample]))
        testTargets.append(copy.deepcopy(_targets[testSample]))
        _features[testSample] = _features[randomEnd - 1]
        _targets[testSample] = _targets[randomEnd - 1]
        randomEnd -= 1

    for i in range(len(_features) - len(testFeatures)):
        trainSample = random.randrange(randomEnd)
        trainFeatures.append(copy.deepcopy(_features[trainSample]))
        trainTargets.append(copy.deepcopy(_targets[trainSample]))
        _features[testSample] = _features[randomEnd - 1]
        _targets[testSample] = _targets[randomEnd - 1]
        randomEnd -= 1

    if randomEnd != 0:
        raise Exception()


    return np.array(trainFeatures), np.array(trainTargets), np.array(testFeatures), np.array(testTargets)

trainFeatures, trainTargets, testFeatures, testTargets = trainTestSplit(features, targets, 30)


network1 = Network.Network([4,53,3])
network2 = Network.Network([4,16,16,3])
network3 = Network.Network([4,9,9,9,8,3])

network4sizes = [4,13,13,13,13,3]
cluster1 = [1,2,3,4]
cluster2 = [5,6,7,8]
cluster3 = [9,10,11,12]

network4Edges = ['dense',[],'dense',[],'dense']

for n1 in cluster1:
    for n2 in cluster1:
        network4Edges[1].append((n1,n2))
for n1 in cluster2:
    for n2 in cluster2:
        network4Edges[1].append((n1,n2))
for n1 in cluster3:
    for n2 in cluster3:
        network4Edges[1].append((n1,n2))
for n3 in range(13):
    network4Edges[1].append((0,n3))


for n1 in cluster1:
    for n2 in cluster1:
        network4Edges[3].append((n1,n2))
for n1 in cluster2:
    for n2 in cluster2:
        network4Edges[3].append((n1,n2))
for n1 in cluster3:
    for n2 in cluster3:
        network4Edges[3].append((n1,n2))
for n3 in range(13):
    network4Edges[3].append((0,n3))

network4 = Network.Network(network4sizes, network4Edges)

colors = ['b']#,'g','r','c']
labels = ['N1: Single very large layer']#, 'N2: Two large layers', 'N3: "deep" - 4 medium layers', 'N4: Sector divied coputaution']
networks = [network1]#,network2,network3,network4]
learningRates = [0.01] #[0.0000001, 0.00001, 0.01, 10]


def trainWithLossAndActivationCombination(loss, activation, networks, colors, labels, imgName, learningRate, epochs = 300):
    fig, axs = plt.subplots(3)
    for i in range(len(networks)):
        network = networks[i]
        network.resetToInitialState()
        results = network.fit(trainFeatures, trainTargets, testFeatures, testTargets, loss[0], loss[1], activation[0], activation[1], epochs, learningRate)

        trainingLossesMean = []
        testingLossesMean = []
        correctness = []

        for res in results:
            trainingLossesMean.append(res[0])
            testingLossesMean.append(res[1])
            correctness.append(res[2])

        axs[0].plot(trainingLossesMean, color=colors[i], label=labels[i])
        axs[0].set_ylim(ymin=0, ymax=5)
        axs[0].set_title("trainingLossesMean")
        axs[1].plot(testingLossesMean, color=colors[i])
        axs[1].set_ylim(ymin=0, ymax=5)
        axs[1].set_title("testingLossesMean")
        axs[2].plot(correctness, color=colors[i])
        axs[2].set_ylim(ymin=0,ymax=1.1)
        axs[2].set_title("correctness")

    axs[0].legend()
    fig.set_size_inches(13,13)
    fig.savefig(imgName)


for lossFunction in fn.lossCombinations:
    for activationFunction in fn.activationCombinations:
        for learningRate in learningRates:
            trainWithLossAndActivationCombination(lossFunction, activationFunction, networks, colors, labels, lossFunction[2] + ' - ' + activationFunction[2] + ' - ' + str(learningRate) + 'R.png', learningRate, epochs=20)



    

# fig, axs = plt.subplots(3)
# for i in range(len(networks)):
#     network = networks[i]
#     results = network.fit(trainFeatures, trainTargets, testFeatures, testTargets, fn.binaryCrossEntropy, fn.dfBinaryCrossEntropy, fn.sigmoid, fn.dfSigmoid, 300, 0.02)

#     trainingLossesMean = []
#     testingLossesMean = []
#     correctness = []

#     for res in results:
#         trainingLossesMean.append(res[0])
#         testingLossesMean.append(res[1])
#         correctness.append(res[2])

#     axs[0].plot(trainingLossesMean, color=colors[i], label=labels[i])
#     axs[0].set_ylim(ymin=0, ymax=5)
#     axs[0].set_title("trainingLossesMean")
#     axs[1].plot(testingLossesMean, color=colors[i])
#     axs[1].set_ylim(ymin=0, ymax=5)
#     axs[1].set_title("testingLossesMean")
#     axs[2].plot(correctness, color=colors[i])
#     axs[2].set_ylim(ymin=0,ymax=1.1)
#     axs[2].set_title("correctness")

# axs[0].legend()
# fig.set_size_inches(13,13)
# plt.show()
# print()










# network1 = Network.Network([4,10,10,10,3])
# results = network1.fit(trainFeatures, trainTargets, testFeatures, testTargets, fn.MSE, fn.dfMSE, fn.sigmoid, fn.dfSigmoid, 100, 0.05)

# trainingLossesMean = []
# testingLossesMean = []
# correctness = []

# for res in results:
#     trainingLossesMean.append(res[0])
#     testingLossesMean.append(res[1])
#     correctness.append([2])

# fig, axs = plt.subplots(3)
# axs[0].plot(trainingLossesMean)
# axs[0].set_ylim(ymin=0)
# axs[0].set_title("trainingLossesMean")
# axs[1].plot(testingLossesMean)
# axs[1].set_ylim(ymin=0)
# axs[1].set_title("testingLossesMean")
# axs[2].plot(correctness)
# axs[2].set_ylim(ymin=0)
# axs[2].set_title("correctness")

# plt.show()



# network1 = Network.Network([4,53,3])
# results = network1.fit(trainFeatures, trainTargets, testFeatures, testTargets, fn.binaryCrossEntropy, fn.dfBinaryCrossEntropy, fn.sigmoid, fn.dfSigmoid, 500, 0.02)

# trainingLossesMean = []
# testingLossesMean = []
# correctness = []

# for res in results:
#     trainingLossesMean.append(res[0])
#     testingLossesMean.append(res[1])
#     correctness.append(res[2])

# fig, axs = plt.subplots(3)
# axs[0].plot(trainingLossesMean)
# axs[0].set_ylim(ymin=0)
# axs[0].set_title("trainingLossesMean")
# axs[1].plot(testingLossesMean)
# axs[1].set_ylim(ymin=0)
# axs[1].set_title("testingLossesMean")
# axs[2].plot(correctness)
# axs[2].set_ylim(ymin=0)
# axs[2].set_title("correctness")

# plt.show()
