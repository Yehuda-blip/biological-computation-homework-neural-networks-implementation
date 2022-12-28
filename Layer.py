import numpy as np
import random

class Layer:
    def __init__(self, size, nextLayerSize, activeEdges = None, dense = True) -> None:
        self.size = size
        self.beforeActivation = np.zeros((size,))
        self.dfActivation = np.zeros((size,))
        self.nodeVals = np.zeros((size,))
        self.backDelta = np.zeros((size,))
        self.edges = np.zeros((nextLayerSize,size))
        self.activeEdges = activeEdges
        self.isDense = dense

        # rand init edges
        if dense:
            for i in range(len(self.edges)):
                for j in range(len(self.edges[0])):
                    self.edges[i,j] = random.uniform(-1,1)
        else:
            for edge in activeEdges:
                i, j = edge
                self.edges[j,i] = random.uniform(-1,1)
