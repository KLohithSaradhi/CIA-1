import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


class Layer : 
    def __init__(self, n_in, n_out, activation_function = None):
        self.weights = np.random.random((n_out, n_in + 1))
        self.activation_function = activation_function()
        
    def forwardpass(self, X):
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis = 1)
        if self.activation_function:
            
            temp = torch.tensor((X @ self.weights.T).astype(np.float32))
            return self.activation_function(temp).numpy()
        else:
            return X @ self.weights.T
        
    

class NN:
    def __init__(self, layers, activation_functions = None):
        if activation_functions is None:
            self.layerSizes = []
            self.layers = []        
            for _ in range(len(layers) - 1):
                self.layers += [Layer(layers[_], layers[_ + 1])]
                self.layerSizes += [(layers[_] + 1, layers[_ + 1])]
        else:
            self.layerSizes = []
            self.layers = []        
            for _ in range(len(layers) - 1):
                self.layers += [Layer(layers[_], layers[_ + 1], activation_functions[_])]
                self.layerSizes += [(layers[_] + 1, layers[_ + 1])]
    
    def forwardpass(self, X):
        for layer in self.layers:
            X = layer.forwardpass(X)
        return X
    
    def getWeightVector(self):
        weightVector = []
        for layer in self.layers:
            weightVector += [layer.weights.flatten()]
        weightVector = np.concatenate(weightVector)
        return weightVector
    
    def setLayerWeights(self, weightVector):
        start = 0
        for l in range(len(self.layerSizes)):
            inputSize, outputSize = self.layerSizes[l]
            weights = weightVector[start:start + inputSize*outputSize]
            weights = weights.reshape((outputSize, inputSize))
            self.layers[l].weights = weights
            start += inputSize * outputSize


class ACO:
    def __init__(self, model, X, Y):
        self.model = model
        self.X = X
        self.Y = Y
        
        self.individualSize = self.model.getWeightVector().shape[0]
        
    def generateFirstBatch(self, populationSize):
        self.populationSize = populationSize
        means, stds = np.random.randint(100,  size = self.populationSize), np.random.randint(100,  size = self.populationSize)
        
        self.ANTS = []
        for _ in range(self.populationSize):
            self.ANTS += [means[_] + stds[_]*np.random.randn(self.individualSize)]
        self.ANTS = np.array(self.ANTS)
            
    def evaluateBatch(self):
        scores = []
        for ant in self.ANTS:
            self.model.setLayerWeights(ant)
            pred = np.round(self.model.forwardpass(self.X))
            scores += [accuracy_score(self.Y, pred)]
        
        return np.array(scores)
    
    def calcPheremonesForEachAnt(self):
        scores = self.evaluateBatch()
        self.pheremones = scores/sum(scores)
        self.STDS = 1/scores
    
    def roulette(self, probDist):
        rouletteWheel = np.cumsum(probDist)
        
        roulleteBall = np.random.random()
        
        return np.where(rouletteWheel >= roulleteBall)[0][0]
    
    def nextBatch(self):
        
        next_batch = []
        for antIndex in range(self.populationSize):
            next_batch += [[0 for _ in range(self.individualSize)]]
            for dimension in range(self.individualSize):
                toBeFollowedAntIndex = self.roulette(self.pheremones)
                
                next_batch[antIndex][dimension] = self.ANTS[toBeFollowedAntIndex][dimension] + self.STDS[toBeFollowedAntIndex]*np.random.randn()
                
        next_batch = np.array(next_batch)
        self.ANTS = next_batch.copy()
        
#%%
import pandas as pd

data = pd.read_csv("Bank_Personal_Loan_Modelling.csv")
data = data.drop(["ID", "ZIP Code"], axis = 1)
X = data.drop("Personal Loan", axis = 1).values
Y = data["Personal Loan"].values.reshape(-1,1)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
#%%

model = NN([11, 6, 1], [nn.GELU, nn.Sigmoid])
model.forwardpass(x_train)
#%%
epochs = 100
aco = ACO(model, x_train, y_train)

aco.generateFirstBatch(100)
#%%

for epoch in range(epochs):
    aco.calcPheremonesForEachAnt()
    aco.nextBatch()
    
    l = max(aco.evaluateBatch())
    print(f"epoch : {epoch}  :  {l}")
                
        
                


                
            


















