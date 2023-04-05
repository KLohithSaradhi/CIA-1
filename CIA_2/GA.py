import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report


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




class GA:
    def __init__(self, model, X, Y, mutation_rate = 1):
        
        self.model = model
        self.X = X
        self.Y = Y
        self.individualSize = self.model.getWeightVector().shape[0]
        self.mutationRate = mutation_rate
        
    def generateFirstPopulation(self, populationSize = 10):
        self.populationSize = populationSize
        self.POPULATION = 10 * np.random.randn(self.populationSize, self.individualSize)     
        
    
    def evaluatePopulation(self):
        self.scores = []
        for individual in self.POPULATION:
            
            self.model.setLayerWeights(individual)
            pred = np.round(self.model.forwardpass(self.X))
            self.scores += [accuracy_score(self.Y, pred)]            
        return self.scores
    
    def selectNextPopulationRoulette(self):
        self.evaluatePopulation()
        temp = self.POPULATION.copy()
        
        for _ in range(self.populationSize):
            self.scores = self.scores/np.sum(self.scores)

            ## cumulative sum
            roulette = np.cumsum(self.scores)

            ## if fitnessvalue of an individual is large, there is a larger chunk allocated for it in the range of 0-1,
            ## hence proportional probability for higher fitness individuals to get selected.
            newPopIndex = []
            for count in range(self.populationSize):
                roulette_ball = np.random.random()
                newPopIndex += [(roulette >= roulette_ball).tolist().index(True)]
                
        self.POPULATION = temp[newPopIndex]
        
    def selectNextPopulation(self):
        self.evaluatePopulation()
        
        nextPopIndices = np.argsort(self.scores)[::-1][:self.populationSize]
        
        self.POPULATION = self.POPULATION[nextPopIndices]
            
         
    def mutate(self, ind):
        if np.random.random() > (1-self.mutationRate):
            index = np.random.randint(self.individualSize)
            ind[index] = np.random.randint(100) * np.random.rand()
            
        return ind
            
    def crossOver(self, ind1, ind2):
        splitPoint = np.random.randint(self.individualSize)
        
        child1 = np.concatenate((ind1[:splitPoint], ind2[splitPoint:]))
        child2 = np.concatenate((ind2[:splitPoint], ind1[splitPoint:]))
        
        
        child1 = self.mutate(child1)
        child2 = self.mutate(child2)
        
        return np.array([child1, child2])
    
    def reproduce(self):
        for p1 in range(self.populationSize):
            for p2 in range(self.populationSize):
                
                if p1 != p2:
                    self.POPULATION = np.concatenate((self.POPULATION, self.crossOver(self.POPULATION[p1], self.POPULATION[p2])))
                 
    


#%%

import pandas as pd

data = pd.read_csv("Bank_Personal_Loan_Modelling.csv")
data = data.drop(["ID", "ZIP Code"], axis = 1)
X = data.drop("Personal Loan", axis = 1).values
Y = data["Personal Loan"].values.reshape(-1,1)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state=257)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
#%%

model = NN([11, 6, 1], [nn.GELU, nn.Sigmoid])
model.forwardpass(x_train)
#%%
epochs = 100

ga = GA(model, x_train, y_train, mutation_rate=1)
ga.generateFirstPopulation(30)
#%%
epoch = 1
acc = max(ga.evaluatePopulation())
prevacc = 0
while epoch < epochs:
    prevacc = acc
    ga.reproduce()
    ga.selectNextPopulation()
    acc = max(ga.evaluatePopulation())

    epoch += 1
    print("epoch : ",epoch," : ",acc)

#%%    
bestWeights = ga.POPULATION[0]
    
model.setLayerWeights(bestWeights)

pred = np.round(model.forwardpass(x_test))

print(accuracy_score(pred, y_test))
    
#%%
print(classification_report(pred, y_test))
    
                    
                
                    
                    
            
        
        
        
        
            
    
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        