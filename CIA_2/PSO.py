import numpy as np
from sklearn.metrics import accuracy_score,classification_report
import torch.nn as nn
import torch


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

#decay functions
class Linear:
    def __init__(self, decayvalue = 0.01):
        self.decayvalue = decayvalue
        
    def __call__(self, x):
        return x - 0.01
        
def normalize(vector):
    
    for i in range(vector.shape[0]):
        den = np.sqrt(np.sum(vector[i] ** 2))
        if den != 0 and den != np.nan:
            vector[i] /= den
            return vector
        else:
            return np.zeros(vector.shape)
        
class PSO:
    def __init__(self, model, X, Y, w = 1, c1 = 0.5, c2 = 0.5, decayFunc = Linear, decayvalue = 0.01):
        self.model = model
        self.X = X
        self.Y = Y
        
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        self.decayFunc = Linear(decayvalue)
        
    def evaluateSwarm(self):
        scores = []
        for particle in self.swarm:
            self.model.setLayerWeights(particle)
            pred = np.round(self.model.forwardpass(self.X))
            scores += [accuracy_score(self.Y, pred)]
        
        return np.array(scores)
        
    def generateSwarm(self, swarmSize = 100):
        self.vectorSize = self.model.getWeightVector().shape[0]
        self.swarmSize = swarmSize
        
        self.swarm = 100 * np.random.randn(self.swarmSize, self.vectorSize)
        
        self.inertia = np.random.random((self.swarmSize, self.vectorSize))
        
        self.personalBest = self.swarm.copy()
        self.personalBestScores = self.evaluateSwarm()
        
        self.globalBestScore = max(self.personalBestScores)
        self.globalBest = self.swarm[np.where(self.personalBestScores == self.globalBestScore)[0][0]]
        
    def updateSwarm(self, alpha = 1):
        self.vpb = self.personalBest - self.swarm
        self.vgb = self.globalBest - self.swarm
        
        #self.inertia = normalize(self.inertia) 
        #self.vpb = normalize(self.vpb)
        #self.vgb = normalize(self.vgb)
        
        r1, r2 = np.random.rand(2)
        self.newVector = (self.w * self.inertia + self.c1 * r1 * self.vpb + self.c2 * r2 * self.vgb) *  alpha
        
        self.swarm += self.newVector
        
        self.w = self.decayFunc(self.w)
        
        self.inertia = self.newVector
        
    def updatePBandGB(self):
        currentScores = self.evaluateSwarm()
        for _ in range(len(currentScores)):
            if self.personalBestScores[_] < currentScores[_]:
                self.personalBestScores[_] = currentScores[_]
                self.personalBest[_] = self.swarm[_]
                
        if max(self.personalBestScores) > self.globalBestScore:
                
            self.globalBestScore = max(self.personalBestScores)
            self.globalBest = self.swarm[np.where(self.personalBestScores == self.globalBestScore)[0][0]]
         
    def calculateAccuracy(self):
        accScores = []
        for particle in self.swarm:
            self.model.setLayerWeights(particle)
            pred = np.round(self.model.forwardpass(self.X))
            accScores += [accuracy_score(self.Y, pred)]
        return min(accScores), np.where(accScores == min(accScores))[0][0]
            
        
        
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
pso = PSO(model, x_train, y_train, decayvalue = 1/epochs)

pso.generateSwarm(1000)
#%%
alpha = 1
epoch = 0
l = max(pso.evaluateSwarm())
prevl = 0
while epoch<epochs:
    prevl = l
    pso.updateSwarm(alpha)
    pso.updatePBandGB()
    l = max(pso.evaluateSwarm())
    
    epoch += 1
    print("epoch : ",epoch," : ",l)

    
#%%
bestWeights = pso.globalBest
    
model.setLayerWeights(bestWeights)

pred = np.round(model.forwardpass(x_test))

print(accuracy_score(pred, y_test))
    
#%%
print(classification_report(pred, y_test))









