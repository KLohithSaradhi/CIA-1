import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


class NN(nn.Module):
    
    def __init__(self):
        super(NN, self).__init__()
        
        self.l1 = nn.Linear(11, 6)
        self.l2 = nn.Linear(6, 1)
        
        self.gelu = nn.GELU()
        self.sig = nn.Sigmoid()
        
    def forward(self, X):
        X = self.gelu(self.l1(X))
        X = self.sig(self.l2(X))
        
        return X
    
    
#%%
import pandas as pd
import numpy as np

data = pd.read_csv("Bank_Personal_Loan_Modelling.csv")
data = data.drop(["ID", "ZIP Code"], axis = 1)
X = data.drop("Personal Loan", axis = 1).values
Y = data["Personal Loan"].values.reshape(-1,1)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state=257)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = torch.from_numpy(sc.fit_transform(x_train).astype(np.float32))
x_test = torch.from_numpy(sc.transform(x_test).astype(np.float32))

y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))
#%%

model = NN()

epochs = 100

loss = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

#%%

for epoch in range(epochs):
    output = model(x_train)
    
    l = loss(output, y_train)
    l.backward()
    
    optimizer.step()
    optimizer.zero_grad()
    with torch.no_grad():

        pred = np.round(model(x_test).numpy())
        acc = accuracy_score(pred, y_test.numpy())
    
    print(f"epoch : {epoch}, loss : {acc}")
    
#%%

