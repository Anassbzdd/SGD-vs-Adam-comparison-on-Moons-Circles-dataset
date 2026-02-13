import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons , make_circles

# MOONS 
# Generate Dataset
X , y = make_moons(n_samples = 1000 , noise = 0.2 , random_state = 42)
X_train = torch.FloatTensor(X)
y_train = torch.LongTensor(y)

# Visualize the data 
plt.scatter(X[:,0], X[:,1], c=y)
plt.title('The Moons Dataset - Can your optimizer solve this?')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig('dataset.png')
plt.show()

# Build MLP 
class Moons_mlp(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )

    def forward(self, x):
        return self.layer(x)
    
# Train the Model
def train_model(model , optimizer_name , lr = 0.01 , epochs = 100):
    loss = nn.CrossEntropyLoss()

    if optimizer_name == 'SGD' :
        optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    if optimizer_name == 'Adam' :
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    losses = []

    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train)
        loss_fnct = loss(outputs , y_train)

        # Backward pass
        optimizer.zero_grad()
        loss_fnct.backward()
        optimizer.step()

        # Append the Loss
        losses.append(loss_fnct.item())

        if epoch % 10 == 0:
            print(f"{optimizer_name} - Epoch {epoch}, Loss: {loss_fnct.item():.4f}")
        
    return losses

# Test SGD
model_SGD = Moons_mlp()
SGD_losses = train_model(model_SGD, 'SGD', 0.01, 100)

# Test Adam 
model_Adam = Moons_mlp()
Adam_losses= train_model( model_Adam , 'Adam', 0.01, 100)

# Visualize the data 
plt.figure(figsize= (10, 5))
plt.plot(SGD_losses , label= 'SGD / lr = 0.01')
plt.plot(Adam_losses , label= 'Adam / lr = 0.01')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('SGD VS Adam: Who Wins?')
plt.legend()
plt.grid(True)
plt.savefig('optimizer_comparison.png')
plt.show()

# Final Result
print(f"Final SGD loss : {SGD_losses[-1]:.4f}")
print(f"Final Adam loss : {Adam_losses[-1]:.4f}")


# Circles
# Generate Dataset
X , y = make_circles(n_samples=1000, noise = 0.2, random_state = 42)
X_train = torch.FloatTensor(X)
y_train = torch.LongTensor(y)

# Visualize the data 1
plt.scatter(X[:,0], X[:,1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('The Moons Dataset - Can your optimizer solve this?')
plt.savefig('scatter.png')
plt.show()

# Model
class Circles_mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )

    def forward(self , x):
        return self.layer(x)
    
def train_model(model , optimizer_name , lr = 0.01 , epochs = 100):
    loss = nn.CrossEntropyLoss()

    if optimizer_name == 'SGD' :
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    if optimizer_name == 'Adam' : 
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []

    for epoch in range(epochs):
        # Forward pass
        output = model(X_train)
        loss_fnct = loss(output, y_train)

        # Backward pass
        optimizer.zero_grad()
        loss_fnct.backward()
        optimizer.step()

        # Append loss
        losses.append(loss_fnct.item())

        if epoch % 10 == 0:
            print(f"{optimizer_name} - Epoch {epoch}, Loss: {loss_fnct.item():.4f}")

    return losses

model_SGD = Circles_mlp()
SGD_losses = train_model(model_SGD, 'SGD', 0.01, 200)

# Test Adam 
model_Adam = Circles_mlp()
Adam_losses= train_model( model_Adam , 'Adam', 0.01, 200)

# Visualize the data 
plt.figure(figsize= (10, 5))
plt.plot(SGD_losses , label= 'SGD / lr = 0.01')
plt.plot(Adam_losses , label= 'Adam / lr = 0.01')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('SGD VS Adam: Who Wins?')
plt.legend()
plt.grid(True)
plt.savefig('optimizer_comparison.png')
plt.show()

# Final Result
print(f"Final SGD loss : {SGD_losses[-1]:.4f}")
print(f"Final Adam loss : {Adam_losses[-1]:.4f}")





