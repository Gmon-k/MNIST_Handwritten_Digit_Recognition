
"""
2.Experiment with Network Variations
The next task is to undertake some experimentation with the 
deep network for the MNIST task

###Varying the activation function for each layers, keeping others constants

Submitted by : Gmon Kuzhiyanikkal
NU ID: 002724506
Date: 14/03/2023

"""


import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

batch_size = 128

# Load the dataset
train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define the hyperparameters
num_filters = 32
filter_size = (3, 3)
pool_size = (2, 2)
hidden_layer_size = 128
num_classes = 10
batch_size = 128

# Define the function to build and compile the model
def build_model(activation):
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, num_filters, filter_size),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(pool_size),
        torch.nn.Flatten(),
        torch.nn.Linear(num_filters*13*13, hidden_layer_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_layer_size, num_classes),
        torch.nn.Softmax(dim=1)
    )
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, loss_function, optimizer

# Define the activation functions to evaluate
activations = ['relu', 'sigmoid', 'tanh']

# Evaluate the effect of changing the activation function for each layer
for activation in activations:
    model, loss_function, optimizer = build_model(activation)
    print(f"\nActivation function: {activation}\n")
    for epoch in range(10):
        train_loss = 0
        train_correct = 0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_correct += (predicted == target).sum().item()

        test_loss = 0
        test_correct = 0
        model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                loss = loss_function(output, target)
                test_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                test_correct += (predicted == target).sum().item()

        train_accuracy = train_correct / len(train_dataset)
        test_accuracy = test_correct / len(test_dataset)
        print(f" Epoch: {epoch+1}, Train loss: {train_loss/len(train_loader)}, Train accuracy: {train_accuracy}, Test loss: {test_loss/len(test_loader)}, Test accuracy: {test_accuracy}")
