
"""
3.Transfer Learning on Greek Letters
The goal of this step is to re-use the the MNIST digit recognition network you built in 
step 1 to recognize three different greek letters: alpha, beta, and gamma.

Submitted by : Gmon Kuzhiyanikkal
NU ID: 002724506
Date: 14/03/2023
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define the neural network
class Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

# Define the data loaders and transformations
batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.1307,), (0.3081,))])

train_set = MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = MNIST('./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Define the optimizer and loss function
learning_rate = 0.001
net = Net()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()



# Train the MNIST network and plot the training error
epochs = 5
train_losses = []
for epoch in range(1, epochs + 1):
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    train_losses.append(running_loss / len(train_loader))

# Plot the training error
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Save the pre-trained weights to a file
torch.save(net.state_dict(), 'mnist_net.pt')

# Load the pre-trained weights into the network
net = Net()
net.load_state_dict(torch.load('mnist_net.pt'))

# Freeze the network weights
for param in net.parameters():
    param.requires_grad = False

# Replace the last layer with a new Linear layer with three nodes
net.fc2 = nn.Linear(128, 3)

#print the new modified network
print("\nmodified network is:\n")
print(net)

# Define the new data loaders and transformations for the greek letters
greek_transform = transforms.Compose([transforms.Grayscale(),
                                      transforms.Resize((28, 28)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])

greek_set = ImageFolder('./greek_train', transform=greek_transform)
greek_loader = DataLoader(greek_set, batch_size=1, shuffle=False)



# Test the network on the Greek letters
net.eval()
greek_names = ['alpha', 'beta', 'gamma']
with torch.no_grad():
    for i, (data, target) in enumerate(greek_loader):
        output = net(data)
        pred = torch.argmax(output, dim=1)
        print(f"Predicted class: {pred.item()}, Greek letter: {greek_names[pred.item()]}")

