"""
1.MNIST Tutorial
Go through one or more tutorials on building a convolutional neural network
for the MNIST digit recognition task

Submitted by : Gmon Kuzhiyanikkal
NU ID: 002724506
Date: 14/03/2023

"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
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

net = Net()

# Define the optimizer and loss function
learning_rate = 0.001
net = Net()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Define the training function
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader)
    return train_loss

# Define the testing function
def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    return test_loss

# Initialize lists to store training and test losses
train_losses = []
test_losses = []

# Train the model
epochs = 10
for epoch in range(1, epochs + 1):
    train_loss = train(net, train_loader, criterion, optimizer, epoch)
    test_loss = test(net, test_loader, criterion)
    train_losses.append(train_loss)
    test_losses.append(test_loss)


# Plot example outputs and their labels
with torch.no_grad():
    data, target = next(iter(test_loader))
    output = net(data)
    _, preds = torch.max(output, dim=1)
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    for i, ax in enumerate(axs.flat):
        ax.imshow(data[i].squeeze(), cmap='gray')
        ax.set_title(f'Pred: {preds[i]}, Label: {target[i]}')
        ax.axis('off')
    plt.show()


# Plot the training and test losses
plt.plot(range(1, epochs+1), train_losses, 'b-', label='Training Loss')
plt.plot(range(1, epochs+1), test_losses, 'r-', label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Printing the decpition
print(net)
