import numpy as np
import scipy.io
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

max_iters = 20
batch_size = 50
learning_rate = 0.001

# load dataset
transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize((0.1307,), (0.3081,))])

train_set = torchvision.datasets.MNIST('../data', train=True, download=True,
                             transform=transform)
valid_set = torchvision.datasets.MNIST('../data', train=False, download=True,
                             transform=transform)

trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
validloader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

# Declare network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.drop_out = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.drop_out(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Initialize loss function and optimization technique
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

# Train
print("Training network")
train_counter = []
train_losses = []
epoch_counter = []
epoch_accuracy = []
epoch_loss = []
samples = 0
for epoch in range(max_iters):  # loop over the dataset multiple times
    running_loss = 0.
    correct = 0
    for batch_count, data in enumerate(trainloader, 0):
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        predicted = torch.max(outputs.data,1)[1]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        correct += (predicted == labels).sum().item()

        samples += len(data)
        if batch_count%25==0:
            train_counter.append(samples)
            train_losses.append(loss.item()/len(data))

    acc = correct/len(train_set)
    print("End of epoch",epoch)
    print("\tAccuracy:", acc)
    print("\tLoss:", running_loss/len(train_set))

    epoch_counter.append(samples)
    epoch_accuracy.append(acc)
    epoch_loss.append(running_loss/len(train_set))


print('Finished Training')

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(epoch_counter, epoch_accuracy, color='red')
plt.legend(['Train Loss', 'Train Acc.'], loc='upper right')
plt.xlabel('Training data points')
fig
plt.show()
