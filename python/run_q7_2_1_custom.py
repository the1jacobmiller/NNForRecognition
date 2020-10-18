import numpy as np
import scipy.io
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.datasets import ImageFolder

batch_size = 64
max_iters = 15

# load data
train_transform = T.Compose([
    T.Resize(256),
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
])

train_set = ImageFolder('../data/oxford-flowers17', transform=train_transform)
trainloader = DataLoader(train_set,
                batch_size=batch_size,
                num_workers=2,
                shuffle=True)
num_classes = len(train_set.classes)

# Declare network architecture
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(128*26*26, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.dropout(x)
        x = x.view(-1, 128*26*26)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize loss function and optimization technique
model = CustomModel()

# Create a loss function
loss_fn = nn.CrossEntropyLoss()

# Construct an Optimizer object
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# Train
print("Training network")
train_counter = []
train_losses = []
epoch_counter = []
epoch_accuracy = []
epoch_losses = []
samples = 0
for epoch in range(max_iters):  # loop over the dataset multiple times
    running_loss = 0.
    correct = 0
    for batch_count, data in enumerate(trainloader, 0):
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        predicted = torch.max(outputs.data,1)[1]
        loss = loss_fn(outputs, labels)
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
    epoch_losses.append(running_loss/len(train_set))

print('Finished Training')
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(epoch_counter, epoch_accuracy, color='red')
plt.legend(['Train Loss', 'Train Acc.'], loc='upper right')
plt.xlabel('Training data points')
fig
plt.show()
