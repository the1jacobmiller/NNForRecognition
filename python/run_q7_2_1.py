import numpy as np
import scipy.io
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

batch_size = 64
max_iters = 15
max_iters2 = 3

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

# Load pretrained SqueezeNet
model = torchvision.models.squeezenet1_0(pretrained=True)
in_features = 86528

# Reinitialize the last layer of the model
num_classes = len(train_set.classes)
model.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(in_features, num_classes)
                    )

# Create a loss function
# model.type(dtype)
loss_fn = nn.CrossEntropyLoss()

# Set requires_grad=True for the parameters in the last layer only
for param in model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True

# Construct an Optimizer object for updating the last layer only.
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

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
    epoch_accuracy.append(acc*100.)
    epoch_losses.append(running_loss/len(train_set))

print('Finished Training')
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(epoch_counter, epoch_accuracy, color='red')
plt.legend(['Train Loss', 'Train Acc. (%)'], loc='upper right')
plt.xlabel('Training data points')
fig
plt.show()

# set up model to fine tune
for param in model.parameters():
    param.requires_grad = True

# Construct a new Optimizer that will update all model parameters
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

print("Fine tuning network")
train_counter = []
train_losses = []
epoch_counter = []
epoch_accuracy = []
epoch_losses = []
samples = 0
for epoch in range(max_iters2):  # loop over the dataset multiple times
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
    epoch_accuracy.append(acc*100.)
    epoch_losses.append(running_loss/len(train_set))

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(epoch_counter, epoch_accuracy, color='red')
plt.legend(['Train Loss', 'Train Acc.'], loc='upper right')
plt.xlabel('Training data points')
fig
plt.show()
