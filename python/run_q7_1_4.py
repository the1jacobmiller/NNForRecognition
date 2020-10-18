import numpy as np
import scipy.io
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

max_iters = 10
batch_size = 64
learning_rate = 0.0001

# load dataset
transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor()])

torchvision.datasets.EMNIST.url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'
train_set = torchvision.datasets.EMNIST(root='../data', split='balanced', download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

# Declare network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(1600, 1000)
        self.fc2 = nn.Linear(1000, 47)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.dropout(x)
        x = x.view(-1, 1600)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Initialize loss function and optimization technique
net = Net()
criterion = nn.NLLLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# Train
print("Training network")
train_counter = []
train_losses = []
epoch_counter = []
epoch_accuracy = []
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


print('Finished Training')

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(epoch_counter, epoch_accuracy, color='red')
plt.legend(['Train Loss', 'Train Acc.'], loc='upper right')
plt.xlabel('Training data points')
fig
plt.show()

torch.save(net.state_dict(), 'emnist_model.pt')
