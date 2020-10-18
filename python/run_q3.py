import numpy as np
import scipy.io
from nn import *

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 50
batch_size = 20
learning_rate = 0.01

input_size = 1024
hidden_size = 64
output_size = 36

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
initialize_weights(input_size,hidden_size,params,'layer1')
initialize_weights(hidden_size,output_size,params,'output')

# with default settings, you should get loss < 150 and accuracy > 80%
train_losses = []
train_accuracy = []
for itr in range(max_iters):
    total_loss = 0.
    total_acc = 0.
    for xb,yb in batches:
        # forward
        h1 = forward(xb,params,'layer1')
        probs = forward(h1,params,'output',softmax)

        # loss
        # be sure to add loss and accuracy to epoch totals
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        total_acc += acc

        # backward
        delta1 = probs
        delta1[np.arange(probs.shape[0]),np.argmax(yb,axis=1)] -= 1
        delta2 = backwards(delta1,params,'output',linear_deriv)
        grad_X = backwards(delta2,params,'layer1',sigmoid_deriv)

        # apply gradient
        params['W' + 'layer1'] = params['W' + 'layer1'] - learning_rate*params['grad_W' + 'layer1']
        params['b' + 'layer1'] = params['b' + 'layer1'] - learning_rate*params['grad_b' + 'layer1']
        params['W' + 'output'] = params['W' + 'output'] - learning_rate*params['grad_W' + 'output']
        params['b' + 'output'] = params['b' + 'output'] - learning_rate*params['grad_b' + 'output']

    total_loss /= batch_num
    total_acc /= batch_num
    train_losses.append(total_loss)
    train_accuracy.append(total_acc*100.)
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

fig = plt.figure()
plt.plot(range(max_iters), train_losses, color='blue')
plt.scatter(range(max_iters), train_accuracy, color='red')
plt.legend(['Train Loss', 'Train Acc.'], loc='upper right')
plt.xlabel('Training data points')
fig
plt.show()

# run on validation set and report accuracy! should be above 75%
correct = 0
for i in range(len(valid_x)):
    x,y = valid_x[i],valid_y[i]
    # forward
    h1 = forward(x,params,'layer1')
    probs = forward(h1,params,'output',softmax)
    pred = np.argmax(probs)
    actual = np.argmax(y)
    correct += int(pred == actual)
valid_acc = correct/len(valid_x)

print('Validation accuracy: ',valid_acc)
if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.1.3
fig = plt.figure(1, (8., 8.))
if hidden_size < 128:
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.
                    )
    img_w = params['Wlayer1'].reshape((32,32,hidden_size))
    for i in range(hidden_size):
        grid[i].imshow(img_w[:,:,i])  # The AxesGrid object work as a list of axes.

    plt.show()

# Q3.1.4
fig = plt.figure(1, (6., 8.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(12, 6),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

indices = params['cache_output'][2].argmax(axis=0)
images = valid_x[indices]
images = images.reshape(1, 32, 32)

vis = np.zeros((36, 1024))
inps = np.eye(36)
for i,inp in enumerate(inps):
    vis[i] = inp @ params['Woutput'].T @ params['Wlayer1'].T
vis = vis.reshape(36, 32, 32)

displayed = np.zeros((72, 32, 32))
displayed[::2] = images
displayed[1::2] = vis
for ax, im in zip(grid, displayed):
    ax.imshow(im.T)
plt.savefig("out.jpg")
plt.show()

# Q3.1.5
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

# compute comfusion matrix here
for i in range(len(valid_x)):
    x,y = valid_x[i],valid_y[i]
    # forward
    h1 = forward(x,params,'layer1')
    probs = forward(h1,params,'output',softmax)
    pred = np.argmax(probs)
    actual = np.argmax(y)
    confusion_matrix[actual,pred] += 1
print(confusion_matrix)

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()
