# Advanced Course in Machine Learning
# Exercise 3
# Neural Networks

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Flag for training or not
TRAINING = 1
TRAINING_SHUFFLE = 1

###################################################
#                 Exercise 3.a                    #
###################################################

# Copy pasting the pythorch example
# Downloading CIFAR10
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Function for showing the image to learn
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Random training images
#dataiter = iter(trainloader)
#images, labels = dataiter.next()

# Show images and print its labels
#print("Showing a random image")
#imshow(torchvision.utils.make_grid(images))
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


# Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


cnn = CNN()
PATH_CNN = './cifar_cnn.pth'

if TRAINING:
    # Loss funcstion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

    print("Starting to train a CNN Model")
    # Training the network
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = cnn(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training CNN')
    # Save the trained model
    torch.save(cnn.state_dict(), PATH_CNN)

# If we dont train we load the model
else:
    print("Loading a trained model CNN")
    cnn.load_state_dict(torch.load(PATH_CNN))


# Testing the trained model
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth CNN: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
outputs = cnn(images)

# Prediction
_, predicted = torch.max(outputs, 1)

print('Predicted CNN: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))


# Counting the correct predictions
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the CNN-network on the 10000 test images: %d %%' % (
    100 * correct / total))

# Counting the classes that work correctly
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = cnn(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

accuracy_cnn = []
for i in range(10):
    accuracy_cnn.append(100 * class_correct[i] / class_total[i])
    print('Accuracy of %5s : %2d %%' % (
        classes[i], accuracy_cnn[i]))

###################################################
#                 Exercise 3.a                    #
###################################################
total_params_cnn = sum(p.numel() for p in cnn.parameters())
total_params_trainable_cnn = sum(p.numel() for p in cnn.parameters())

# Counting the parameters of the cnn model
print('Total parameters CNN: ', total_params_cnn)
print('Total Trainable parameters CNN: ', total_params_trainable_cnn)


###################################################
#                 Exercise 3.b                    #
###################################################

# Define the MLP Neural Network
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


mlp = MLP()
PATH_MLP = './cifar_mlp.pth'

if TRAINING:
    # Loss funcstion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mlp.parameters(), lr=0.001, momentum=0.9)

    print("Starting to train a MLP Model")
    # Training the network
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = mlp(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training MLP')
    # Save the trained model
    torch.save(mlp.state_dict(), PATH_MLP)

# If we dont train we load the model
else:
    print("Loading a MLP trained model")
    mlp.load_state_dict(torch.load(PATH_MLP))


# Testing the trained model
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth MLP: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
outputs = mlp(images)

# Prediction
_, predicted = torch.max(outputs, 1)

print('Predicted MLP: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))


# Counting the correct predictions
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = mlp(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the MLP-network on the 10000 test images: %d %%' % (
    100 * correct / total))

# Counting the classes that work correctly
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = mlp(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

accuracy_mlp = []
for i in range(10):
    accuracy_mlp.append(100 * class_correct[i] / class_total[i])
    print('Accuracy of %5s : %2d %%' % (
        classes[i], accuracy_mlp[i]))


# Counting the parameters of the MLP model
total_params_mlp = sum(p.numel() for p in mlp.parameters())
total_params_trainable_mlp = sum(p.numel() for p in mlp.parameters())

print('Total parameters for MLP: ', total_params_mlp)
print('Total Trainable parameters for MLP: ', total_params_trainable_mlp)

# Plot comparing both accuracies
x = range(10)
plt.plot(x, [accuracy_cnn[i] for i in x])
plt.plot(x, [accuracy_mlp[i] for i in x])
plt.title('Accuray CNN - Accuracy MLP')
plt.xlabel('Classes')
plt.ylabel('Accuracy')
plt.show()


###################################################
#                 Exercise 3.c                    #
###################################################

# Shuffle the pixels
def shuffle(image):
    size = image.size()
    perm = torch.randperm(size[1] * size[2])
    for idx in range(size[0]):
        image[idx, :, :] =  image[idx, :, :].view(-1)[perm].view(size[1], size[2])
    return image

# Reload the pictures but shuffled
transform_shuffle = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     transforms.Lambda(shuffle)])

trainset_shuffle = torchvision.datasets.CIFAR10(root='./data_shuffle', train=True,
                                        download=True, transform=transform_shuffle)
trainloader_shuffle = torch.utils.data.DataLoader(trainset_shuffle, batch_size=4,
                                          shuffle=True, num_workers=0)

testset_shuffle = torchvision.datasets.CIFAR10(root='./data_shuffle', train=False,
                                       download=True, transform=transform_shuffle)
testloader_shuffle = torch.utils.data.DataLoader(testset_shuffle, batch_size=4,
                                         shuffle=False, num_workers=0)

# Retrain CNN but shuffled

cnn_shuffle = CNN()
PATH_CNN_SHUFFLE = './cifar_cnn_shuffle.pth'

if TRAINING_SHUFFLE:
    # Loss funcstion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn_shuffle.parameters(), lr=0.001, momentum=0.9)

    print("Starting to train a CNN SHUFFLE Model")
    # Training the network
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = cnn_shuffle(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training CNN SHUFFLE')
    # Save the trained model
    torch.save(cnn_shuffle.state_dict(), PATH_CNN_SHUFFLE)

# If we dont train we load the model
else:
    print("Loading a trained model CNN SHUFFLE")
    cnn_shuffle.load_state_dict(torch.load(PATH_CNN_SHUFFLE))


# Testing the trained model
dataiter = iter(testloader_shuffle)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth CNN SHUFFLE: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
outputs = cnn_shuffle(images)

# Prediction
_, predicted = torch.max(outputs, 1)

print('Predicted CNN SHUFFLE: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))


# Counting the correct predictions
correct = 0
total = 0
with torch.no_grad():
    for data in testloader_shuffle:
        images, labels = data
        outputs = cnn_shuffle(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the CNN-network on the 10000 test images: %d %%' % (
    100 * correct / total))

# Counting the classes that work correctly
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader_shuffle:
        images, labels = data
        outputs = cnn_shuffle(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

accuracy_cnn_shuffle = []
for i in range(10):
    accuracy_cnn_shuffle.append(100 * class_correct[i] / class_total[i])
    print('Accuracy of %5s : %2d %%' % (
        classes[i], accuracy_cnn_shuffle[i]))


# Counting the parameters of the cnn_shuffle model
total_params_cnn_shuffle = sum(p.numel() for p in cnn_shuffle.parameters())
total_params_trainable_cnn_shuffle = sum(p.numel() for p in cnn_shuffle.parameters())

print('Total parameters cnn_shuffle: ', total_params_cnn_shuffle)
print('Total Trainable parameters cnn_shuffle: ', total_params_trainable_cnn_shuffle)

# Retrain MLP but shuffled
mlp_shuffle = MLP()
PATH_MLP_SHUFFLE = './cifar_mlp_shuffle.pth'

if TRAINING_SHUFFLE:
    # Loss funcstion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mlp_shuffle.parameters(), lr=0.001, momentum=0.9)

    print("Starting to train a MLP_SHUFFLE Model")
    # Training the network
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = mlp_shuffle(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training MLP_SHUFFLE')
    # Save the trained model
    torch.save(mlp_shuffle.state_dict(), PATH_MLP_SHUFFLE)

# If we dont train we load the model
else:
    print("Loading a MLP_SHUFFLE trained model")
    mlp_shuffle.load_state_dict(torch.load(PATH_MLP_SHUFFLE))


# Testing the trained model
dataiter = iter(testloader_shuffle)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth MLP SHUFFLE: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
outputs = mlp_shuffle(images)

# Prediction
_, predicted = torch.max(outputs, 1)

print('Predicted MLP SHUFFLE: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))


# Counting the correct predictions
correct = 0
total = 0
with torch.no_grad():
    for data in testloader_shuffle:
        images, labels = data
        outputs = mlp_shuffle(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the MLP SHUFFLE-network on the 10000 test images: %d %%' % (
    100 * correct / total))

# Counting the classes that work correctly
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader_shuffle:
        images, labels = data
        outputs = mlp_shuffle(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

accuracy_mlp_shuffle = []
for i in range(10):
    accuracy_mlp_shuffle.append(100 * class_correct[i] / class_total[i])
    print('Accuracy of %5s : %2d %%' % (
        classes[i], accuracy_mlp_shuffle[i]))


# Counting the parameters of the MLP model
total_params_mlp_shuffle = sum(p.numel() for p in mlp_shuffle.parameters())
total_params_trainable_mlp_shuffle = sum(p.numel() for p in mlp_shuffle.parameters())

print('Total parameters for MLP SHUFFLE: ', total_params_mlp_shuffle)
print('Total Trainable parameters for MLP SHUFFLE: ', total_params_trainable_mlp_shuffle)

# Plot comparing both accuracies
x = range(10)
plt.plot(x, [accuracy_cnn_shuffle[i] for i in x])
plt.plot(x, [accuracy_mlp_shuffle[i] for i in x])
plt.title('Accuray CNN SHUFFLE - Accuracy MLP SHUFFLE SHUFFLE')
plt.xlabel('Classes')
plt.ylabel('Accuracy')
plt.show()


###################################################
#                 Exercise 3.d                    #
###################################################

def gen_adv(model, dataloader, epsilon, confidence_threshold=0.8):
    def predict(X):
        return F.softmax(model(X), dim=1).max(1)
    criterion = nn.CrossEntropyLoss()

    # Find a correct, high-confidence input and prediction
    for data in dataloader:
        inputs, label = data
        outputs = predict(inputs)
        p = outputs.values > confidence_threshold
        # If high-confidence sample
        if p.any():
            match = False
            for i, (pred, true_label) in enumerate(zip(outputs.indices[p], label[p])):
                if pred == true_label:  # Found a matching prediction with high confidence
                    match = True
                    break
            if not match:
                continue
            inputs = inputs[p][i]
            label = outputs.indices[p][i]
            break
    if not p.any():
        raise RuntimeError("Could not find high confidence correct prediction!")
    # Reshape and copy as needed
    inputs = inputs.view(1, 3, 32, 32).clone().detach().requires_grad_(True)
    label = torch.Tensor([label]).long()
    outputs = model(inputs)
    loss = criterion(outputs, label)
    loss.backward()
    # Create the adversarial example as derived from the optimization problem
    adv = inputs + epsilon * inputs.grad.detach().sign()
    return label, (inputs, predict(inputs)), (adv, predict(adv))

epsilon = 0.08
label, (im, im_label), (adv, adv_label) = gen_adv(cnn, testloader, epsilon)

def advimg(img):
    img = img / 2 + 0.5
    npimg = np.clip(np.squeeze(img.detach().numpy()), a_min=0, a_max=1)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


plt.figure(figsize=(12, 5))
plt.subplot(131)
advimg(im)
plt.axis('off')
plt.title("Original image")
plt.subplot(132)
advimg(adv - im)
plt.axis('off')
plt.title("Added noise".format(epsilon))
plt.subplot(133)
advimg(adv)
plt.axis('off')
plt.title("Adversarial image")
plt.suptitle("Adversarial true class")
plt.tight_layout(pad=2)
plt.show()
