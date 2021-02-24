#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import argparse
from PIL import Image

def load_data(batch_size=128):
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='./data.cifar10', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(root='./data.cifar10', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return train_loader, test_loader, classes


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(3*32*32, 100)
        self.fc2 = torch.nn.Linear(100, 25)
        self.dropout = torch.nn.Dropout(0.1)
        self.output = torch.nn.Linear(25, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.output(x)
        return x


def test(net, inputs, device, criterion=None, flag='test'):
    net.eval()
    if flag == 'inference':
        with torch.no_grad():
            image = inputs.view(-1, np.array(inputs.size()).prod()).to(device)
            output = net(image)
            _, predicted = torch.max(output.data, 1)
            return predicted.int()
    else:
        correct = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for data in inputs:
                images, labels = data[0].view(-1, np.array(data[0].size()[1:]).prod()).to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total_loss += criterion(outputs, labels)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total, total_loss/len(inputs)


def train(device, total_epoch):
    train_loader, test_loader, classes = load_data()
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    net.to(device)
    best_test_accuracy = 0
    for epoch in range(total_epoch):
        for i, data in enumerate(train_loader, 0):
            net.train()
            # get the inputs; data is a list of [inputs, labels]
            inputs = data[0].view(-1, np.array(data[0].size()[1:]).prod()).to(device)
            labels = data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        train_accuracy, train_loss = test(net, train_loader, device, criterion)
        test_accuracy, test_loss = test(net, test_loader, device, criterion)
        print("### Epoch [{}/{}]\t\ttrain accuracy: {}%\t\ttrain loss: {}\t\ttest accuracy: {}%\t\ttest loss:{}"
              .format(epoch+1,total_epoch, train_accuracy, train_loss, test_accuracy, test_loss))
        if best_test_accuracy < test_accuracy:
            best_test_accuracy = test_accuracy
            if not os.path.isdir('./model'):
                os.mkdir('./model')
            torch.save(net.state_dict(), "./model/fc.pt")
            # print("### Epoch {}\t\tSave Model fot test accuracy {}% ".format(epoch+1, best_test_accuracy))
    print("Finish!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fully Connected Neural Network')
    parser.add_argument('mode', type=str, help='Mode: train / test')
    parser.add_argument('image', type=str, default='', nargs='?', help='Image file in test')
    parser.add_argument('--epoch', type=int, default=20, help='Total Number of Epoch')
    arg = parser.parse_args()

   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using Device %s" % device)
    if arg.mode == 'train':
        train(device, arg.epoch)
    elif arg.mode == 'test':
        net = Net()
        net.to(device)
        net.load_state_dict(torch.load('./model/fc.pt'))
        print("Load network from ./model/fc.pt")
       
        img = Image.open(arg.image)
        
        
        transform = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        image_tensor = transform(img).float()
        label = test(net, image_tensor, device, flag='inference')
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        print("Predict: %s" % classes[label])

