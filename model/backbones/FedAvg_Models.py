import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFAR10_CNN(nn.Module):
    def __init__(self):
        '''
        The simple CNN is also used in paper: Federated Learning on Non-IID Data Silos: An Experimental Study[http://arxiv.org/abs/2102.02079]
    
        Used for Cifar dataset. The input image size is 32x32.
        '''
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    @property
    def num_features(self):
        return self.fc2.out_features

class Mnist_2NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, inputs):
        tensor = F.relu(self.fc1(inputs))
        tensor = F.relu(self.fc2(tensor))
        tensor = self.fc3(tensor)
        return tensor
    
    @property
    def num_features(self):
        return self.fc2.out_features

class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7*7*64)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)   #在FedAvg论文中提到有一个softmax输出层，这里的虽然是输出层，但没有softmax
        return tensor

    @property
    def num_features(self):
        return self.fc1.out_features
    
class Mnist_CNN_bn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = self.conv1(tensor)
        tensor = F.relu(self.bn1(tensor))
        tensor = self.pool1(tensor)

        tensor = self.conv2(tensor)
        tensor = F.relu(self.bn2(tensor))
        tensor = self.pool2(tensor)
        
        tensor = tensor.view(-1, 7*7*64)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)   #在FedAvg论文中提到有一个softmax输出层，这里的虽然是输出层，但没有softmax
        return tensor

    @property
    def num_features(self):
        return self.fc1.out_features

if __name__ == '__main__':
    net = Mnist_CNN_bn()
    # print(sum([p.nelement() for p in net.parameters()]))
    print(net)