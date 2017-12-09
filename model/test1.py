import torch.nn as nn
import torch.nn.functional as F

SIZE1 = 6
SIZE2 = 18
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.softmax = nn.LogSoftmax(dim = 0)
        self.pool   = nn.MaxPool2d(2,2)
        self.conv11 = nn.Conv2d(3, SIZE1, 3, padding = 1) # 1 input image channel, 64 output channels, 5x5 square convolution kernel
        self.conv12 = nn.Conv2d(SIZE1, SIZE1, 3, padding = 1);
        
        self.conv21 = nn.Conv2d(SIZE1, SIZE2, 3, padding = 1)
        self.conv22 = nn.Conv2d(SIZE2, SIZE2, 3, padding = 1)
        
        self.fc1   = nn.Linear(SIZE2*20*45, 16*20*45) # an affine operation: y = Wx + b
        self.fc2   = nn.Linear(16*20*45, 1024)
        self.fc3   = nn.Linear(1024, 256)
        self.fc4   = nn.Linear(256, 4*36)

    def forward(self, x):
        
        #####layer 1#####
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.pool(F.relu(x))
        #####layer 1#####
        
        #####layer 2#####
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.pool(F.relu(x))
        #####layer 2#####
        
        x = x.view(-1, SIZE2*20*45)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(-1,4,36)
        x = self.softmax(x)
        print(x)
        return x

