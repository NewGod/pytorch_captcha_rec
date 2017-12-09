import torch.nn as nn
import torch.nn.functional as F

SIZE1 = 6
SIZE2 = 12
SIZE3 = 24
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool  = nn.MaxPool2d(2,2)
        
        #####layer 1##### input 3*180*80
        self.conv11 = nn.Conv2d(3, SIZE1, 3, padding = 1) # 1 input image channel, 64 output channels, 5x5 square convolution kernel
        self.conv12 = nn.Conv2d(SIZE1, SIZE1, 3, padding = 1);
        #####layer 1##### output 128*90*40
        
        #####layer 2##### input 128*90*40
        self.conv21 = nn.Conv2d(SIZE1, SIZE2, 3, padding = 1)
        self.conv22 = nn.Conv2d(SIZE2, SIZE2, 3, padding = 1)
        #####layer 2##### output 256*45*20
        
        #####layer 3##### input 256*45*20
        self.conv31 = nn.Conv2d(SIZE2, SIZE3, 3, padding = 1)
        self.conv32 = nn.Conv2d(SIZE3, SIZE3, 3, padding = 1)
        self.conv33 = nn.Conv2d(SIZE3, SIZE3, 3, padding = 1)
        self.conv34 = nn.Conv2d(SIZE3, SIZE3, 3, padding = 1)
        #####layer 3##### output 256*45*20
        
        self.fc1   = nn.Linear(SIZE3*45*20, SIZE2*45*20) # an affine operation: y = Wx + b
        self.fc2   = nn.Linear(SIZE2*45*20, SIZE1*45*20)
        self.fc3   = nn.Linear(SIZE1*45*20, 16*45*20)
        self.fc4   = nn.Linear(16*20*45, 1024)
        self.fc5   = nn.Linear(1024, 256)
        self.fc6   = nn.Linear(256, 4*36)

    def forward(self, x):
        
        #####layer 1##### input 3*180*80
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.pool(F.relu(x))
        #####layer 1##### output 128*90*40
        
        #####layer 2##### input 128*90*40
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.pool(F.relu(x))
        #####layer 2##### output 256*45*20
        
        #####layer 3##### input 256*45*20
        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)
        x = self.conv34(x)
        # x = self.pool(F.relu(x))
        #####layer 3##### output 256*45*20
        
        #####output#####
        x = x.view(-1, SIZE3*45*20)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        x = F.softmax(x)
        x = x.view(-1,4,36)
        #####output#####
        return x

