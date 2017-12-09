import torch.nn as nn
import torch.nn.functional as F

SIZE1 = 32
SIZE2 = 64
SIZE3 = 128
SIZE4 = 256
stringlen = 4
characterlen = 36

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool  = nn.MaxPool2d(2,2)
        
        #####layer 1##### input 3 * [80*180]
        self.conv11 = nn.Conv2d(3, SIZE1, 3) # 1 input image channel, 64 output channels, 5x5 square convolution kernel
        self.conv12 = nn.Conv2d(SIZE1, SIZE1, 3, padding = 1);
        #####layer 1##### output 32 * [76*176]
        
        #maxpool output 32 * [38*83]
        
        #####layer 2##### input 32 * [38*88]
        self.conv21 = nn.Conv2d(SIZE1, SIZE2, 3)
        self.conv22 = nn.Conv2d(SIZE2, SIZE2, 3)
        #####layer 2##### output 64 * [34*84]
        
        #maxpool output 64 * [17*42]
        
        #####layer 3##### input 64 * [17*42]
        self.conv31 = nn.Conv2d(SIZE2, SIZE3, 3)
        self.conv32 = nn.Conv2d(SIZE3, SIZE3, 3)
        #####layer 3##### output 128 * [13*38]
        
        #maxpool output 128 * [6*19]
        
        #####layer 4##### input 128 * [6*19]
        self.conv41 = nn.Conv2d(SIZE3, SIZE4, 3)
        self.conv42 = nn.Conv2d(SIZE4, SIZE4, 3)
        #####layer 4##### output 256 * [2*15]
        
        #maxpool output 256 * [1*7]
        
        #####output#####
        self.fc1   = nn.Linear(SIZE4*7, SIZE4*7) # an affine operation: y = Wx + b
        self.fc2   = nn.Linear(SIZE4*7, stringlen*characterlen)
        #####output#####

    def forward(self, x):
        
        #####layer 1##### input
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.pool(F.relu(x))
        #####layer 1##### output
        
        #####layer 2##### input
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.pool(F.relu(x))
        #####layer 2##### output
        
        #####layer 3##### input
        x = self.conv31(x)
        x = self.conv32(x)
        x = self.pool(F.relu(x))
        #####layer 3##### output
        
        #####layer 3##### input
        x = self.conv41(x)
        x = self.conv42(x)
        x = self.pool(F.relu(x))
        #####layer 3##### output
        
        #####output#####
        x = x.view(-1, SIZE4*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, stringlen, characterlen)
        x = F.softmax(x)
        #####output#####
        return x

