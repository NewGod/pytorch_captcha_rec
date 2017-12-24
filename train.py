import numpy
import torch
from torch.autograd import Variable
from model.vgg_like import Net
from torch import optim,nn
from tqdm import tqdm
from dataset import *
import os
import pickle
import datetime
from logger import Logger


has_gpu = False
PATH = 'logs/'
MODEL_PATH = os.path.join(PATH,'model.pkl')
LEARNING_RATE = 0.01
BATCH_SIZE = 64
EPOCH = 40
EPOCH_SIZE = 600
VERI_SIZE = 600

def model_load(net):
    net.load_state_dict(torch.load(MODEL_PATH))
    print('model load from {}'.format(MODEL_PATH))

def train(net):
    criterion = nn.BCELoss() # use a Classification Cross-Entropy loss
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)
    logger = Logger(PATH)
    step = 0

    for epoch in range(EPOCH): # loop over the dataset multiple times
        trainloader = data_generator(times = EPOCH_SIZE,batch_size = BATCH_SIZE)
        
        running_loss = 0.0
        for i, data in tqdm(enumerate(trainloader, 0),total = EPOCH_SIZE):
            # get the inputs
            inputs, labels = data

            if has_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()        
            optimizer.step()
            running_loss += loss.data[0]
            logger.scalar_summary('loss', loss.data[0], step);
            step = step + 1
            
        print('epoch: %d   loss: %.3f' % (epoch+1, running_loss / EPOCH_SIZE))
        running_loss = 0.0
        # print statistics
    print('Finished Training')
    
    torch.save(net.state_dict(), MODEL_PATH)
    print('model save at {}'.format(MODEL_PATH))

def verification(net):
    print("start verification")
    testloader = data_generator(times = VERI_SIZE, batch_size = BATCH_SIZE)
    correct = 0
    total = 0
    for i,data in tqdm(enumerate(testloader,0),total = VERI_SIZE):
        images, labels = data
        if has_gpu:
            images,labels = images.cuda(),labels.cuda()
        outputs = net(Variable(images))
        total += labels.size(0)
        if has_gpu:
            labels = labels.cpu()
            outputs = outputs.cpu()
        outputs = decode(outputs.data.numpy())
        labels = decode(labels.numpy())
        correct += np.array([outputs[i]==labels[i] for i in range(BATCH_SIZE)]).sum()

    print('Accuracy of the network on the %d test images: %d %%' % (total,100 * correct / total))
"""
    class_correct = list(0. for i in range(100))
    class_total = list(0. for i in range(100))
    for data in testloader:
        images, labels = data
        if has_gpu:
            images,labels = images.cuda(),labels.cuda()
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        for i in range(min(BATCH_SIZE,list(c.size())[0])):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    for i in range(100):
        print('Accuracy of %5s : %2d %%' % (i, 100 * class_correct[i] / class_total[i]))
"""
if __name__ =='__main__':
    starttime = datetime.datetime.now()
    if not os.path.isdir(PATH):
        os.mkdir(PATH)

    net = Net()
    if has_gpu :
        net = net.cuda()
    train(net)
    #model_load(net)

    endtime = datetime.datetime.now()

    print(str(endtime - starttime))
