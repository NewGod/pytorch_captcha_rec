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
import argparse

has_gpu = False
PATH = 'logs/'
MODEL_PATH = os.path.join(PATH,'model.pkl')
LEARNING_RATE = 0.01
BATCH_SIZE = 64
EPOCH = 40
EPOCH_SIZE = 600
VERI_SIZE = 600

def model_load(net):
    net.load_state_dict(torch.load(args.path))
    print('model load from {}'.format(args.path))

def train(net):
    criterion = nn.BCELoss() # use a Classification Cross-Entropy loss
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9)
    logger = Logger(PATH)
    step = 0

    for epoch in range(args.epoch): # loop over the dataset multiple times
        trainloader = data_generator(times = EPOCH_SIZE,batch_size = args.batch_size)
        
        running_loss = 0.0
        for i, data in tqdm(enumerate(trainloader, 0),total = EPOCH_SIZE):
            # get the inputs
            inputs, labels = data

            if args.gpu:
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
    
    torch.save(net.state_dict(), args.path)
    print('model save at {}'.format(args.path))

def verification(net):
    print("start verification")
    testloader = data_generator(times = VERI_SIZE, batch_size = args.batch_size)
    correct = 0
    total = 0
    for i,data in tqdm(enumerate(testloader,0),total = VERI_SIZE):
        images, labels = data
        if args.gpu:
            images,labels = images.cuda(),labels.cuda()
        outputs = net(Variable(images))
        total += labels.size(0)
        if args.gpu:
            labels = labels.cpu()
            outputs = outputs.cpu()
        outputs = decode(outputs.data.numpy())
        labels = decode(labels.numpy())
        correct += np.array([outputs[i]==labels[i] for i in range(args.batch_size)]).sum()

    print('Accuracy of the network on the %d test images: %d %%' % (total,100 * correct / total))
if __name__ =='__main__':
    parser = argparse.ArgumentParser(description="Train or verify your model")
    parser.add_argument('-v','--verify',help='Verify the model',action='store_true')
    parser.add_argument('-g','--gpu',help='use the gpu to speed up the program',action='store_true')
    parser.add_argument('-b','--batch_size',help='Change the batch_size of train process',type = int, default=BATCH_SIZE)
    parser.add_argument('-p','--path',help='the model path',type = str, default = MODEL_PATH)
    parser.add_argument('-e','--epoch',help='the epoch size',type = int, default = EPOCH)
    parser.add_argument('-lr','--learning_rate',help='the learning rate',type = float, default = LEARNING_RATE)
    args = parser.parse_args()
    if not os.path.isdir(PATH):
        os.mkdir(PATH)

    net = Net()
    if args.gpu :
        net = net.cuda()

    if args.verify != None:
        starttime = datetime.datetime.now()
        train(net)
        endtime = datetime.datetime.now()
        verification(net)
        print(str(endtime - starttime))
    else:
        model_load(net)
        verification(net)
