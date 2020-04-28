import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from get_data import *
# define transform
from preprocess import *
from CNN_utils import *
savedTimit = r'C:\Users\YAbraham\PycharmProjects\pytorch\audio\timit\darpa-timit-acousticphonetic-continuous-speech\data\TIMIT.pkl'
transform = transforms.Compose([Spectograph(),NormSpectograph(),NormSpectograph(),ToTensor()])

timitTrain = TIMIT(savedTimit,transform = transform)

trainloader = torch.utils.data.DataLoader(timitTrain, batch_size=4, shuffle=True, num_workers=0)

#testloader = torch.utils.data.DataLoader(testset, batch_size=4,   shuffle=False, num_workers=0)



def imshow(img):
      # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
d = dataiter.next()

# show images
#imshow(torchvision.utils.make_grid(d['image']))
# print labels


class Net(nn.Module):
    def __init__(self):

        super(Net, self).__init__()
        H,W = [81,81]
        s1 = Conv2dOut([2,15])
        N2,C2,H2,W2 = s1(1,36,H,W)

        s2 = Conv2dOut([3,3])
        N3, C3, H3, W3 = s2(C2, 31, H2, W2)

        N4, C4, H4, W4 = s2(N3, C3, H3, W3) #maxpool


        s3 = Conv2dOut([8,1])
        N5, C5, H5, W5 = s3(C4, 31, H4, W4)

        s4 = Conv2dOut([3, 3])
        N6, C6, H6, W6 = s3(N5, C5, H5, W5)

        CNN_out = int(N6*H6*W6)


        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 36, kernel_size =(2, 15)),
            nn.ReLU(),
        )


        self.layer2 = nn.Sequential(
            nn.Conv2d(36, 31,  kernel_size =(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(31, 15,  kernel_size =(8, 1)),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2)
        )

        self.fc1 = nn.Linear(15* 15* 15, 1000)
        self.fc2 = nn.Linear(1000, 61)


    def forward(self, x):
        x = x.float()
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out  = out.view(-1, 15* 15* 15)
        #out = self.drop_out(out)

        out = self.fc1(out)
        out = self.fc2(out)
        return out


net = Net()
net = net.float()
# loss function

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# training
for epoch in range(2):  # loop over the dataset multiple times


    running_loss = 0.0
    data = iter(trainloader)
    i = 0
    while True:
        try:
            i+=1
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data.next().values()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            inputs = inputs.reshape((4, 1, 81, 81))
            outputs = net(inputs)
            loss = criterion(outputs, Variable(labels.long()))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
        except StopIteration:
            break
        except:
            pass
print('Finished Training')
# save model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)


# test model
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# load model
net = Net()
net.load_state_dict(torch.load(PATH))

# run on images
outputs = net(images)

# get ansewr
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))


end = 1