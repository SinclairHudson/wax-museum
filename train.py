import torch
import torchvision

from torchvision import transforms, datasets
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import os
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import numpy as np

# weights and biases login
import wandb

conf = {
    "epochs": 10,
    "batch_size": 20,
    "learning_rate": 0.008,
    "momentum": 0.9,
    "reports_per_epoch": 7,
    "width": 480,
    "height": 480,
    "dropout": 0.1
}
wandb.init(project="waxmuseum", config=conf)

class_names = ['looking', 'notlooking']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

best = 0


# 640 x 480 is the dimension of the images being inputted
class Net(nn.Module):
    def __init__(self):  # constructor
        super(Net, self).__init__()  # parent constructor

        self.drop = nn.Dropout2d(conf["dropout"])

        # in_channels, out_channels, kernel_size
        self.conv1 = nn.Conv2d(3, 10, 5, padding=2)  # 640 x 480 x 10
        self.conv1_bn = nn.BatchNorm2d(10)
        # self.conv2 = nn.Conv2d(10, 16, 9, padding=4)  # 640 x 480 x 16
        self.pool1 = nn.MaxPool2d(2, 2)  # 320 x 240 x 16
        self.conv3 = nn.Conv2d(10, 10, 15, padding=7)  # 320 x 240 x 16
        self.conv3_bn = nn.BatchNorm2d(10)
        self.pool2 = nn.MaxPool2d(2, 2)  # 160 x 120 x 16
        # self.conv4 = nn.Conv2d(16, 16, 15, padding=7)  # 160 x 120 x 16
        self.conv5 = nn.Conv2d(10, 10, 15, padding=7)  # 160 x 120 x 16
        self.conv5_bn = nn.BatchNorm2d(10)
        self.pool3 = nn.MaxPool2d(4, 4)  # 40 x 30 x 16
        self.conv6 = nn.Conv2d(10, 4, 9, padding=4)  # 40 x 30 x 4
        self.conv6_bn = nn.BatchNorm2d(4)
        self.pool4 = nn.MaxPool2d(2, 2)  # 20 x 15 x 4
        self.fc1 = nn.Linear(4 * conf["height"] * conf["width"] // (32 ** 2), 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.leaky_relu(self.conv1_bn(self.conv1(x)))
        # x = F.leaky_relu(self.conv2(x))
        x = self.pool1(x)
        x = self.drop(x)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)))
        x = self.pool2(x)
        x = self.drop(x)
        # x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5_bn(self.conv5(x)))
        x = self.pool3(x)
        x = self.drop(x)
        x = F.leaky_relu(self.conv6_bn(self.conv6(x)))
        x = self.pool4(x)
        x = self.drop(x)
        x = x.view(-1, 4 * conf["height"] * conf["width"] // (32 ** 2))
        x = self.fc1(x)

        return x


net = Net()

net.to(device)
wandb.watch(net)

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


# compose applies in the order of the array that's passed
norm = transforms.Compose([transforms.CenterCrop((conf["height"], conf["width"])),  # center crop a square
                           transforms.ToTensor(),  # now turn into a tensor
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize from [0,1] to [-1, 1]
                           ])
classes = ('looking', 'notlooking')

net.conv1.register_forward_hook(get_activation('conv1'))
net.conv3.register_forward_hook(get_activation('conv3'))
net.conv5.register_forward_hook(get_activation('conv5'))
net.conv6.register_forward_hook(get_activation('conv6'))


def checkcorrupt(filename):
    try:
        img = Image.open('./' + filename)  # open the image file
        img.verify()  # verify that it is, in fact an image
        return True
    except (IOError, SyntaxError) as e:
        print('Bad file:', filename)  # print out the names of corrupt files
        return False


train_set = datasets.ImageFolder("./dataset", transform=norm, is_valid_file=checkcorrupt)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=conf["batch_size"],
                                           shuffle=True, num_workers=3)

train_length = len(train_loader)
test_set = datasets.ImageFolder("./testset", transform=norm)
sample = []
for x in range(5):
    sample.append(wandb.Image(test_set[x][0]))
wandb.log({"test set": sample})  # test set is iterable, it's tuples of PIL images and
test_loader = torch.utils.data.DataLoader(test_set, batch_size=conf["batch_size"],
                                          shuffle=True, num_workers=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=conf["learning_rate"], momentum=conf["momentum"])
running_loss = 0.0
for epoch in range(conf["epochs"]): # for the specified number of epochs
    for i, data in enumerate(train_loader, 0):  # i here is the batch number

        # report!

        if i % (train_length // conf["reports_per_epoch"]) == 0:
            with torch.no_grad():
                net.eval()  # set to evaluation (no batch norm or dropout)
                total = 0
                correct = 0
                li = []
                pi = []
                confm = np.zeros((2, 2), dtype=int)
                # time to test accuracy
                first = True
                for d in test_loader:
                    inputs, labels = d[0].to(device), d[1].to(device)
                    outputs = net(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    l, p = Tensor.cpu(labels).numpy(), Tensor.cpu(predicted).numpy()
                    cm = confusion_matrix(l,p)
                    confm = np.add(confm, cm)
                    correct += (predicted == labels).sum().item()
                    watchActivations = ['conv1', 'conv3', 'conv5', 'conv6']
                    watchFilters = [net.conv1, net.conv3, net.conv5, net.conv6]
                    if first:
                        report = {}
                        li, pi = l, p
                        for layer in watchActivations:  # report on layer activations
                            act = Tensor.cpu(activation[layer]).numpy()[0]  # the first image only
                            images = []
                            for kernel in act:
                                images.append(wandb.Image(kernel))
                            report[layer+"_activations"] = images
                        for name, module in net.named_children():
                            filters = []
                            if name in watchActivations:
                                f = module.cpu().weight.data.numpy()
                                f = np.swapaxes(np.swapaxes(f, 1, 3), 1, 2)  # switch axis so channel is the last
                                for filt in f:
                                    if filt.shape[2] != 3:  # if it can't be represented as an RGB image
                                        filt = np.sum(filt, axis=2)  # flatten the array
                                    filters.append(wandb.Image(filt))
                                module.cuda()  # move the layer weights back to the GPU for training
                                report[name + "_filters"] = filters
                        first = False

                ta = correct / total
                if ta > best:
                    best = ta
                    if best > 0.98:
                        torch.save(net.state_dict(), os.path.join(wandb.run.dir, 'model-' + str(ta*100) + '.pt'))

                plt.close()
                fig, ax = plot_confusion_matrix(conf_mat=confm, show_absolute=True,
                                                show_normed=True,
                                                colorbar=True)

                report['loss'] = running_loss / 2000
                report['test_accuracy'] = ta
                report['confusion matrix'] = plt
                report['highest test accuracy'] = best

                wandb.log(report)
                print('[%d, %5d] loss: %.5f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

                net.train()  # back to training mode

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

print('Finished Training')
