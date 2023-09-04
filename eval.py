import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import tome
import timm
import torch.optim as optim
import argparse
import onnxruntime
import sys
import pdb

transform = transforms.Compose(
    [transforms.Resize((224,224)),
      transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



model_path = sys.argv[1]
session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

correct = 0
total = 0
with torch.no_grad():
    for data in tqdm(testloader):
        images, labels = data
        images, labels = images.numpy(), labels
       
        #pdb.set_trace()
        outputs = session.run([output_name], {input_name: images})[0]

        _, predicted = torch.max(torch.tensor(outputs), 1)
        total += 1
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

