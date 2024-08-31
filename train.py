import torch
from torch import nn

from tqdm import tqdm

import dataloader
# from net.lenet import LeNet as Net
# from net.mynet import Net
# from net.alexnet import AlexNet as Net
# from net.alexnet import AlexNet_Tiny as Net
# from net.vgg import VGG11 as Net
# from net.nin import NiN as Net
# from net.googlenet import GoogLeNet as Net
# from net.resnet import ResNet18 as Net
from net.densenet import DenseNet as Net
from optim.ranger21 import Ranger21

lr = 0.01
epochs = 100
batch_size = 128
model_name = 'densenet'
resize = 96 # set to None (input_size=28) or input_size (input_size!=28)

trainloader, testloader = dataloader.FashionMNIST(batch_size, resize=resize)
print('dataset loaded')

device = torch.device('cuda:0')
net = Net(1, 10).to(device)
print('net initialized')

def accuracy(net, dataloader):
    total = correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.cpu().numpy()
            outputs = net(inputs).max(1, keepdim=True)[1].flatten().cpu().numpy()
            total += len(outputs)
            correct += (outputs == labels).sum().item()
    return correct / total

loss = nn.CrossEntropyLoss()
optimizer = Ranger21(net.parameters(),
                     lr=lr, num_epochs=epochs, num_batches_per_epoch=len(trainloader))

for epoch in range(epochs):
    with tqdm(enumerate(trainloader), desc=f'Epoch #{epoch + 1}', total=len(trainloader)) as bar:
        net.train()
        for i, (inputs, labels) in bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            l = loss(outputs, labels)
            l.backward()
            optimizer.step()
            bar.set_postfix(loss=l.item())
        net.eval()
    print('acc:', accuracy(net, testloader))

print('Finished training')
torch.save(net.state_dict(), f'models/{model_name}.pt')