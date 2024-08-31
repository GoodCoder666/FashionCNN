import torch
import torch.nn.functional as F

import dataloader
from net.mynet import Net
from time import time

batch_size = 128
model_name = 'mynet'
resize = None

trainloader, testloader = dataloader.FashionMNIST(batch_size, resize=resize)

device = torch.device('cuda:0')
net = Net(1, 10).to(device)
net.load_state_dict(torch.load(f'models/{model_name}.pt'))
net.eval()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'params: {count_parameters(net)/1e6:.1f}M')

def loss_accuracy(net, dataloader):
    total = correct = 0
    loss = count = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            # loss
            loss += F.cross_entropy(outputs, labels).item()
            count += 1
            # accuracy
            predictions = outputs.max(1, keepdim=True)[1].flatten().cpu().numpy()
            total += len(predictions)
            correct += (predictions == labels.cpu().numpy()).sum().item()
    return loss / count, correct / total, total


print()

start_time = time()
train_loss, train_acc, n1 = loss_accuracy(net, trainloader)
test_loss, test_acc, n2 = loss_accuracy(net, testloader)
end_time = time()

print(f'train loss: {train_loss:.3f}')
print(f'train acc: {train_acc:.3f}')

print()

print(f'test loss: {test_loss:.3f}')
print(f'test acc: {test_acc:.3f}')

print()

speed = (n1 + n2) / (end_time - start_time)
print(f'speed: {speed:.1f} examples/s')