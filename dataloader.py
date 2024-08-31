'''
Data loader for deep learning demos.
All functions here return a tuple, consisting of two DataLoaders with train and test data.

**This module is NOT intended for production use.**
'''
import torch
import torchvision

__all__ = ['FashionMNIST']

NUM_WORKERS = 0  # Set to 0 for stability on Windows
DATA_ROOT = '../data'  # Data root path, arbitrary.

def FashionMNIST(batch_size, resize=None):
    '''Fashion-MNIST dataset.'''
    trans = [torchvision.transforms.ToTensor()]
    if resize:
        trans.insert(0, torchvision.transforms.Resize(resize))
    trans = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root=DATA_ROOT, train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root=DATA_ROOT, train=False, transform=trans, download=True)
    return (torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True,
                                        num_workers=NUM_WORKERS),
            torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False,
                                        num_workers=NUM_WORKERS))