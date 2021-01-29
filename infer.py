# @Author: ZWY
# @Date  : 2021/01/29

import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
from libs.networks.networks import EmbeddingNet, TripletNet

cuda = torch.cuda.is_available()
mean, std = 0.1307, 0.3081

train_dataset = MNIST('/mnt/c/Users/wuyang.zhang/Documents/work/data/MNIST', train=True, download=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((mean,), (std,))
                             ]))
test_dataset = MNIST('/mnt/c/Users/wuyang.zhang/Documents/work/data/MNIST', train=False, download=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((mean,), (std,))
                            ]))
mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
embedding_net = EmbeddingNet()
# print(embedding_net)
model = TripletNet(embedding_net)
checkpoint = torch.load('/mnt/c/Users/wuyang.zhang/Documents/work/stamps_triplet/checkpoints/19.pth')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(10,10))
    for i in range(10):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes)
    plt.savefig('/mnt/c/Users/wuyang.zhang/Documents/work/stamps_triplet/test.jpg')

def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels
# train_embeddings_tl, train_labels_tl = extract_embeddings(train_loader, model)
# plot_embeddings(train_embeddings_tl, train_labels_tl)
val_embeddings_tl, val_labels_tl = extract_embeddings(test_loader, model)
plot_embeddings(val_embeddings_tl, val_labels_tl)