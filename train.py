# @Author: ZWY
# @Date  : 2021/01/29

import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
# Set up the network and training parameters
# from .libs.networks.networks import EmbeddingNet, ClassificationNet
from libs.networks.networks import EmbeddingNet, TripletNet
from libs.loss.losses import TripletLoss
from libs.metrics import AccumulatedAccuracyMetric

from libs.trainer import fit
import numpy as np

from torchvision.datasets import MNIST
from torchvision import transforms
from libs.datasets import TripletMNIST
import os
# import struct

model_save_path = '/mnt/c/Users/wuyang.zhang/Documents/work/stamps_triplet'

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
n_classes = 10

triplet_train_dataset = TripletMNIST(train_dataset) # Returns triplets of images
triplet_test_dataset = TripletMNIST(test_dataset)
batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

margin = 1.
embedding_net = EmbeddingNet()
# print(embedding_net)
model = TripletNet(embedding_net)
if cuda:
    model.cuda()
loss_fn = TripletLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 100
start_epoch = 0

for epoch in range(start_epoch, n_epochs):
    fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval,epoch=epoch)
    print('saving model...')
    torch.save({
        'epoch': epoch, 
        'state_dict': model.state_dict()}, 
        os.path.join(model_save_path, str(epoch)+'.pth'))