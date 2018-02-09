import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.utils.data
import torch.backends.cudnn as cudnn

from readData import custom_dataset, label_loader_64x64_31t, label_loader_64x64_62tp

import random
import argparse
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from models.models import _netG, _netD

parser = argparse.ArgumentParser(description='train SNDCGAN model')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
#parser.add_argument('--gpu_ids', default=[0,1,2,3], help='gpu ids: e.g. 0,1,2, 0,2.')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--n_dis', type=int, default=1, help='discriminator critic iters')
parser.add_argument('--nz', type=int, default=128, help='dimention of lantent noise')
parser.add_argument('--batchsize', type=int, default=64, help='training batch size')
parser.add_argument('--test_num_per_label', type=int, default=5, help='number of test images per label.')
parser.add_argument('--test_epoch', type=int, default=1, help='number of epoch to be tested (G output)')
parser.add_argument('--epoch', type=int, default=200, help='number of epoches.')
parser.add_argument('--name', help='experiment name')
parser.add_argument('--test', action='store_true', help='test mode')
parser.add_argument('--savepath', help='model save path to load for testing')

if opt.name is None:
    print('You must choose experiment name, result is in log/$name')
    exit(-1)
if not os.path.exists('log/' + opt.name):
    os.mkdir('log/' + opt.name)

opt = parser.parse_args()
print(opt)
'''
dataset = datasets.ImageFolder(root='/media/scw4750/25a01ed5-a903-4298-87f2-a5836dcb6888/AIwalker/dataset/msceleb12k/celeba',
                           transform=transforms.Compose([
                               transforms.Scale(64),
                               transforms.CenterCrop(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
                                      )
'''
dataset = None
dataloader = None

if not opt.test:
    dataset = custom_dataset(root='../rendered_chairs',
                            names_mat_path='../rendered_chairs/all_chair_names.mat',
                            img_hdf5_path='../rendered_chairs/all_chair_img_256.h5',
                            label_loader=label_loader,
                            transform=transforms.Compose([
                                #transforms.Scale(64),
                                #transforms.CenterCrop(64),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                            shuffle=True, num_workers=int(2))
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
    #torch.cuda.set_device(opt.gpu_ids[2])

cudnn.benchmark = True

def weight_filler(m):
    classname = m.__class__.__name__
    if classname.find('Conv' or 'SNConv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

n_dis = opt.n_dis
nz = opt.nz

G = _netG(nz, 3, 64)
SND = _netD(3, 64)
#print(G)
#print(SND)
G.apply(weight_filler)
SND.apply(weight_filler)

input = torch.FloatTensor(opt.batchsize, 3, 256, 256)
noise = torch.FloatTensor(opt.batchsize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchsize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchsize)
real_label = 1
fake_label = 0

fixed_noise = Variable(fixed_noise)
criterion = nn.BCELoss()

if opt.cuda:
    G.cuda()
    SND.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

optimizerG = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerSND = optim.Adam(SND.parameters(), lr=0.0002, betas=(0.5, 0.999))

for epoch in range(200):
    for i, data in enumerate(dataloader, 0):
        step = epoch * len(dataloader) + i
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        SND.zero_grad()
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        #if opt.cuda:
        #    real_cpu = real_cpu.cuda()
        input.resize_(real_cpu.size()).copy_(real_cpu)
        #label.resize_(batch_size).fill_(real_label)
        inputv = Variable(input)
        #labelv = Variable(label)
        output = SND(inputv)

        errD_real = torch.mean(F.softplus(-output))
        #errD_real = criterion(output, labelv)
        #errD_real.backward()
        D_x = output.data.mean()
        # train with fake
        noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = G(noisev)
        #labelv = Variable(label.fill_(fake_label))
        output = SND(fake.detach())
        errD_fake = torch.mean(F.softplus(output))
        #errD_fake = criterion(output, labelv)

        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake

        errD.backward()
        optimizerSND.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        if step % n_dis == 0:
            G.zero_grad()
            #labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
            output = SND(fake)
            errG = torch.mean(F.softplus(-output))
            #errG = criterion(output, labelv)
            errG.backward()
            D_G_z2 = output.data.mean()
            optimizerG.step()
        if i % 20 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, opt.epoch, i, len(dataloader),
                     errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
    if epoch % opt.test_epoch == 0:
        vutils.save_image(real_cpu,
                '%s/real_samples.png' % 'log/' + opt.name,
                normalize=True)
        fake = G(fixed_noise)
        vutils.save_image(fake.data,
                '%s/fake_samples_epoch_%03d.png' % ('log/' + opt.name, epoch),
                normalize=True)
    if epoch % 20 == 0:
    # do checkpointing
    torch.save(G.state_dict(), '%s/netG_epoch_%d.pth' % ('log/' + opt.name, epoch))
    torch.save(SND.state_dict(), '%s/netD_epoch_%d.pth' % ('log/' + opt.name, epoch))




























