import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.utils.data
from torch.nn.modules import conv, Linear
from torch.nn.modules.utils import _pair, _triple
import torch.backends.cudnn as cudnn

from readData import custom_dataset, label_loader_64x64_31t, label_loader_64x64_62tp

import random
import argparse
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='train SNDCGAN model')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
#parser.add_argument('--gpu_ids', default=[0,1,2,3], help='gpu ids')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--batchSize', type=int, default=32, help='with batchSize=1 equivalent to instance normalization.')
# parser.add_argument('--label_num', type=int, default=31, help='number of labels.')
parser.add_argument('--test_num_per_label', type=int, default=5, help='number of test images per label.')
parser.add_argument('--test_epoch', type=int, default=1, help='number of epoch to be tested (G output)')
parser.add_argument('--epoch', type=int, default=200, help='number of epoches.')
parser.add_argument('--label_mode', type=int, help='label mode, 1(31t) or 2(31t x 2p) or 3(object index).')
parser.add_argument('--name', help='experiment name')
parser.add_argument('--test', action='store_true', help='test mode')
parser.add_argument('--savepath', help='model save path to load for testing')

opt = parser.parse_args()
print(opt)

if opt.name is None:
    print('You must choose experiment name, result is in log/$name')
    exit(-1)
if not os.path.exists('log/' + opt.name):
    os.mkdir('log/' + opt.name)

if opt.label_mode is None:
    print('You must choose the label mode!')
    exit(-1)

label_loader = None
label_num_list = ()
label_num = 0

if opt.label_mode == 1:
    label_loader =  label_loader_64x64_31t
    label_num_list = [31]
if opt.label_mode == 2:
    label_loader = label_loader_64x64_62tp
    label_num_list = (31, 2)

for i in label_num_list:
    label_num += i

'''
dataset = datasets.ImageFolder(root='exp/64x64_31t',
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
                            img_hdf5_path='../rendered_chairs/all_chair_img.h5',
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

def _l2normalize(v, eps=1e-12):
    return v / (((v**2).sum())**0.5 + eps)

def max_singular_value(W, u=None, Ip=1):
    """
    power iteration for weight parameter
    """
    #xp = W.data
    if u is None:
        u = torch.FloatTensor(1, W.size(0)).normal_(0, 1).cuda()
    _u = u
    for _ in range(Ip):
        #print(_u.size(), W.size())
        _v = _l2normalize(torch.matmul(_u, W.data), eps=1e-12)
        _u = _l2normalize(torch.matmul(_v, torch.transpose(W.data, 0, 1)), eps=1e-12)
    sigma = torch.matmul(torch.matmul(_v, torch.transpose(W.data, 0, 1)), torch.transpose(_u, 0, 1))
    return sigma, _v

class SNConv2d(conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(SNConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _ = max_singular_value(w_mat)
        #print(sigma.size())
        self.weight.data = self.weight.data / sigma
        #print(self.weight.data)
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class SNLinear(Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SNLinear, self).__init__(in_features, out_features, bias)
        self.u = None
    def forward(self, input):
        w_mat = self.weight
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u = _u
        self.weight.data = self.weight.data / sigma
        return F.linear(input, self.weight, self.bias)


def weight_filler(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class _netG(nn.Module):
    def __init__(self, nz, nc, ngf):
        super(_netG, self).__init__()

        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=True),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )
        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(label_num, ngf * 4, 4, 1, 0, bias=True),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=True),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input, input_c):
        out1 = self.convT1(input)
        out2 = self.convT2(input_c)
        output = torch.cat([out1, out2], 1)
        output = self.main(output)

        return output

class _netD(nn.Module):
    def __init__(self, nc, ndf):
        super(_netD, self).__init__()

        self.conv1_1 = SNConv2d(nc, ndf/2, 4, 2, 1, bias=True)
        self.conv1_2 = SNConv2d(label_num, ndf/2, 4, 2, 1, bias=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.snlinear = SNLinear(ndf * 16 * 4 * 4, 1)
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            #SNConv2d(nc, ndf, 4, 2, 1, bias=False),
            #nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            SNConv2d(ndf, ndf * 2, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*2) x 16 x 16
            SNConv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*4) x 8 x 8
            SNConv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),

            # state size. (ndf*8) x 4 x 4
            SNConv2d(ndf * 8, ndf * 16, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
            #nn.Softplus()
        )

    def forward(self, input, input_c):
        out1 = self.lrelu(self.conv1_1(input))
        out2 = self.lrelu(self.conv1_2(input_c))
        output = torch.cat([out1, out2], 1)
        output = self.main(output)
        output = output.view(output.size(0), -1)
        output = self.snlinear(output)
        return output.view(-1, 1).squeeze(1)

nz = opt.nz

G = _netG(nz, 3, 64)
SND = _netD(3, 64)
#print(G)
#print(SND)
G.apply(weight_filler)
SND.apply(weight_filler)

input = torch.FloatTensor(opt.batchSize, 3, 64, 64).cuda()
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).cuda()
# label_list = []
# for label_num in label_num_list:
#     label_list.append(torch.FloatTensor(opt.batchSize).cuda())
real_label = 1
fake_label = 0


fix_onehot_list = []
fill_list = []
total_fix_length = opt.test_num_per_label
for label_num in label_num_list:
    total_fix_length *= label_num


for label_num in label_num_list:
    fix_length = opt.test_num_per_label * label_num
    fix_label = torch.FloatTensor(fix_length)
    for i in range(opt.test_num_per_label):
        for j in range(label_num):
            fix_label[i * label_num + j] = j
    fix = torch.LongTensor(fix_length, 1).copy_(fix_label)
    fix_onehot = torch.FloatTensor(fix_length, label_num)
    fix_onehot.zero_()
    fix_onehot.scatter_(1, fix, 1)
    fix_onehot = fix_onehot.view(-1, label_num, 1, 1)
    fix_onehot = Variable(fix_onehot).cuda()
    fix_onehot_list.append(fix_onehot)

    fill = torch.zeros([label_num, label_num, 64, 64])
    for i in range(label_num):
        fill[i, i, :, :] = 1
    fill_list.append(fill)

for i in range(len(fix_onehot_list)):
    fix_onehot = fix_onehot_list[i]
    repeat_time = total_fix_length / (opt.test_num_per_label * fix_onehot.shape[1])
    fix_onehot_list[i] = fix_onehot.repeat(repeat_time, 1, 1, 1)

fix_onehot_concat = torch.cat(fix_onehot_list, 1)

fixed_noise = torch.FloatTensor(total_fix_length, nz, 1, 1).normal_(0, 1)
#fixed_input = torch.cat([fixed_noise, fix_onehot],1)
fixed_noise = Variable(fixed_noise).cuda()

criterion = nn.BCELoss()


if opt.cuda:
    G.cuda()
    SND.cuda()
    criterion.cuda()


if opt.test:
    save_path = opt.savepath
    G.load_state_dict(torch.load(save_path))

    fix_onehot_list = []
    fill_list = []
    test_num_per_label = 1
    total_fix_length = test_num_per_label
    for label_num in label_num_list:
        total_fix_length *= label_num


    for label_num in label_num_list:
        fix_length = test_num_per_label * label_num
        fix_label = torch.FloatTensor(fix_length)
        for i in range(test_num_per_label):
            for j in range(label_num):
                fix_label[i * label_num + j] = j
        fix = torch.LongTensor(fix_length, 1).copy_(fix_label)
        fix_onehot = torch.FloatTensor(fix_length, label_num)
        fix_onehot.zero_()
        fix_onehot.scatter_(1, fix, 1)
        fix_onehot = fix_onehot.view(-1, label_num, 1, 1)
        fix_onehot = Variable(fix_onehot).cuda()
        fix_onehot_list.append(fix_onehot)

        fill = torch.zeros([label_num, label_num, 64, 64])
        for i in range(label_num):
            fill[i, i, :, :] = 1
        fill_list.append(fill)

    for i in range(len(fix_onehot_list)):
        fix_onehot = fix_onehot_list[i]
        repeat_time = total_fix_length / (test_num_per_label * fix_onehot.shape[1])
        fix_onehot_list[i] = fix_onehot.repeat(repeat_time, 1, 1, 1)

    fix_onehot_concat = torch.cat(fix_onehot_list, 1)

    fixed_noise = torch.FloatTensor(total_fix_length, nz, 1, 1).normal_(0, 1)
    #fixed_input = torch.cat([fixed_noise, fix_onehot],1)
    fixed_noise = Variable(fixed_noise).cuda()


    fake = G(fixed_noise, fix_onehot_concat)
    vutils.save_image(fake.data,
        '%s/test.png' % ('log/' + opt.name),
        normalize=True)
    exit()



optimizerG = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerSND = optim.Adam(SND.parameters(), lr=0.0002, betas=(0.5, 0.999))

for epoch in range(opt.epoch + 1):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        SND.zero_grad()
        real_cpu, labels_list = data
        batch_size = real_cpu.size(0)
        #print(labels_list)
        y_onehot_v_list = []
        y_fill_list = []

        shape = 1
        if opt.label_mode == 2:
            shape = 2
        for j in range(shape):
            labels = labels_list[0]
            if shape > 1:
                labels = labels_list[j]

            label_num = label_num_list[j]
            fill = fill_list[j]
            #label = label_list[j]
            y = torch.LongTensor(batch_size, 1).copy_(labels)
            y_onehot = torch.zeros(batch_size, label_num)
            y_onehot.scatter_(1, y, 1)
            y_onehot_v = y_onehot.view(batch_size, -1, 1, 1)
            #print(y_onehot_v.size())
            y_onehot_v = Variable(y_onehot_v.cuda())
            y_onehot_v_list.append(y_onehot_v)

            # y_fill: (batch_size, label_num, 64, 64)
            y_fill = fill[labels]
            y_fill = Variable(y_fill.cuda())
            y_fill_list.append(y_fill)

        y_onehot_v_concat = y_onehot_v_list[0]
        if opt.label_mode == 2:
            y_onehot_v_concat = torch.cat([y_onehot_v_list[0], y_onehot_v_list[1]], 1)
        y_fill_concat = y_fill_list[0]
        if opt.label_mode == 2:
            y_fill_concat = torch.cat([y_fill_list[0], y_fill_list[1]], 1)
        input.resize_(real_cpu.size()).copy_(real_cpu)
        # label.resize_(batch_size).fill_(real_label)
        inputv = Variable(input)
        # labelv = Variable(label)
        output = SND(inputv, y_fill_concat)
        #print(output)
        errD_real = torch.mean(F.softplus(-output).mean())
        #errD_real = criterion(output, labelv)
        #errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise.resize_(batch_size, opt.nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        #y_nz = torch.cat([noisev, y_onehot], 1)
        fake = G(noisev, y_onehot_v_concat)
        # labelv = Variable(label.fill_(fake_label))
        output = SND(fake.detach(), y_fill_concat)
        errD_fake = torch.mean(F.softplus(output))
        #errD_fake = criterion(output, labelv)
        #errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        errD.backward()
        optimizerSND.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        G.zero_grad()
        # labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        output = SND(fake, y_fill_concat)
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
            '%s/real_samples.png' % ('log/' + opt.name),
            normalize=True)
        fake = G(fixed_noise, fix_onehot_concat)
        vutils.save_image(fake.data,
            '%s/fake_samples_epoch_%03d.png' % ('log/' + opt.name, epoch),
            normalize=True)
    if epoch % 20 == 0:
        # do checkpointing
        torch.save(G.state_dict(), '%s/netG_epoch_%d.pth' % ('log/' + opt.name, epoch))
        torch.save(SND.state_dict(), '%s/netD_epoch_%d.pth' % ('log/' + opt.name, epoch))

