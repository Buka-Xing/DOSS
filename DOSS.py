# Copyright (C) <2023> Xingran Liao
# @ City University of Hong Kong

# Permission is hereby granted, free of charge, to any person obtaining a copy of this code and
# associated documentation files (the "code"), to deal in the code without restriction, including without
# limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the code,
# and to permit persons to whom the code is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the code.

# THE CODE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE Xingran Liao BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE code OR THE USE OR OTHER DEALINGS IN THE code.
# --------------------------------------------------

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models as tv
import torchvision.transforms as transforms
from math import ceil

def downsample(img1):
    img1 = transforms.functional.resize(img1,256)
    return img1

def duplicate_to_match_lengths(arr1, arr2):
    """
    Duplicates randomly selected entries from the smaller array to match its size to the bigger one
    :param arr1: (r, n) torch tensor
    :param arr2: (r, m) torch tensor
    :return: (r,max(n,m)) torch tensor
    """
    if arr1.shape[2] == arr2.shape[2]:
        return arr1, arr2
    elif arr1.shape[2] < arr2.shape[2]:
        tmp = arr1
        arr1 = arr2
        arr2 = tmp

    b = arr1.shape[2] // arr2.shape[2]
    arr2 = torch.cat([arr2] * b, dim=2)
    if arr1.shape[2] > arr2.shape[2]:
        indices = torch.randperm(arr2.shape[2])[:arr1.shape[2] - arr2.shape[2]]
        arr2 = torch.cat([arr2, arr2[:, :, indices]], dim=2)

    return arr1, arr2

class DOSS(torch.nn.Module):
    def __init__(self, as_loss=False):
        super(DOSS, self).__init__()
        self.patch_size = 5
        self.stride = 1
        self.num_proj = 256
        self.as_loss = as_loss

        self.filters = torch.nn.ParameterList()
        self.weights = nn.Parameter(torch.ones(6))
        chns = [3, 64, 128, 256, 512, 512] # vgg
        for i in range(len(chns)):
            rand = torch.randn(256, chns[i], 5, 5) # (slice_size**2*ch)
            num = rand.view(rand.shape[0], -1).norm(p=1,dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(3)  # 求对应维度的p范数
            rand = rand / num
            self.filters.append(nn.Parameter(rand))

    def forward_once(self, x, y, stride=1, mask=None):
        mask  = mask.to(x.device)
        projx = F.conv2d(x, mask, stride=stride)
        projx = projx.reshape(projx.shape[0], projx.shape[1], -1)
        projy = F.conv2d(y, mask, stride=stride)
        projy = projy.reshape(projy.shape[0], projy.shape[1], -1)

        projx, _ = torch.sort(projx, dim=-1)
        projy, _ = torch.sort(projy, dim=-1)

        projx, projy = duplicate_to_match_lengths(projx, projy)
        s1 = F.cosine_similarity(projx, projy,dim=-1)

        return torch.sum(s1,dim=-1)

    def forward(self, x, y):
        mask = self.filters

        scores = []
        for i in range(len(x)):
            if x[i].shape[2] < 5 or x[i].shape[3] < 5:
                row_padding = ceil(x[i].size(2) / 5) * 5 - x[i].size(2)
                column_padding = ceil(x[i].size(3) / 5) * 5 - x[i].size(3)
                pad = nn.ZeroPad2d((column_padding, 0, 0, row_padding))
                xi_k = pad(x[i])
                yi_k = pad(y[i])
            else:
                xi_k = x[i]
                yi_k = y[i]
            s = self.forward_once(xi_k, yi_k, mask=mask[i])  # x[i].shape[1]
            scores.append(s)  # *self.weights[i]

        score = sum(scores)

        if self.as_loss:
            return torch.log(score + 1)
        else:
            return torch.log(score + 1)

class VGG(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, resize=True):
        super(VGG, self).__init__()
        self.chns = [3, 64, 128, 256, 512, 512]
        self.name = 'VGG'
        self.resize = resize

        vgg19_pretrained_features = tv.vgg19(pretrained=pretrained).features

        # print(vgg_pretrained_features)
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()

        # vgg19 maxpool
        for x in range(0, 4):
            self.stage1.add_module(str(x), vgg19_pretrained_features[x])
        for x in range(4, 9):
            self.stage2.add_module(str(x), vgg19_pretrained_features[x])
        for x in range(9, 18):
            self.stage3.add_module(str(x), vgg19_pretrained_features[x])
        for x in range(18, 27):
            self.stage4.add_module(str(x), vgg19_pretrained_features[x])
        for x in range(27, 36):
            self.stage5.add_module(str(x), vgg19_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def get_features(self, x, resize=True):
        if resize == True:
            x = downsample(x)
        h = x
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        outs = [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]  #

        return outs

    def forward(self, x):
        feats_x = self.get_features(x,resize=self.resize)

        return feats_x

if __name__ == '__main__':
    import argparse
    from utils import prepare_image256
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default='images/26-0.png')
    parser.add_argument('--dist', type=str, default='images/26-4.png')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ref = prepare_image256(Image.open(args.ref).convert("RGB"), resize=False).to(device)
    dist = prepare_image256(Image.open(args.dist).convert("RGB"), resize=False).to(device)

    vgg = VGG(resize=True).to(device)
    vgg.eval()

    ref_stage  = vgg(ref)
    dist_stage = vgg(dist)

    loss = DOSS()
    state_dict = loss.state_dict()  # 进行参数的读取.
    loss.load_state_dict(torch.load('./Projection_kernel.pth', map_location='cuda:0'))
    c = loss(ref_stage,dist_stage)
    # c = 7.334
    print(c.item())