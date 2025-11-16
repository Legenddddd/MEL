from models.resnet12_encoder import *
from timm.models import create_model
import math
import timm

import numpy as np


class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args

        self.encoder = ResNet12()  # pretrained=False
        self.num_features = 640
        self.num_features2 = 320

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.complex_embed = max(self.args.part_num * self.num_features2, self.num_features)

        self.mask_branch = nn.Sequential(
            nn.Conv2d(self.num_features, self.args.part_num*8, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.args.part_num*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.args.part_num*8, self.args.part_num, kernel_size=1, stride=1, padding=0)
        )
        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)

        self.part_fc = nn.Linear(self.args.part_num * self.num_features2, self.args.num_classes, bias=False)

        self.conv_block = nn.Sequential(
            nn.Conv2d(self.num_features, self.args.part_num * self.num_features2, kernel_size=1,
                                  stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(self.args.part_num * self.num_features2, eps=1e-5,
                                 momentum=0.01, affine=True),
            nn.ReLU()
        )

        self.both_mlp = nn.Sequential(
            nn.BatchNorm1d(self.num_features),
            nn.Linear(self.num_features, self.num_features),
            nn.ELU(inplace=True)
        )

        self.both_mlp1 = nn.Sequential(
            nn.BatchNorm1d(self.args.part_num * self.num_features2),
            nn.Linear(self.args.part_num * self.num_features2, self.complex_embed),
            nn.ELU(inplace=True)
        )

        self.attention = nn.Parameter(torch.randn([1, self.num_features, 1, 1]))



    def integration(self, layer1, layer2):

        batch_size = layer1.size(0)
        channel_num = layer1.size(1)
        disturb_num = layer2.size(1)
        layer1 = layer1.unsqueeze(2)
        layer2 = layer2.unsqueeze(1)

        sum_of_weight = layer2.view(batch_size, disturb_num, -1).sum(-1) + 0.00001
        vec = (layer1 * layer2).reshape(batch_size, channel_num, disturb_num, -1).sum(-1)
        vec = vec / sum_of_weight.unsqueeze(1)
        vec = vec.view(batch_size, channel_num*disturb_num)
        return vec




    def forward_metric(self, x):

        x, part_vector = self.encode(x)

        x1 = F.normalize(x, p=2, dim=-1)
        x2 = F.normalize(part_vector, p=2, dim=-1)

        x = self.args.temperature * F.linear(x1, F.normalize(self.fc.weight, p=2, dim=-1))
        part_vector = self.args.temperature * F.linear(x2, F.normalize(self.part_fc.weight, p=2, dim=-1))

        return x, part_vector, x1, x2




    def encode(self, x):
        x, x1 = self.encoder(x)
        batch_size, channels, h, w = x1.shape

        attention = F.leaky_relu(self.attention)
        heat_map = F.interpolate(x, size=(x1.shape[-1], x1.shape[-1]), mode='bilinear', align_corners=False)
        heat_map = heat_map * attention
        heat_map = self.mask_branch(heat_map)
        mask = nn.Sigmoid()(heat_map)
        part_vector = self.integration(x1, mask).view(batch_size, self.num_features2 * self.args.part_num)

        x = self.avgpool(x)
        x = x.squeeze(-1).squeeze(-1)

        x = 0.5 * x + 0.5 * self.both_mlp(x)
        part_vector = 0.5 * part_vector + 0.5 * self.both_mlp1(part_vector)

        return x, part_vector


    def forward(self, input):
        if self.mode != 'encoder':
            input = self.forward_metric(input)
            return input
        elif self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            raise ValueError('Unknown mode')

    def encode1(self, x):
        x, x1 = self.encoder(x)
        x = self.avgpool(x)
        x = x.squeeze(-1).squeeze(-1)
        return x


    def get_logits(self,x,fc):
        return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))

    def update_fc(self,dataloader,class_list,criterion_Cross, session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            size = data.shape[1:]

            data, data2 = self.encode(data)
            data = data.detach()
            data2 = data2.detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
            new_fc2 = nn.Parameter(
                torch.rand(len(class_list), self.num_features2 * self.args.part_num, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc2, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)
            new_fc2 = self.update_fc_avg2(data2, label, class_list)

        if self.args.complex_weight>0 and self.args.epochs_new>0:  # further finetune
            self.update_fc_ft(new_fc, new_fc2, data,data2,criterion_Cross, label,  session)

    def update_fc_avg(self,data,label,class_list):
        new_fc=[]
        for class_index in class_list:
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc

    def update_fc_avg2(self,data,label,class_list):
        new_fc2=[]
        for class_index in class_list:
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc2.append(proto)
            self.part_fc.weight.data[class_index]=proto
        new_fc2=torch.stack(new_fc2,dim=0)
        return new_fc2


    def update_fc_ft(self,new_fc, new_fc2, data,data2, criterion_Cross, label, session):
        new_fc = new_fc.clone().detach()
        new_fc.requires_grad=True
        new_fc2 = new_fc2.clone().detach()
        new_fc2.requires_grad = True

        optimized_parameters = [{'params': new_fc},{'params': new_fc2}]
        optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=0)

        with torch.enable_grad():
            for epoch in range(self.args.epochs_new):
                old_fc = self.fc.weight[:(self.args.base_class + self.args.way * (session - 1)), :].detach()
                part_old_fc = self.part_fc.weight[:(self.args.base_class + self.args.way * (session - 1)), :].detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                part_fc = torch.cat([part_old_fc, new_fc2], dim=0)

                logits1 = self.get_logits(data, fc)
                logits2 = self.get_logits(data2, part_fc)

                loss = criterion_Cross(logits1, logits2) + criterion_Cross(logits2, logits1) + F.cross_entropy(logits1, label) + F.cross_entropy(logits2, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pass

        self.fc.weight.data[(self.args.base_class + self.args.way * (session - 1)):(self.args.base_class + self.args.way * session), :].copy_(new_fc.data)
        self.part_fc.weight.data[(self.args.base_class + self.args.way * (session - 1)):(self.args.base_class + self.args.way * session), :].copy_(new_fc2.data)

