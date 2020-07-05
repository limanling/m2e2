import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class FMapLayerImage(nn.Module):
    def __init__(self, fine_tune=False, dropout=0.5,
                 device=torch.device("cpu"), backbone='resnet152'):

        super(FMapLayerImage, self).__init__()

        net = getattr(models, backbone)(pretrained=True)
        #print(net)
        if backbone=='vgg16':
            b1, pool, b2 = list(net.children())
            modules_1 = list(b1.children()) 
            modules_2 = [pool, Flatten()] + list(b2.children())[:-1]
            
        else:
            b1 = list(net.children())
            modules_1 = b1[:-2]
            modules_2 = [b1[-2]]
            
        
        self.backbone = nn.Sequential(*modules_1)
        self.pooler = nn.Sequential(*modules_2)
        #print(self.backbone, self.pooler)


        for p in self.backbone.parameters():
            p.requires_grad = fine_tune
        for p in self.pooler.parameters():
            p.requires_grad = fine_tune

        self.device = device
        self.to(device)

    def forward(self, images):
        fmap = self.backbone(images)
        emb = self.pooler(fmap)
        return fmap, emb