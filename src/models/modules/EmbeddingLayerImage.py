import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class EmbeddingLayerImage(nn.Module):
    def __init__(self, fine_tune=False, dropout=0.5,
                 device=torch.device("cpu"), backbone='resnet152'):

        super(EmbeddingLayerImage, self).__init__()

        resnet = getattr(models, backbone)(pretrained=True)
        #print(resnet)
        if backbone=='vgg16':
            b1, pool, b2 = list(resnet.children())
            modules = list(b1.children()) + [pool, Flatten()] + list(b2.children())[:-1]
            self.dropout = None
        else:
            modules = list(resnet.children())[:-1]
            self.dropout = dropout if type(dropout) == float and -1e-7 < dropout < 1 + 1e-7 else None
        
        self.resnet = nn.Sequential(*modules)
        #print(self.resnet)


        for p in self.resnet.parameters():
            p.requires_grad = fine_tune

        self.device = device
        self.to(device)

    def forward(self, images):
        if self.dropout is not None:
            return F.dropout(self.resnet(images), p=self.dropout, training=self.training)
        else:
            return self.resnet(images)  # batchsize * 2048 * imageLength * imageLength (imageLength=1)