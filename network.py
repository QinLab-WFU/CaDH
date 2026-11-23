import torch
from torch import nn


class CaDH(nn.Module):
    """FUME inference credibility, explicit P/N outputs; no label-conditioned inference."""
    def __init__(self, classes, bits, backbone='resnet50', pretrained=True):
        super().__init__()
        if backbone == 'tiny':
            self.backbone = nn.Sequential(nn.Conv2d(3,8,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten())
            dim = 8
        elif backbone == 'resnet50':
            from torchvision.models import resnet50, ResNet50_Weights
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'vit_b_16':
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
            dim = self.backbone.hidden_dim
            self.backbone.heads = nn.Identity()
        else:
            raise ValueError(backbone)
        self.hash_head = nn.Linear(dim,bits)
        self.credibility_head = nn.Linear(dim,classes)
        for head in [self.hash_head,self.credibility_head]:
            nn.init.xavier_uniform_(head.weight);nn.init.zeros_(head.bias)

    def forward(self,images):
        f = self.backbone(images)
        h = self.hash_head(f).tanh()
        possibility = self.credibility_head(f).sigmoid()
        values,indices = possibility.topk(2,dim=1)
        competitor = values[:,:1].expand_as(possibility).clone()
        competitor.scatter_(1,indices[:,:1],values[:,1:2])
        necessity = 1-competitor.detach()
        credibility = (possibility+necessity)/2
        return {'hash':h,'codes':torch.where(h>=0,1.,-1.),'membership':possibility,
                'possibility':possibility,'necessity':necessity,'credibility':credibility}
