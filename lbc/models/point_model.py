import torch
from torch import nn
from common.resnet import resnet18, resnet34
from common.normalize import Normalize
from common.segmentation import SegmentationHead
from .spatial_softmax import SpatialSoftmax


class PointModel(nn.Module):
    def __init__(self, backbone, pretrained=False, height=96, width=96, input_channel=3, output_channel=20, num_labels=7, need_prob=False):
        super().__init__()
        
        self.kh = height//32
        self.kw = width//32
        self.num_labels = num_labels
        self.need_prob = need_prob
        
        self.backbone = eval(backbone)(pretrained=pretrained, num_channels=input_channel)

        self.spd_encoder = nn.Sequential(
            nn.Linear(1,128),
            nn.ReLU(True),
            nn.Linear(128,128),
        )
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(640,256,3,2,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256,128,3,2,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128,64,3,2,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64,output_channel,1,1,0),
            SpatialSoftmax(height//4, width//4, need_prob=self.need_prob)
        )
        
    def forward(self, bev, spd):
        bev = (bev>0).float()#二值化
        inputs = self.backbone(bev/255.)#二值化bev图进行归一化,输入backbone
        spd_embds = self.spd_encoder(spd[:,None])[...,None,None].repeat(1,1,self.kh,self.kw)
        outputs = self.upconv(torch.cat([inputs, spd_embds], 1))
        
        return outputs

class RGBPointModel(PointModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.seg_head = SegmentationHead(512, self.num_labels)
        self.img_size = nn.Parameter(torch.tensor([self.kw*32,self.kh*32]).float(), requires_grad=False)
        
    def forward(self, rgb, spd, pred_seg=True):
        if self.need_prob:
            pred_seg = False
        inputs = self.backbone(self.normalize(rgb/255.))
        spd_embds = self.spd_encoder(spd[:,None])[...,None,None].repeat(1,1,self.kh,self.kw)
        prob_points = self.upconv(torch.cat([inputs, spd_embds], 1))
        
        
        
        if self.need_prob:
            return prob_points
        else:
            prob_points[...,1] = (prob_points[...,1] + 1)/2
            if pred_seg:
                segs = self.seg_head(inputs)
                return prob_points, segs
            else:
                return prob_points
