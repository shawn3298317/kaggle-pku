from efficientnet_pytorch import EfficientNet
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

IMG_WIDTH = 2048 - 512
IMG_HEIGHT = IMG_WIDTH // 16 * 5
MODEL_SCALE = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda:2")

def get_mesh(batch_size, shape_x, shape_y):
    '''
    Params :
        batch_size : Integer
            Represents batch size of the data
        shape_x : ?
            Shape of x
        shape_y : ?
            Shape of y
    Returns :

    '''
    mg_x, mg_y = np.meshgrid(np.linspace(0, 1, shape_y), np.linspace(0, 1, shape_x))
    mg_x = np.tile(mg_x[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mg_y = np.tile(mg_y[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mesh = torch.cat([torch.tensor(mg_x).to(device), torch.tensor(mg_y).to(device)], 1)
    return mesh

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    """
        Class that performs upsampling
    """
    def __init__(self, in_ch, x_in_ch, out_ch, bilinear=False):
        """
            Define the upsampling and conv layers, inspired by Unet
            Params :
                in_ch : int
                    Number of input depth channels for the conv layer
                out_ch : int
                    Number of output depth channels for conv layer
                x_in_ch : int
                    Number of input and output depth channels for the upsampling transpose convolutions layer
                bilinear : bool
                    False : use transpose convolutions
                    True : use bilinear intrapolation
        """
        super(up, self).__init__()

        if bilinear:
            # keeps the same input output shape, just increases width and height
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up=nn.ConvTranspose2d(x_in_ch, x_in_ch, 2, stride=2)

        # could use both a bilinear first, then use a learnt upsampling
        # https://medium.com/activating-robotic-minds/up-sampling-with-transposed-convolution-9ae4f2df52d0
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2=None):
        """
            Params :
                x1 : numpy array of shape (batch_size,chanels,width,height)
                    Feature Map produced by a CNN
                x2 : numpy array of shape (batch_size,chanels,width,height)
                    Feature Map produced by a CNN
            Returns :
                x : numpy array of shape (batch_size,chanels,width,height)
                    Performs upsampling using x1, conncats x1 and x2 and convolves it
        """
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        
        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        x = self.conv(x)
        return x

class MyUNet(nn.Module):
    '''Mixture of previous classes'''
    def __init__(self, n_classes):
        super(MyUNet, self).__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        
        # double conv uses a 3x3 kernel with padding =1, stride =1 

        # intutively the number of classes should be 2 -> object or not
        print('The number of classes is', n_classes)

        self.conv0 = double_conv(5, 64)
        self.conv1 = double_conv(64, 128)
        self.conv2 = double_conv(128, 512)
        self.conv3 = double_conv(512, 1024)
        
        # kernel size = stride = 2
        self.mp = nn.MaxPool2d(2)
        
        # self.up1 = up(1282 + 1024, 512)
        self.up1 = up(1282 + 1024, 1282, 512)

        # self.up2 = up(512 + 512, 256)
        self.up2 = up(512 + 512, 512 , 256)

        self.up3=up(128+256,256,128)

        # self.outc = nn.Conv2d(256, n_classes, 1)
        self.outc = nn.Conv2d(128, n_classes, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        print("Input shape to Unet", x.shape)
        mesh1 = get_mesh(batch_size, x.shape[2], x.shape[3])
        print("mesh1 shape",mesh1.shape)
        x0 = torch.cat([x, mesh1], 1)
        x1 = self.mp(self.conv0(x0))
        x2 = self.mp(self.conv1(x1))
        x3 = self.mp(self.conv2(x2))
        x4 = self.mp(self.conv3(x3))
        print("Done woth all the mynet layer")
        print()
        x_center = x[:, :, :, IMG_WIDTH // MODEL_SCALE: -IMG_WIDTH // MODEL_SCALE]
        # print("\n Input shape to Efficient Net B0",x_center.shape)
        feats = self.base_model.extract_features(x_center)
        # print("feats original shape",feats.shape)
        bg = torch.zeros([feats.shape[0], feats.shape[1], feats.shape[2], feats.shape[3] // MODEL_SCALE]).to(device)
        # print("bg shape", bg.shape)
        feats = torch.cat([bg, feats, bg], 3)
        # print("Feats after first concat shape bg, feats, bg",feats.shape)
        # Add positional info
        mesh2 = get_mesh(batch_size, feats.shape[2], feats.shape[3])
        # print("Mesh shape",mesh2.shape)
        feats = torch.cat([feats, mesh2], 1)
        # print("Feats shape",feats.shape)
        # print(" X4 shape ",x4.shape)
        # print(" x_center shape",x_center.shape)

        x = self.up1(feats, x4)
        x = self.up2(x, x3)
        print("entering experimental phase")
        x = self.up3(x, x2)
        x = self.mp(x)
        print("Sucess")
        x = self.outc(x)
        return x

def criterion(prediction, mask, regr, size_average=True):
    # Binary mask loss
    pred_mask = torch.sigmoid(prediction[:, 0])
    # mask_loss = mask * (1 - pred_mask)**2 * torch.log(pred_mask + 1e-12) + (1 - mask) * pred_mask**2 * torch.log(1 - pred_mask + 1e-12)
    mask_loss = mask * torch.log(pred_mask + 1e-12) + (1 - mask) * torch.log(1 - pred_mask + 1e-12)
    mask_loss = -mask_loss.mean(0).sum()
    
    # Regression L1 loss
    pred_regr = prediction[:, 1:]
    regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
    regr_loss = regr_loss.mean(0)
    
    # Sum
    loss = mask_loss + regr_loss
    if not size_average:
        loss *= prediction.shape[0]
    return loss

