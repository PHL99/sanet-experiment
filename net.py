import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor


### Helper functions ###
def channel_norm(x, eps=1e-5):
    # Channel-wise mean-variance normalization
    x_mn = x.mean(dim=(-1, -2), keepdim=True)
    x_std = (x.var(dim=(-1, -2), keepdim=True) + eps).sqrt()    # add eps to avoid div by zero
    return (x - x_mn) / x_std     # broadcast to image dim


def pretrained_encoder(encoder_path="vgg_normalised.pth"):
    # normalized vgg from Github user https://github.com/GlebSBrykin/SANET
    vgg = nn.Sequential(
        nn.Conv2d(3, 3, kernel_size=1),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, kernel_size=3),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, kernel_size=3),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, kernel_size=3),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, kernel_size=3),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, kernel_size=3),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, kernel_size=3),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, kernel_size=3),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, kernel_size=3),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, kernel_size=3),
        nn.ReLU(),  # relu4-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, kernel_size=3),
        nn.ReLU(),  # relu4-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, kernel_size=3),
        nn.ReLU(),  # relu4-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, kernel_size=3),
        nn.ReLU(),  # relu4-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, kernel_size=3),
        nn.ReLU(),  # relu5-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, kernel_size=3),
        nn.ReLU(),  # relu5-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, kernel_size=3),
        nn.ReLU(),  # relu5-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, kernel_size=3),
        nn.ReLU()   # relu5-4
    )
    vgg.load_state_dict(torch.load(encoder_path))
    for p in vgg.parameters():
        p.requires_grad = False
    vgg.eval()
    return vgg


def pretrained_decoder(decoder_path="decoder_iter_100000.pth", trainable=False):
    # pretrained decoder from Github user https://github.com/GlebSBrykin/SANET
    decoder = nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 256, kernel_size=3),
        nn.ReLU(),
        nn.Upsample(scale_factor=2),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, kernel_size=3),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, kernel_size=3),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, kernel_size=3),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 128, kernel_size=3),
        nn.ReLU(),
        nn.Upsample(scale_factor=2),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, kernel_size=3),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 64, kernel_size=3),
        nn.ReLU(),
        nn.Upsample(scale_factor=2),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, kernel_size=3),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 3, kernel_size=3),
    )
    decoder.load_state_dict(torch.load(decoder_path))
    if not trainable:
        for p in decoder.parameters():
            p.requires_grad = False
        decoder.eval()
    return decoder


### Model ###
class SANet(nn.Module):
    def __init__(self, in_channels):
        super(SANet, self).__init__()
        # Learnable weights
        self.conv_c1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_s1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_s2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_csc = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, f_c, f_s):
        assert (f_c.size() == f_s.size())
        B, C, H, W = f_c.size()
        f_c_hat, f_s_hat = channel_norm(f_c), channel_norm(f_s)
        f = self.conv_c1(f_c_hat).view(B, C, H*W)
        g = self.conv_s1(f_s_hat).view(B, C, H*W)
        attention = F.softmax(torch.bmm(f.mT, g), dim=-1)
        h = self.conv_s2(f_s_hat).view(B, C, H*W)
        out = torch.bmm(h, attention.mT).view(B, C, H, W)
        f_csc = self.conv_csc(out) + f_c
        return f_csc
    

class SANTransfer(nn.Module):
    def __init__(self, train_decoder=False):
        super(SANTransfer, self).__init__()
        # Feature mapping
        self.encode = create_feature_extractor(pretrained_encoder(), return_nodes={
            "3": "relu1_1",
            "10": "relu2_1",
            "17": "relu3_1",
            "30": "relu4_1",
            "43": "relu5_1"
        })
        self.decode = pretrained_decoder(trainable=train_decoder)
        # Output construction
        self.sanet_r4 = SANet(512)
        self.sanet_r5 = SANet(512)
        self.conv_m_csc = nn.Conv2d(512, 512, kernel_size=3, padding=1, padding_mode='reflect')

    def encode_img(self, i):
        with torch.no_grad():
            e_i = self.encode(i)
        return e_i
    
    def transfer(self, i_c, i_s, return_encode=False):
        # Encode
        e_c, e_s = self.encode_img(i_c), self.encode_img(i_s)
        # SANet r4_1
        f_r4_c, f_r4_s = e_c['relu4_1'], e_s['relu4_1']
        f_r4_csc = self.sanet_r4(f_r4_c, f_r4_s)
        # SANet r5_1
        f_r5_c, f_r5_s = e_c['relu5_1'], e_s['relu5_1']
        f_r5_csc = self.sanet_r5(f_r5_c, f_r5_s)
        # Raw output
        f_m_csc = self.conv_m_csc(f_r4_csc + F.interpolate(f_r5_csc, scale_factor=2))
        # Decode
        i_cs = self.decode(f_m_csc)
        return i_cs if not return_encode else (i_cs, (e_c, e_s))

    def get_identity(self, i_c, i_s):
        return (self.transfer(i_c, i_c), self.transfer(i_s, i_s))

    def forward(self, i_c, i_s, return_encode=False):
        return self.transfer(i_c, i_s, return_encode)
    
