import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from net_utils import GradientReversalFunction
from torch.optim import lr_scheduler
from resnet_custom import *


class Mine(nn.Module):
    """Code from https://github.com/sungyubkim/MINE-Mutual-Information-Neural-Estimation-"""
    def __init__(self, x_size=2048, y_size=2048, output_size=1, hidden_size=256):
        super(Mine, self).__init__()
        self.fc1_x = nn.Linear(x_size, hidden_size, bias=False)
        self.fc1_y = nn.Linear(y_size, hidden_size, bias=False)
        self.fc1_bias = nn.Parameter(torch.zeros(hidden_size))
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x, y):
        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)
        x = self.fc1_x(x)
        y = self.fc1_y(y)
        mi = F.leaky_relu(x + y + self.fc1_bias, negative_slope=2e-1)
        mi = F.leaky_relu(self.fc2(mi), negative_slope=2e-1)
        mi = F.leaky_relu(self.fc3(mi), negative_slope=2e-1)
        return mi


class FeatureExtractor(nn.Module):
    def __init__(self, input_nc=3, output_nc=128, avgPool=False):
        super(FeatureExtractor, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_nc, output_nc // 4, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(output_nc // 4), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.conv_2 = nn.Sequential(
            nn.Conv2d(output_nc // 4, output_nc // 2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(output_nc // 2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        if avgPool:
            self.conv_3 = nn.Sequential(
                nn.Conv2d(output_nc // 2, output_nc, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(output_nc), nn.ReLU(),
                nn.AdaptiveAvgPool2d(7))
        else:
            self.conv_3 = nn.Sequential(
                nn.Conv2d(output_nc // 2, output_nc, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(output_nc), nn.ReLU())

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x


"""
class ResnetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(ResnetFeatureExtractor, self).__init__()
        self.model = resnet101(pretrained=pretrained)
        self.feature_size = self.model.feature_size

    def forward(self, x):
        return self.model(x)
"""

def resnetFeatureExtractor(pretrained=True, name='resnet101'):
    if name == 'resnet101':
        model = resnet101(pretrained=pretrained)
    elif name == 'resnet18':
        model = resnet18(pretrained=pretrained)
    elif name == 'resnet34':
        model = resnet34(pretrained=pretrained)
    elif name == 'resnet50':
        model = resnet50(pretrained=pretrained)

    return model

class GRL(nn.Module):
    def __init__(self, lambda_=1):
        super(GRL, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class Classifier(nn.Module):
    def __init__(self, input_nc, output_nc, resnet=False):
        super(Classifier, self).__init__()
        layers = []
        if resnet:
            layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(input_nc, output_nc))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc, output_nc=1, hidden_size=256, resnet=False):
        super(Discriminator, self).__init__()
        layers = []
        if resnet:
            layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Sequential(
            nn.Linear(input_nc, hidden_size), 
            nn.LeakyReLU(negative_slope=2e-1)))
        layers.append(nn.Linear(hidden_size, output_nc))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SelfAttention(nn.Module):
    def __init__(self, input_nc):
        super(SelfAttention, self).__init__()
        output_nc = max(input_nc // 8, 1)
        self.query_conv = nn.Conv2d(input_nc, output_nc, kernel_size=1)
        self.key_conv = nn.Conv2d(input_nc, output_nc, kernel_size=1)
        self.value_conv = nn.Conv2d(input_nc, input_nc, kernel_size=1)
        self.gamma = nn.Parameter(torch.randn(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, W, H = x.size()
        proj_query = self.query_conv(x).view(B, -1, W*H).permute(0,2,1)
        proj_key = self.key_conv(x).view(B, -1, W*H)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(B, -1, W*H)

        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(B, C, W, H)
        out = self.gamma * out + x

        return out


class CustomAttention(nn.Module):
    def __init__(self, input_nc):
        super(CustomAttention, self).__init__()
        output_nc = max(input_nc // 8, 1)
        self.query_conv = nn.Conv2d(input_nc, output_nc, kernel_size=1)
        self.key_conv = nn.Conv2d(input_nc, output_nc, kernel_size=1)
        self.value_conv = nn.Conv2d(input_nc, input_nc, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, att, x):
        B, C, W, H = x.size()
        proj_query = self.query_conv(att).view(B, -1, W*H).permute(0,2,1)
        proj_key = self.key_conv(att).view(B, -1, W*H)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(B, -1, W*H)

        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(B, C, W, H)
        out = self.gamma * out + x

        return out


class FeatureShifter_CNN(nn.Module):
    def __init__(self, input_nc, output_nc=None, n_layers=2):
        super(FeatureShifter_CNN, self).__init__()
        if output_nc is None:
            output_nc = input_nc
        layers = []
        input_nc *= 2
        for _ in range(n_layers):
            layers += [nn.Conv2d(input_nc, output_nc, kernel_size=5, stride=1, padding=2),
                           nn.BatchNorm2d(output_nc), nn.ReLU()]
            input_nc = output_nc
        
        self.layers = nn.Sequential(*layers)

    def forward(self, att, x):
        x = torch.cat((att, x), dim=1)
        return self.layers(x)


class FeatureShifter_SA(nn.Module):
    """Self-attention network (Not use for now)"""
    def __init__(self, input_nc, output_nc=None, n_layers=1, conv_layers=0):
        super(FeatureShifter_SA, self).__init__()
        if output_nc is None:
            output_nc = input_nc // 2
        layers = []
        for _ in range(n_layers):
            layers.append(SelfAttention(input_nc))
            if conv_layers > 0:
                layers += [nn.Conv2d(input_nc, output_nc, kernel_size=5, stride=1, padding=2),
                           nn.BatchNorm2d(output_nc), nn.ReLU()]
                input_nc = output_nc
                conv_layers -= 1
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FeatureShifter_Att(nn.Module):
    """Custom Attention based project module"""
    def __init__(self, input_nc, output_nc=None):
        super(FeatureShifter_Att, self).__init__()
        if output_nc is None:
            output_nc = input_nc

        self.attn1 = CustomAttention(input_nc)
        self.layer1 = nn.Sequential(nn.Conv2d(input_nc, output_nc, kernel_size=5, stride=1, padding=2),
                                    nn.BatchNorm2d(output_nc), nn.ReLU())
        self.attn2 = CustomAttention(output_nc)

    def forward(self, att, x):
        x = self.attn1(att, x)
        x = self.layer1(x)
        x = self.attn2(att, x)
        return x


class FeatureShifter_FC(nn.Module):
    """Fully-Connected version of feature shifter"""
    def __init__(self, input_nc, output_nc, hidden_size=3072, n_layers=2, use_droput=False, resnet=False):
        super(FeatureShifter_FC, self).__init__()
        self.resnet = resnet
        if resnet:
            self.avgPool = nn.AdaptiveAvgPool2d(1)

        if n_layers >= 2:
            layers = [
                nn.Linear(input_nc, hidden_size),
                nn.BatchNorm1d(hidden_size), nn.ReLU()]
            if use_droput:
                layers.append(nn.Dropout())
            layers += [
                nn.Linear(hidden_size, output_nc),
                nn.BatchNorm1d(output_nc), nn.ReLU()]
        else:
            layers = [
                nn.Linear(input_nc, output_nc),
                nn.BatchNorm1d(output_nc), nn.ReLU()]
            
        self.layers = nn.Sequential(*layers)
    
    def forward(self, att, x):
        if self.resnet:
            att = self.avgPool(x)
            x = self.avgPool(x)
        
        x = torch.cat((att, x), dim=1)
        x = torch.flatten(x, start_dim=1)
        x = self.layers(x)
        return x


class FeatureExtractorSA(nn.Module):
    def __init__(self, input_nc=3, output_nc=128, n_layers=1, conv_layers=0):
        super(FeatureExtractorSA, self).__init__()
        layers = []
        layers.append(FeatureExtractor(input_nc, output_nc))
        layers.append(FeatureShifter_SA(output_nc, n_layers, conv_layers))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


"""
Decoder networks
"""

class ResnetDecoder(nn.Module):
    def __init__(self, input_nc, output_nc=3, n_blocks=2, upsample=2):
        super(ResnetDecoder, self).__init__()
        layers = []
        nc = input_nc

        for i in range(n_blocks):
            layers += [ResnetBlock(nc)]

        for i in range(upsample):
            layers += [nn.ConvTranspose2d(nc, nc // 2, kernel_size=3, stride=2,
                                          padding=1, output_padding=1),
                       nn.BatchNorm2d(nc // 2),
                       nn.ReLU(True)]
            nc = nc // 2
        
        layers += [nn.ReflectionPad2d(3)]
        layers += [nn.Conv2d(nc, output_nc, kernel_size=7, padding=0)]
        layers += [nn.Tanh()]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""
    def __init__(self, dim, use_dropout=False):
        super(ResnetBlock, self).__init__()
        conv_block = []
        p = 1

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), nn.BatchNorm2d(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), nn.BatchNorm2d(dim)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


def define_net(net_type, input_nc, output_nc, cfg, hidden_size=None):
    if net_type == 'G':
        net = FeatureExtractor(input_nc)
    elif net_type == 'I':
        net = Classifier(input_nc, output_nc)

    return init_net(net, cfg.init_type, cfg.init_gain, cfg.gpu_ids)


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    if init_type is not None:
        init_weights(net, init_type, init_gain=init_gain)
    return net


def get_scheduler(optimizer, cfg):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        cfg (config class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              cfg.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <cfg.n_epochs> epochs
    and linearly decay the rate to zero over the next <cfg.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if cfg.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + cfg.epoch_count - cfg.n_epochs) / float(cfg.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif cfg.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.lr_decay_iters, gamma=0.1)
    elif cfg.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif cfg.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', cfg.lr_policy)
    return scheduler