from torch import nn
import torchvision


class ResnetBlock(nn.Module):
    def __init__(self, inchannel, mid_channel, outchannel, stride=2):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(inchannel, mid_channel, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid_channel)
        self.conv2 = nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channel)
        self.conv3 = nn.Conv2d(inchannel, outchannel, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(outchannel)
        self.ds_conv = nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride)
        self.ds_bn = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.ds_conv(identity)
        identity = self.ds_bn(identity)

        out += identity
        out = self.relu(out)

        return out


class CNNcell(nn.Module):
    def __init__(self, inchannel, mid_channel, outchannel, ds_stride=2):
        super(CNNcell, self).__init__()
        self.resblock1 = ResnetBlock(inchannel, inchannel, inchannel, stride=1)
        self.resblock2 = ResnetBlock(inchannel, mid_channel, outchannel, stride=ds_stride)
        self.resblock3 = ResnetBlock(outchannel, outchannel, outchannel, stride=1)

    def forward(self, input):
        out = self.resblock1(input)
        out = self.resblock2(out)
        out = self.resblock3(out)
        return out


class StandardResnet(nn.Module):
    def __init__(self, num_class=101, pretrained=True):
        super(StandardResnet, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=pretrained)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.sub_resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, num_class)

    def forward(self, input):
        out = self.sub_resnet(input)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out