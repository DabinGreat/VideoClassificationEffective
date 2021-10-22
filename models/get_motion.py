from torch import nn


class PA(nn.Module):
    def __init__(self, inchannel, midchannel, outchannel, ds_stride=2):
        super(PA, self).__init__()
        self.pa_conv = nn.Conv2d(inchannel, midchannel, kernel_size=7, stride=1, padding=3)
        self.pa_bn = nn.BatchNorm2d(midchannel)
        self.ds_conv = nn.Conv2d(midchannel, outchannel, kernel_size=3, stride=ds_stride, padding=1)
        self.ds_bn = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.001)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, past_x, next_x):
        past_x_pa = self.pa_conv(past_x)
        next_x_pa = self.pa_conv(next_x)
        PA = next_x_pa - past_x_pa
        PA = self.pa_bn(PA)
        PA = self.relu(PA)
        PA = self.ds_conv(PA)
        PA = self.ds_bn(PA)
        PA = self.relu(PA)
        return PA


class RGBDiff(nn.Module):
    def __init__(self, inchannel, midchannel, outchannel, ds_stride=2):
        super(RGBDiff, self).__init__()
        self.diff_conv = nn.Conv2d(inchannel, midchannel, kernel_size=7, stride=1, padding=3)
        self.diff_bn = nn.BatchNorm2d(midchannel)
        self.ds_conv = nn.Conv2d(midchannel, outchannel, kernel_size=3, stride=ds_stride, padding=1)
        self.ds_bn = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.001)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, past_x, next_x):
        RGBDiff = next_x - past_x
        RGBDiff = self.diff_conv(RGBDiff)
        RGBDiff = self.diff_bn(RGBDiff)
        RGBDiff = self.relu(RGBDiff)
        RGBDiff = self.ds_conv(RGBDiff)
        RGBDiff = self.ds_bn(RGBDiff)
        RGBDiff = self.relu(RGBDiff)
        return RGBDiff


class FixRGBchannel(nn.Module):
    def __init__(self, inchannel, midchannel, outchannel, ds_stride=2):
        super(FixRGBchannel, self).__init__()
        self.conv = nn.Conv2d(inchannel, midchannel, kernel_size=7, stride=1, padding=3)
        self.bn = nn.BatchNorm2d(midchannel)
        self.ds_conv = nn.Conv2d(midchannel, outchannel, kernel_size=3, stride=ds_stride, padding=1)
        self.ds_bn = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.001)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, next_x):
        x = self.conv(next_x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.ds_conv(x)
        x = self.ds_bn(x)
        x = self.relu(x)
        return x