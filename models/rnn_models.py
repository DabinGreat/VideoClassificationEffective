import torch
from torch import nn


class ResnetBlock(nn.Module):
    def __init__(self, inchannel, mid_channel, outchannel, ds_stride=2):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(inchannel, mid_channel, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid_channel)
        self.conv2 = nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=ds_stride, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channel)
        self.conv3 = nn.Conv2d(inchannel, outchannel, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(outchannel)
        self.ds_conv = nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=ds_stride)
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


class ConvLSTMcell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=[3, 3], bias=True):
        super(ConvLSTMcell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        self.bn = nn.BatchNorm2d(self.hidden_dim*4)
        self.relu = nn.ReLU()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):dabi
        #         nn.init.normal_(m.weight.data, 0, 0.001)
        #         nn.init.constant_(m.bias.data, 0)

    def forward(self, input_tensor, last_cell_state):
        h_last, c_last = last_cell_state

        if h_last is None:
            h_last = self.init_hidden(batch_size=input_tensor.size(0), height=input_tensor.size(2),
                                      width=input_tensor.size(3))
        if c_last is None:
            c_last = self.init_hidden(batch_size=input_tensor.size(0), height=input_tensor.size(2),
                                      width=input_tensor.size(3))

        combined = torch.cat([input_tensor, h_last], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        # combined_bn = self.bn(combined_conv)
        # combined_relu = self.relu(combined_bn)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_last + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, [h_next.detach(), c_next.detach()]

    def init_hidden(self, batch_size, height, width):
        return torch.zeros((batch_size, self.hidden_dim, height, width)).cuda()


class RNNcell(nn.Module):
    def __init__(self, inchannel, midchannel, outchannel, ds_stride):
        super(RNNcell, self).__init__()
        self.resblockup = ResnetBlock(inchannel, inchannel, inchannel, ds_stride=1)
        self.rnncell = ConvLSTMcell(inchannel, inchannel, [3,3])
        self.resblockdown = ResnetBlock(inchannel, midchannel, outchannel, ds_stride=ds_stride)

    def forward(self, input, last_cell_state):
        out = self.resblockup(input)
        out, next_cell_state = self.rnncell(out, last_cell_state)
        out = self.resblockdown(out)
        return out, next_cell_state
