from torch import nn


class Fusion(nn.Module):
    def __init__(self, consensus):
        super(Fusion, self).__init__()
        self.consensus = consensus

    def forward(self, cnn_in, rnn_in):
        if self.consensus == 'cnn2rnn_add':
            out = rnn_in + cnn_in
            return [cnn_in, out]
        else:
            raise RuntimeError("Not support this consensus type '{}' yet!"
                               .format(self.consensus))


class Conv2Linear(nn.Module):
    def __init__(self, in_channel, num_class, pool_type):
        super(Conv2Linear, self).__init__()
        self.pool_type = pool_type
        if pool_type == 'view':
            mid_channel = in_channel * 49
        elif pool_type == 'maxpool' or 'avgpool':
            mid_channel = in_channel
        else:
            raise RuntimeError("Not support this pool type '{}' yet!"
                               .format(self.pool_type))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(mid_channel, num_class)

    def forward(self, x):
        if self.pool_type == 'view':
            out = x.view(x.size()[0], -1)
            out = self.fc(out)
        elif self.pool_type == 'maxpool':
            out = self.maxpool(x)
            out = out.view(x.size()[0], -1)
            out = self.fc(out)
        elif self.pool_type == 'avgpool':
            out = self.avgpool(x)
            out = out.view(x.size()[0], -1)
            out = self.fc(out)
        return out
