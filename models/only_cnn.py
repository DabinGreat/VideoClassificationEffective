from torch import nn
from .cnn_models import CNNcell
from .get_motion import FixRGBchannel
from .fusion import Conv2Linear


class OnlyCNN(nn.Module):
    def __init__(self,
                 num_class,
                 motion_type='RGB',
                 cnn_model='resnet',
                 pool_type='maxpool'):
        super(OnlyCNN, self).__init__()
        self.num_class = num_class
        self.motion_type = motion_type
        self.cnn_model = cnn_model
        self.pool_type = pool_type

        print('-' * 60)
        print("""
                Initializing FAST with base model:
                OnlyCNN Configurations:
                        cnn_model:          {}
                        motion_type:        {}
                        output_num_class:   {}
        """.format(self.cnn_model,
                   self.motion_type,
                   self.num_class))
        print('-' * 60)

        self._prepare_base_model()

    def _prepare_base_model(self):
        # initialize cnn model
        if 'resnet' in self.cnn_model:
            self.cnn_layer1 = CNNcell(16, 16, 32, 2)
            self.cnn_layer2 = CNNcell(32, 32, 64, 2)
            self.cnn_layer3 = CNNcell(64, 64, 256, 2)
            self.cnn_layer4 = CNNcell(256, 256, 512, 2)
        else:
            raise RuntimeError("Not support this type cnn model '{}' yet!"
                               .format(self.cnn_model))
        # initialize motion features model
        if self.motion_type != 'RGB':
            raise RuntimeError("OnlyRNN model just support RGB motion type!")
        self.fix_rgb_channel = FixRGBchannel(3, 8, 16)
        # initialize head layers
        self.conv2linear = Conv2Linear(512, self.num_class, self.pool_type)

    def forward(self, last_cell_info, past_x, next_x):
        # get motion features
        fix_rgb = self.fix_rgb_channel(next_x)
        motion_feature = fix_rgb

        # cnn and rnn layer1 forward
        cnn_layer1_x = self.cnn_layer1(motion_feature)

        # cnn and rnn layer2 forward
        cnn_layer2_x = self.cnn_layer2(cnn_layer1_x)

        # cnn and rnn layer3 forward
        cnn_layer3_x = self.cnn_layer3(cnn_layer2_x)

        # cnn and rnn layer4 forward
        cnn_layer4_x = self.cnn_layer4(cnn_layer3_x)

        # conv to linear layer
        out = self.conv2linear(cnn_layer4_x)

        # collect rnn_cell_info
        rnn_cell_info = last_cell_info
        return out, rnn_cell_info








