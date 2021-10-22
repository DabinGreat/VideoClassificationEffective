from torch import nn
from .rnn_models import RNNcell
from .get_motion import FixRGBchannel
from .fusion import Conv2Linear


class OnlyRNN(nn.Module):
    def __init__(self,
                 num_class,
                 motion_type='PA',
                 rnn_model='lstm',
                 pool_type='view'):
        super(OnlyRNN, self).__init__()
        self.num_class = num_class
        self.motion_type = motion_type
        self.rnn_model = rnn_model
        self.pool_type = pool_type

        print('-' * 60)
        print("""
                Initializing FOL with base model:
                OnlyRNN Configurations:
                        rnn_model:          {}  
                        motion_type:        {}
                        output_num_class:   {}
        """.format(self.rnn_model,
                   self.motion_type,
                   self.num_class))
        print('-' * 60)

        self._prepare_base_model()

    def _prepare_base_model(self):
        # initialize rnn model
        if 'lstm' in self.rnn_model:
            self.rnn_layer1 = RNNcell(16, 16, 32, 2)
            self.rnn_layer2 = RNNcell(32, 32, 64, 2)
            self.rnn_layer3 = RNNcell(64, 64, 256, 2)
            self.rnn_layer4 = RNNcell(256, 256, 512, 2)
        elif 'gru' in self.rnn_model:
            pass
        else:
            raise RuntimeError("Not support this rnn model type '{}' yet!"
                               .format(self.base_model))
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
        rnn_layer1_x, cell_info_1 = self.rnn_layer1(motion_feature, last_cell_info[0])

        # cnn and rnn layer2 forward
        rnn_layer2_x, cell_info_2 = self.rnn_layer2(rnn_layer1_x, last_cell_info[1])

        # cnn and rnn layer3 forward
        rnn_layer3_x, cell_info_3 = self.rnn_layer3(rnn_layer2_x, last_cell_info[2])

        # cnn and rnn layer4 forward
        rnn_layer4_x, cell_info_4 = self.rnn_layer4(rnn_layer3_x, last_cell_info[3])

        # conv to linear layer
        out = self.conv2linear(rnn_layer4_x)

        # collect rnn_cell_info
        rnn_cell_info = [cell_info_1, cell_info_2, cell_info_3, cell_info_4]
        return out, rnn_cell_info








