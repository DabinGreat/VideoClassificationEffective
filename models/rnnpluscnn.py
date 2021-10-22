import torch
from torch import nn
from .rnn_models import RNNcell
from .cnn_models import CNNcell
from .get_motion import PA, RGBDiff, FixRGBchannel
from .fusion import Fusion, Conv2Linear


class RNNplusCNN(nn.Module):
    def __init__(self,
                 num_class,
                 motion_type='PA',
                 rnn_model='lstm',
                 cnn_model='resnet',
                 consensus_type='cnn2rnn_add',
                 pool_type='view'):
        super(RNNplusCNN, self).__init__()
        self.num_class = num_class
        self.motion_type = motion_type
        self.rnn_model = rnn_model
        self.cnn_model = cnn_model
        self.consensus_type = consensus_type
        self.pool_type = pool_type

        print('-' * 30)
        print("""
                Initializing FOL with base model: rnn'{}' + cnn'{}'.
                FAST Configurations:
                        motion_type:        {}
                        output_num_class:   {}
                        consensus_module:   {}
        """.format(self.rnn_model,
                   self.cnn_model,
                   self.motion_type,
                   self.num_class,
                   self.consensus_type))
        print('-' * 30)

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
        if 'PA' in self.motion_type:
            self.PA = PA(3, 8, 16)
        elif 'RGBDiff' in self.motion_type:
            self.RGBDiff = RGBDiff(3, 8, 16)
        elif 'RGB' in self.motion_type:
            pass
        else:
            raise RuntimeError("Not support this motion type '{}' yet!"
                               .format(self.motion_type))
        self.fix_rgb_channel = FixRGBchannel(3, 8, 16)
        # initialize fusion types
        self.fusion_layer1 = Fusion(consensus=self.consensus_type)
        self.fusion_layer2 = Fusion(consensus=self.consensus_type)
        self.fusion_layer3 = Fusion(consensus=self.consensus_type)
        self.fusion_layer4 = Fusion(consensus=self.consensus_type)
        # initialize head layers
        self.conv2linear = Conv2Linear(512, self.num_class, self.pool_type)

    def forward(self, last_cell_info, past_x, next_x):
        # get motion features
        fix_rgb = self.fix_rgb_channel(next_x)
        if 'PA' in self.motion_type:
            motion_feature = self.PA(past_x, next_x)
        elif 'RGBDiff' in self.motion_type:
            motion_feature = self.RGBDiff(past_x, next_x)
        elif 'RGB' in self.motion_type:
            motion_feature = fix_rgb
        else:
            raise RuntimeError("Not support this motion type '{}' yet!"
                               .format(self.motion_type))
        # cnn and rnn layer1 forward
        cnn_layer1_x = self.cnn_layer1(motion_feature)
        rnn_layer1_x, cell_info_1 = self.rnn_layer1(fix_rgb, last_cell_info[0])
        layer1_fusion = self.fusion_layer1(cnn_layer1_x, rnn_layer1_x)

        # cnn and rnn layer2 forward
        cnn_layer2_x = self.cnn_layer2(layer1_fusion[0])
        rnn_layer2_x, cell_info_2 = self.rnn_layer2(layer1_fusion[1], last_cell_info[1])
        layer2_fusion = self.fusion_layer2(cnn_layer2_x, rnn_layer2_x)

        # cnn and rnn layer3 forward
        cnn_layer3_x = self.cnn_layer3(layer2_fusion[0])
        rnn_layer3_x, cell_info_3 = self.rnn_layer3(layer2_fusion[1], last_cell_info[2])
        layer3_fusion = self.fusion_layer3(cnn_layer3_x, rnn_layer3_x)

        # cnn and rnn layer4 forward
        cnn_layer4_x = self.cnn_layer4(layer3_fusion[0])
        rnn_layer4_x, cell_info_4 = self.rnn_layer4(layer3_fusion[1], last_cell_info[3])
        layer4_fusion = self.fusion_layer4(cnn_layer4_x, rnn_layer4_x)

        # conv to linear layer
        out = self.conv2linear(layer4_fusion[1])

        # collect rnn_cell_info
        rnn_cell_info = [cell_info_1, cell_info_2, cell_info_3, cell_info_4]
        return out, rnn_cell_info








