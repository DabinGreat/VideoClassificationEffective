import argparse


parser = argparse.ArgumentParser(description="PyTorch implementation of Fast_Online_LSTM_Networks")

# ========================= Data Configs ==========================
parser.add_argument('dataset', type=str)
parser.add_argument('modality', type=str, choices=['RGB', 'Flow'])
parser.add_argument('--num_segments', '-t', type=int, default=16)
parser.add_argument('--sample_type', '-spt', type=str, default='continuious',
                    choices=['continuious', 'interval_average', 'interval_random'])

# ========================= Model Configs ==========================
parser.add_argument('--model_type',  type=str, default='rnnpluscnn',
                    choices=['rnnpluscnn', 'only_rnn', 'only_cnn', 'standard_resnet', 'standard_rnnresnet'])
parser.add_argument('--rnn_model', type=str, default='lstm', choices=['lstm', 'gru'])
parser.add_argument('--cnn_model', type=str, default='resnet', choices=['resnet'])
parser.add_argument('--motion_type', type=str, default='PA', choices=['PA', 'RGBDiff', 'RGB'])
parser.add_argument('--consensus_type', type=str, default='cnn2rnn_add', choices=['cnn2rnn_add'])
parser.add_argument('--pool_type', type=str, default='maxpool', choices=['view', 'maxpool', 'avgpool'])

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--loss_type', type=str, default="nll", choices=['nll'])
parser.add_argument('--lr_type', default='step', type=str,
                    metavar='LRtype', help='learning rate type')
parser.add_argument('--lr_steps', default=[25, 35], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip_gradient', '--gd', default=200, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('-i', '--iter_size', default=1, type=int,
                    metavar='N', help='number of iterations before on update')

# ========================= Monitor Configs ==========================
parser.add_argument('--print_frame_freq', '-pff', default=4, type=int,
                    metavar='N', help='print frequency (default: 2)')
parser.add_argument('--print_batch_freq', '-pbf', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)

# ========================= Save Configs ==========================
parser.add_argument('--store_name', '-s', type=str, default='')
parser.add_argument('--root_log', type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')
