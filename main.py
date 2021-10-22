import os
import time
import shutil
import torch
import torch.optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_
from torchvision import transforms
from tensorboardX import SummaryWriter

from data.dataset import FOLDataset
from models.rnnpluscnn import RNNplusCNN
from models.only_cnn import OnlyCNN
from models.only_rnn import OnlyRNN
from data.transforms import *
from opts import parser
from data.dataset_config import return_dataset
from tools.utils import *


best_prec1 = 0
# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
now_time = time.strftime("%Y%m%d_%H%M", time.localtime())


def main():
    global args, best_prec1
    args = parser.parse_args()
    num_class, train_list_file, val_list_file, root_path, prefix = return_dataset(args.dataset, args.modality)

    # initialize model
    if args.model_type == 'rnnpluscnn':
        model = RNNplusCNN(num_class,
                           motion_type=args.motion_type,
                           rnn_model=args.rnn_model,
                           cnn_model=args.cnn_model,
                           consensus_type=args.consensus_type,
                           pool_type=args.pool_type).cuda()
    elif args.model_type == 'only_cnn':
        model = OnlyCNN(num_class,
                        motion_type=args.motion_type,
                        cnn_model=args.cnn_model,
                        pool_type=args.pool_type).cuda()
    elif args.model_type == 'only_rnn':
        model = OnlyRNN(num_class,
                        motion_type=args.motion_type,
                        rnn_model=args.rnn_model,
                        pool_type=args.pool_type).cuda()
    else:
        raise RuntimeError("Not support this model type '{}' yet!"
                           .format(args.model_type))
    model = torch.nn.DataParallel(model)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise RuntimeError("Not support this loss type '{}' yet!".format(args.loss_type))
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    milestones = args.lr_steps
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.5)

    # resume
    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['lr_schedule'])
            print(("=> loaded checkpoint with 'EVALUATE = {}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'!".format(args.resume)))

    cudnn.benchmark = True

    # set prefix data process
    train_transform = transforms.Compose([GroupMultiScaleCrop(224, [1, .875, .75, .66]),
                                          # GroupRandomHorizontalFlip(is_flow=False),
                                          Stack(),
                                          ToTorchFormatTensor(),
                                          GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    val_transform = transforms.Compose([GroupCenterCrop(224),
                                        # GroupRandomHorizontalFlip(is_flow=False),
                                        Stack(),
                                        ToTorchFormatTensor(),
                                        GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # initialize and preload dataset
    train_dataset = FOLDataset(root_path,
                               train_list_file,
                               num_segments=args.num_segments,
                               split_mode='train',
                               modality=args.modality,
                               image_tmpl=prefix,
                               sample_type=args.sample_type,
                               transform=train_transform)
    val_dataset = FOLDataset(root_path,
                             val_list_file,
                             num_segments=args.num_segments,
                             split_mode='val',
                             modality=args.modality,
                             image_tmpl=prefix,
                             sample_type=args.sample_type,
                             transform=val_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.workers,
                                             pin_memory=True,
                                             drop_last=True)

    # set log_csv, args_txt, tensorboard
    args.store_name = now_time
    store_log_path = os.path.join(args.root_log, args.store_name)
    if not os.path.exists(store_log_path):
        os.mkdir(store_log_path)
    log_training = open(os.path.join(store_log_path, 'log.csv'), 'w')
    with open(os.path.join(store_log_path, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(store_log_path, 'tb'))

    # start train and val loop
    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, log_training, tf_writer)
        scheduler.step()
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch, log_training, tf_writer)
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        tf_writer.add_scalar('val_best_top1_acc', best_prec1, epoch)
        output_best = 'Best Prec@1: %.4f\n' % (best_prec1)
        print(output_best)
        log_training.write(output_best + '\n')
        log_training.flush()
        # save model state
        save_checkpoint({
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_schedule': scheduler.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, log, tf_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    last_loss = AverageMeter()
    last_top1 = AverageMeter()
    last_top5 = AverageMeter()

    model.train()
    end = time.time()
    for i, (data, label) in enumerate(train_loader):
        data_time.update(time.time() - end)  # load data and print log time consume
        data = data.cuda()
        label = label.cuda()
        c = 3
        b, sc, h, w = [x for x in data.size()]  # batch_size, segment*channel, high, width
        s = sc // c
        data = data.view(b, s, c, h, w).contiguous()
        for t in range(s):
            if t == 0:
                past_frame = torch.zeros((b, c, h, w)).cuda()
                rnn_cell_info = [[None, None], [None, None], [None, None], [None, None]]
            else:
                past_frame = data[:, t-1, :, :, :]
            next_frame = data[:, t, :, :, :]
            out, rnn_cell_info = model(rnn_cell_info, past_frame, next_frame)
            loss = criterion(out, label)
            prec1, prec5 = accuracy(out.data, label, topk=(1, 5))
            losses.update(loss.item(), b)
            top1.update(prec1.item(), b)
            top5.update(prec5.item(), b)
            loss.backward()
            # set loss backward type
            if s < args.iter_size:
                raise ValueError("iter_size '{}' must less num_segments '{}'!"
                                 .format(args.iter_size, args.num_segments))
            else:
                no_grad_cnt = 0
                if (t+1) % args.iter_size == 0:
                    # scale down gradients when iter size is functioning
                    if args.iter_size != 1:
                        for g in optimizer.param_groups:
                            for p in g['params']:
                                if isinstance(p.grad, torch.Tensor):
                                    p.grad /= args.iter_size
                                else:
                                    no_grad_cnt = no_grad_cnt + 1

                    if args.clip_gradient is not None:
                        total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
                    else:
                        total_norm = 0
                    optimizer.step()
                    optimizer.zero_grad()
            # print IN_VIDEO info
            # if s < args.print_frame_freq:
            #     raise ValueError("print_frame_freq '{}' must less num_segments '{}'!"
            #                      .format(args.print_frame_freq, args.num_segments))
            # else:
            #     print("[TRAINING ==> IN_VIDEO]\t"
            #           "epoch[{epoch}]/batch[{batch}]/time[{t}]\t"
            #           "loss[{loss_val:.4f}]/[{loss_avg:.4f}]\t"
            #           "top1[{top1_val:.4f}]/[{top1_avg:.4f}]\t"
            #           "top5[{top5_val:.4f}]/[{top5_avg:.4f}]\t"
            #           .format(epoch=epoch, batch=i, t=t,
            #                   loss_val=losses.val, loss_avg=losses.avg,
            #                   top1_val=top1.val, top1_avg=top1.avg,
            #                   top5_val=top5.val, top5_avg=top5.avg))
        # get last frame log
        last_loss.update(losses.val, b)
        last_top1.update(top1.val, b)
        last_top5.update(top5.val, b)
        batch_time.update(time.time() - end)  # batch time consume
        end = time.time()
        # print train info
        if i % args.print_batch_freq == 0:
            output = ("[TRAINING ==> ALL_VIDEO]\t"
                      "epoch[{epoch}]/[{total}]|[{up}]/[{down}]\t"
                      "lr[{lr:.6f}]\t"
                      "batch_time[{batch_time_val:.4f}]\t"
                      "data_time[{data_time_val:.4f}]\t"
                      "loss[{loss_val:.4f}]/[{loss_avg:.4f}]\t"
                      "top1[{top1_val:.4f}]/[{top1_avg:.4f}]\t"
                      "top5[{top5_val:.4f}]/[{top5_avg:.4f}]\t"
                      .format(epoch=epoch, total=args.epochs, up=i, down=len(train_loader),
                              lr=optimizer.param_groups[-1]['lr'],
                              batch_time_val=batch_time.val,
                              data_time_val=data_time.val,
                              loss_val=last_loss.val, loss_avg=last_loss.avg,
                              top1_val=last_top1.val, top1_avg=last_top1.avg,
                              top5_val=last_top5.val, top5_avg=last_top5.avg))
            print(output)
            log.write(output + '\n')
            log.flush()
    # save log to tensor board
    tf_writer.add_scalar('train_loss', last_loss.avg, epoch)
    tf_writer.add_scalar('train_top1_acc', last_top1.avg, epoch)
    tf_writer.add_scalar('train_top5_acc', last_top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)


def validate(val_loader, model, criterion, epoch, log, tf_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    last_loss = AverageMeter()
    last_top1 = AverageMeter()
    last_top5 = AverageMeter()

    model.eval()
    end = time.time()
    for i, (data, label) in enumerate(val_loader):
        data_time.update(time.time() - end)  # load data and print log time consume
        data = data.cuda()
        label = label.cuda()
        c = 3
        b, sc, h, w = [x for x in data.size()]  # batch_size, segment*channel, high, width
        s = sc // c
        data = data.view(b, s, c, h, w).contiguous()
        for t in range(s):
            if t == 0:
                past_frame = torch.zeros((b, c, h, w)).cuda()
                rnn_cell_info = [[None, None], [None, None], [None, None], [None, None]]
            else:
                past_frame = data[:, t-1, :, :, :]
            next_frame = data[:, t, :, :, :]
            out, rnn_cell_info = model(rnn_cell_info, past_frame, next_frame)
            loss = criterion(out, label)
            # calculate accuracy and record loss
            prec1, prec5 = accuracy(out.data, label, topk=(1, 5))
            losses.update(loss.item(), b)
            top1.update(prec1.item(), b)
            top5.update(prec5.item(), b)
            # print ALL_VIDEO info
            # if s < args.print_frame_freq:
            #     raise ValueError("print_frame_freq '{}' must less num_segments '{}'!"
            #                      .format(args.print_frame_freq, args.num_segments))
            # else:
            #     print("[VALING ==> IN_VIDEO]\t"
            #           "epoch[{epoch}]/batch[{batch}]/time[{t}]\t"
            #           "loss[{loss_val:.4f}]/[{loss_avg:.4f}]\t"
            #           "top1[{top1_val:.4f}]/[{top1_avg:.4f}]\t"
            #           "top5[{top5_val:.4f}]/[{top5_avg:.4f}]\t"
            #           .format(epoch=epoch, batch=i, t=t,
            #                   loss_val=losses.val, loss_avg=losses.avg,
            #                   top1_val=top1.val, top1_avg=top1.avg,
            #                   top5_val=top5.val, top5_avg=top5.avg))
        # get last frame log
        last_loss.update(losses.val, b)
        last_top1.update(top1.val, b)
        last_top5.update(top5.val, b)
        batch_time.update(time.time() - end)  # batch time consume
        end = time.time()
        # print train info
        if i % args.print_batch_freq == 0:
            output = ("[VALING ==> ALL_VIDEO]\t"
                      "epoch[{epoch}][{total}]|[{up}]/[{down}]\t"
                      "batch_time[{batch_time_val:.4f}]\t"
                      "data_time[{data_time_val:.4f}]\t"
                      "loss[{loss_val:.4f}]/[{loss_avg:.4f}]\t"
                      "top1[{top1_val:.4f}]/[{top1_avg:.4f}]\t"
                      "top5[{top5_val:.4f}]/[{top5_avg:.4f}]\t"
                      .format(epoch=epoch, total=args.epochs, up=i, down=len(val_loader),
                              batch_time_val=batch_time.val,
                              data_time_val=data_time.val,
                              loss_val=last_loss.val, loss_avg=last_loss.avg,
                              top1_val=last_top1.val, top1_avg=last_top1.avg,
                              top5_val=last_top5.val, top5_avg=last_top5.avg))
            print(output)
            log.write(output + '\n')
            log.flush()
    # save log to tensor board
    tf_writer.add_scalar('val_loss', last_loss.avg, epoch)
    tf_writer.add_scalar('val_top1_acc', last_top1.avg, epoch)
    tf_writer.add_scalar('val_top5_acc', last_top5.avg, epoch)
    return last_top1.avg


def save_checkpoint(state, is_best):
    ckpt_dir = os.path.join(args.root_model, args.store_name)
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    filename = os.path.join(ckpt_dir, 'ckpt.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


# def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
#     if lr_type == 'step':
#         decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
#         lr = args.lr * decay
#         decay = args.weight_decay
#     elif lr_type == 'cos':
#         import math
#         lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
#         decay = args.weight_decay
#     else:
#         raise RuntimeError("Not support this lr_type '{}' yet!".format(lr_type))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr * param_group['lr_mult']
#         param_group['weight_decay'] = decay * param_group['decay_mult']


if __name__ == '__main__':
    main()


