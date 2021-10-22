import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import numpy as np
import torchvision
from numpy.random import randint
from .transforms import *


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0].split('.')[0]

    @property
    def label(self):
        return int(self._data[1]) - 1

    @property
    def num_frames(self):
        return int(self._data[2])


class FOLDataset(data.Dataset):
    def __init__(self,
                 root_path,
                 list_file,
                 num_segments=16,
                 split_mode='train', # train, val, test
                 modality='RGB', # RGB, Flow
                 image_tmpl='image_{:05d}.jpg',
                 sample_type='continuious', # continuious, interval_average, interval_random
                 transform=None):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.split_mode = split_mode
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.sample_type = sample_type
        self.transform = transform
        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')
            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        if self.sample_type == 'continuious':
            if record.num_frames > self.num_segments:
                offsets = np.arange(self.num_segments) + int(randint(record.num_frames - self.num_segments))
            else:
                offsets = np.sort(randint(record.num_frames, size=self.num_segments))
            return offsets + 1
        elif self.sample_type == 'interval_average':
            average_duration = record.num_frames // self.num_segments
            if average_duration > 0:
                segment_start = int(randint(average_duration))
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + segment_start
            elif record.num_frames > self.num_segments:
                offsets = np.sort(randint(record.num_frames, size=self.num_segments))
            else:
                offsets = np.zeros(self.num_segments)
            return offsets + 1
        elif self.sample_type == 'interval_random':
            average_duration = record.num_frames // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) \
                          + randint(average_duration, size=self.num_segments)
            elif record.num_frames > self.num_segments:
                offsets = np.sort(randint(record.num_frames, size=self.num_segments))
            else:
                offsets = np.zeros(self.num_segments)
            return offsets + 1
        else:
            raise RuntimeError("not support this sample type '{}' yet!".format(self.sample_type))

    def _get_val_indices(self, record):
        if self.sample_type == 'continuious':
            if record.num_frames > self.num_segments:
                offsets = np.arange(self.num_segments) + int(randint(record.num_frames - self.num_segments))
            else:
                offsets = np.sort(randint(record.num_frames, size=self.num_segments))
            return offsets + 1
        elif self.sample_type == 'interval_average' or 'interval_random':
            if record.num_frames > self.num_segments:
                tick = record.num_frames / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros(self.num_segments)
            return offsets + 1
        else:
            raise RuntimeError("not support this sample type '{}' yet!".format(self.sample_type))

    def _get_test_indices(self, record):
        if self.sample_type == 'continuious':
            if record.num_frames > self.num_segments:
                offsets = np.arange(self.num_segments) + int(randint(record.num_frames - self.num_segments))
            else:
                offsets = np.sort(randint(record.num_frames, size=self.num_segments))
            return offsets + 1
        elif self.sample_type == 'interval_average' or 'interval_random':
            if record.num_frames > self.num_segments:
                tick = record.num_frames / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros(self.num_segments)
            return offsets + 1
        else:
            raise RuntimeError("not support this sample type '{}' yet!".format(self.sample_type))

    def __getitem__(self, index):
        record = self.video_list[index]
        if self.split_mode == 'train':
            segment_indices = self._sample_indices(record)
        elif self.split_mode == 'val':
            segment_indices = self._get_val_indices(record)
        elif self.split_mode == 'test':
            segment_indices = self._get_test_indices(record)
        else:
            raise RuntimeError("not support this split mode '{}' yet!".format(self.split_mode))
        return self.get(record, segment_indices)

    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            seg_imgs = self._load_image(os.path.join(self.root_path, record.path), p)
            images.extend(seg_imgs)
        if self.transform is not None:
            process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)

if __name__ == '__main__':
    root_path = '/home/dabingreat666/dataset/ucf101/feature_test_jpg'
    list_file = '/home/dabingreat666/dataset/ucf101/feature_test_list/testlist1.txt'
    transform = torchvision.transforms.Compose([GroupMultiScaleCrop(224, [1, .875, .75, .66]),
                                                GroupRandomHorizontalFlip(is_flow=False),
                                                Stack(),
                                                ToTorchFormatTensor(),
                                                GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dataset = FOLDataset(root_path, list_file, transform=transform)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=2,
                                         shuffle=True,
                                         num_workers=1,
                                         pin_memory=True,
                                         drop_last=True)
    for i, (data, label) in enumerate(loader):
        if i == 0:
            print(data, data.type(),
                  label, label.type())
