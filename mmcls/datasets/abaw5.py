import os
import random
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union, List, Callable, Sequence

import numpy as np
import torch
# from mmcls.models.losses import accuracy, f1_score, precision, recall
# from mmcls.models.losses.eval_metrics import class_accuracy

from .base_dataset import BaseDataset
from .builder import DATASETS


#                  0       1         2          3       4               5           6           7
FER_CLASSES = ['Anger', 'Disgust', 'Fear', 'Sadness', 'Happiness', 'Surprise', 'Neutral', 'Contempt']




def gen_class_map(dataset_class):
    """
    generate the convert map from DATASET_CLASSES to FER_CLASSES
    """
    convert_map = []
    for i in dataset_class:
        convert_map.append(FER_CLASSES.index(i))
    assert sum(convert_map) == sum([i for i in range(len(dataset_class))])
    return convert_map

@DATASETS.register_module()
class ABAW5(BaseDataset):

    CLASSES = [
        'Neutral',
        'Anger',
        'Disgust',
        'Fear',
        'Happiness',
        'Sadness',
        'Surprise',
        'Other'
    ]

    COARSE_CLASSES = [
        'Neutral',
        'Happiness',
        'Surprise',
        'Other',
        'Negative'
    ]
    coarse_map = [0, 4, 4, 4, 1, 4, 2, 3]

    NEGATIVE_CLASSES = [
        'Anger',
        'Disgust',
        'Fear',
        'Sadness',
    ]
    negative_map = [-1, 0, 1, 2, -1, 3, -1, -1]

    task = 'all'  # 'all', coarse', 'negative'
    def __init__(self,
                 data_prefix,
                 pipeline,
                 dataset=None,
                 classes=None,
                 ann_file=None,
                 test_mode=False):
        super().__init__(data_prefix, pipeline, classes, ann_file, test_mode)
        # self.CLASSES = self.update_classes()
    
    def update_classes(self):
        if self.task_type == 'EXPR':
            return self.CLASSES
        elif self.task_type == 'AU':
            return [f'AU{i+1}' for i in range(12)]
        elif self.task_type == 'VA':
            return ['V', 'A']

    def process_one_ann(self, dir, ann_file:str):
        with open(os.path.join(dir, ann_file), 'r') as f:
            data = f.read().strip().split('\n')[1:]
        return data
    
    def list_txt_files(self, dir):
        files = os.listdir(dir)
        files = [i for i in files if i.endswith('.txt')]
        return files

    def load_annotations(self, label_file=None):
        # return self.load_test_annotations()
        if 'EXPR_' in self.ann_file:
            self.task_type = 'EXPR'
            return self.load_ce_annotations(label_file)
        elif 'AU_' in self.ann_file:
            self.task_type = 'AU'
            return self.load_au_annotations()
        elif 'VA_' in self.ann_file:
            self.task_type = 'VA'
            return self.load_va_annotations()
        else:
            raise ValueError('invalid task')

    def load_ce_annotations(self, label_file=None):
        """Load CE annotations"""
        if label_file is None:
            label_file = self.ann_file
        if isinstance(label_file, str): # is a folder
            if label_file.endswith('.txt'):
                with open(label_file, 'r') as f:
                    ann_files = f.read().strip().split('\n')
                label_file = 'data/ABAW5/annotations/EXPR_Classification_Challenge/all'
            else:
                ann_files = os.listdir(label_file)
            ann_files = [i for i in ann_files if i.endswith('.txt')]
        else:
            raise TypeError('ann_file must be a str')

        # label_map = gen_class_map(self.DATASET_CLASSES)
        data_infos = []
        for ann_file in ann_files:  # xxx.txt
            ce_labels = self.process_one_ann(label_file, ann_file)
            rand_offset = np.random.randint(1, 10)
            for i, label in enumerate(ce_labels):
                # use 1/10 samples during training.
                if not self.test_mode:
                    if i % 10 != rand_offset:
                        continue
                label = int(label)
                if label == -1:
                    continue
                if self.task == 'coarse':
                    label = self.coarse_map[label]
                    self.CLASSES = self.COARSE_CLASSES
                elif self.task == 'negative':
                    label = self.negative_map[label]
                    self.CLASSES = self.NEGATIVE_CLASSES
                    if label == -1:     # Only negative has -1
                        continue
                img_prefix = os.path.join(self.data_prefix, ann_file.replace('.txt', ''))
                has_file = False
                for suffix in ('.jpg', '.png'):
                    filename = f'{str(i).zfill(5)}{suffix}'
                    if os.path.isfile(os.path.join(img_prefix, filename)):
                        has_file = True
                        break
                if not has_file:
                    continue
                info = {'img_prefix': img_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(label, dtype=np.int64)
                data_infos.append(info)

        return data_infos

    def load_test_annotations(self, label_file=None):
        label_file = 'data/ABAW5/annotations/EXPR_Classification_Challenge/test_set_format.txt'
        with open(label_file, 'r') as f:
            data = f.read().strip().split('\n')[1:]
        data_infos = []
        for line in data:  # xxx.txt
            filename = line.split(',')[0]
            img_prefix = self.data_prefix
            has_file = False
            for suffix in ('.jpg', '.png'):
                filename = filename[:-4] + suffix
                if os.path.isfile(os.path.join(img_prefix, filename)):
                    has_file = True
                    break
            if not has_file:
                continue
            info = {'img_prefix': img_prefix}
            info['img_info'] = {'filename': filename}
            data_infos.append(info)

        return data_infos

    def load_au_annotations(self,):
        """Load the AU annotations"""
        label_file = self.ann_file
        if isinstance(label_file, str): # is a folder
            ann_files = os.listdir(label_file)
            ann_files = [i for i in ann_files if i.endswith('.txt')]
        else:
            raise TypeError('ann_file must be a str')

        data_infos = []
        for ann_file in ann_files:  # xxx.txt
            # ce_labels = self.process_one_ann(label_file, ann_file)
            au_labels = self.process_one_ann(self.ann_file, ann_file)
            for i, label in enumerate(au_labels):
                if label == '-1':
                    continue
                img_prefix = os.path.join(self.data_prefix, ann_file.replace('.txt', ''))
                filename = f'{str(i).zfill(5)}.jpg'
                if not os.path.isfile(os.path.join(img_prefix, filename)):
                    continue
                info = {'img_prefix': img_prefix}
                info['img_info'] = {'filename': filename}
                # label = label_map[int(label)]
                # info['gt_label'] = np.array(label, dtype=np.int64)
                info['gt_label'] = np.array(au_labels[i].split(','), dtype=np.int64)
                data_infos.append(info)

        return data_infos

    def load_va_annotations(self,):
        """Load the VA annotations"""
        label_file = self.ann_file
        if isinstance(label_file, str): # is a folder
            ann_files = os.listdir(label_file)
            ann_files = [i for i in ann_files if i.endswith('.txt')]
        else:
            raise TypeError('ann_file must be a str')

        data_infos = []
        for ann_file in ann_files:  # xxx.txt
            va_labels = self.process_one_ann(self.ann_file, ann_file)
            for i, label in enumerate(va_labels):
                if float(label.split(',')[0]) < -1. or float(label.split(',')[1]) < -1:  # skip -5
                    continue
                img_prefix = os.path.join(self.data_prefix, ann_file.replace('.txt', ''))
                filename = f'{str(i).zfill(5)}.jpg'
                if not os.path.isfile(os.path.join(img_prefix, filename)):
                    continue
                info = {'img_prefix': img_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(label.split(','), dtype=np.float32)
                assert info['gt_label'].max() <= 1 and info['gt_label'].min() >= -1
                data_infos.append(info)

        return data_infos

    def load_exp_and_au_annotations(self):
        if 'Train_Set' not in self.ann_file:
            return self.load_annotations()

        label_map = gen_class_map(self.DATASET_CLASSES)
        assert 'EXPR_Set' in self.ann_file
        au_file = self.ann_file.replace('EXPR_Set', 'AU_Set')
        ce_files = self.list_txt_files(self.ann_file)
        au_files = self.list_txt_files(au_file)
        samples = dict()    # key: filename, value: [ce, au]

        for file in ce_files:
            samples[file] = [self.process_one_ann(self.ann_file, file), None]
        for file in au_files:
            au_labels = self.process_one_ann(au_file, file)
            if file in samples:
                samples[file][1] = au_labels    # 都保留
                # if len(au_labels) == len(samples[file][0]):
                #     samples[file][1] = au_labels
                # else:
                #     # samples[file] = [None, au_labels]   # 这里有两种可能，CE和AU冲突时保留谁
                #     samples[file][1] = None
            else:
                samples[file] = [None, au_labels]
        
        # va_path = self.ann_file.replace('EXPR_Set', 'VA_Set')
        # va_files = self.list_txt_files(va_path)
        # for file in va_files:
        #     va_labels = self.process_one_ann(va_path, file)
        #     if file in samples:

        #     else:
        #         samples[file] = [None, None, va_labels]


        # build data_infos
        data_infos = []
        for ann_file, labels in samples.items():
            have_ce_label = labels[0] is not None
            have_au_label = labels[1] is not None
            frame_num = len(labels[0]) if have_ce_label else len(labels[1])
            for i in range(frame_num):
                img_prefix = os.path.join(self.data_prefix, ann_file.replace('.txt', ''))
                filename = f'{str(i).zfill(5)}.jpg'
                if not os.path.isfile(os.path.join(img_prefix, filename)):
                    continue
                info = {'img_prefix': img_prefix}
                info['img_info'] = {'filename': filename}
                # info['ann_file'] = self.ann_file
                if have_ce_label:
                    origin_label = labels[0][i]
                    if origin_label != '-1':
                        label = label_map[int(origin_label)]
                        info['gt_label'] = np.array(label, dtype=np.int64)
                    else:
                        info['gt_label'] = np.array(255, dtype=np.int64)
                else:
                    info['gt_label'] = np.array(255, dtype=np.int64)
                if have_au_label:
                    try:
                        origin_label = labels[1][i].split(',')
                    except IndexError:    # i > len(labels[1]) 
                        info['au_label'] = np.array([255]*12, dtype=np.int64)
                    else:
                        if origin_label[0] != '-1':
                            info['au_label'] = np.array(origin_label, dtype=np.int64)
                        else:
                            info['au_label'] = np.array([255]*12, dtype=np.int64)
                else:
                    info['au_label'] = np.array([255]*12, dtype=np.int64)
                data_infos.append(info)
        return data_infos

    def __len__(self):
        # return 1000
        if os.getenv('DEBUG_MODE', '0') == '1':
            return 1000
        return super().__len__()
    
    def dump_results(self, results, save_file):
        results = np.vstack(results)

        assert len(results) == len(self)
        pred_label = np.argmax(results, axis=1)

        result_dict = dict()
        for ann, label in zip(self.data_infos, pred_label):
            key = ann['img_info']['filename'].replace('.jpg', '').replace('.png', '')
            result_dict[key] = int(label)
        
        sample_file = 'data/ABAW5/annotations/EXPR_Classification_Challenge/test_set_format.txt'
        with open(sample_file, 'r') as f:
            data = f.read().strip().split('\n')
        last_label = 7
        for i in range(1, len(data)):
            # print('ann', self.data_infos[0]['img_info']['filename'])
            # print(list(result_dict.keys())[0])
            # print(data[i][:-5])
            # return
            curr_label = result_dict.get(data[i][:-5], last_label)
            last_label = int(curr_label)
            data[i] += str(last_label)
        
        with open(save_file, 'w') as f:
            f.write('\n'.join(data))
