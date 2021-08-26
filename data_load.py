# dataloader，加载dataset_train/val/test的数据
import torch
import json
import os
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import  WeightedRandomSampler
from utils.vocab import Vocabulary

class Sen_Object_data(Dataset):

    def __init__(self, config, dir, mode):
        super(Sen_Object_data, self).__init__()
        self.config = config
        self.text = json.load(open(dir, 'r'))
        self.img_dir_coco = os.path.join(config.resnet_feat_dir, 'coco')
        with open(self.config.vocab, 'rb') as f:
            self.vocab = pickle.load(f)
        self.mode = mode

    def __getitem__(self, item):
        if self.mode == "train":
            sen_token = self.text[item]["caption"]
            sen_id, sen_len = self.vocab.tokenList_to_idList(sen_token, self.config.fixed_len)

            file_name = self.text[item]['filename'][:-4] + '.npy'
            img_path = os.path.join(self.img_dir_coco, file_name)
            img_vec = torch.Tensor(np.load(img_path))  # 根据数据集中给出的路径直接加载图像特征作为输入
            return torch.Tensor(sen_id).long(), sen_len, img_vec

        else:
            file_name = self.text[item]['filename'][:-4]+'.npy'
            img_path = os.path.join(self.img_dir_coco, file_name)
            img_vec = torch.Tensor(np.load(img_path))  # 验证和测试时风格描述也给图片辅助
            return img_vec

    def __len__(self):
        return len(self.text)

def data_load(config, mode, dir):
    dataset = Sen_Object_data(config, dir, mode)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=config.batch_size if mode == 'train' else 1,
                             shuffle=True if mode == 'train' else False,
                             num_workers=config.num_workers,
                             )
    return data_loader
