import torch
import numpy as np
import argparse
from config import config
import pickle
from models.CVAE.vae_framework import VAE_Framework
import os
import json


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
log_path = config.log_dir.format(config.id)
model = VAE_Framework(config).to(device)
test_step = config.step
ckpts_path = log_path + '/model/model_' + str(test_step) + '.pt'
model.load_state_dict(torch.load(ckpts_path))
model.eval()

# 加载mu_k tensor
# tensor_path = log_path + '/tensor/tensor_' + str(0) + '.pt'
# mu_k = torch.load(tensor_path)
# sigma_k = 2

with open(config.vocab, 'rb') as f:
    vocab = pickle.load(f)

def shift_latent_vec(num_samples, latent_dim, length, w):
    latent_vec = []
    for i in range(10,20):
        l_v = torch.randn(2, latent_dim).to(device)
        l_vec = l_v + (torch.tensor(dic_m[i]) - l_v @ w) @ w.t()
        latent_vec.append(l_vec)
    latent_vec = torch.cat(latent_vec,0)
    return latent_vec

def predict():
    word_map = json.load(open(os.path.join(config.data_folder, 'WORD_MAP.json')))
    itow = word_map['itow']
    file_name = 'COCO_train2014_000000511036' + '.npy'
    img_dir_coco = os.path.join(config.resnet_feat_dir, 'coco')
    img_path = os.path.join(img_dir_coco, file_name)

    print(img_path)
    img_vec = torch.Tensor(np.load(img_path)).to(device)  # 根据数据集中给出的路径直接加载图像特征作为输入
    cate = [53, 53, 53, 56, 57, 57, 57, 57, 57, 57, 57]
    cate_len = len(cate)  # 一个样本中的对象类别总数
    img_cate = torch.zeros(90)

    for k in cate:
        img_cate[k - 1] += 1
    img_cate /= cate_len

    img_cate = img_cate.to(device)
    img_cate_temp = img_cate.unsqueeze(0).to(device)  # 90->(1,90)
    # img_cate = torch.sum(img_cate, dim=0)  # (1,90) ->90
    # print(img_cate.shape)
    img_cate = img_cate.unsqueeze(1).expand([90, 150])  # 90->(90,150)

    latent_vec = shift_latent_vec(num_samples, config.latent_dim, config.test_length, w.cuda())
    #seq = model.generate_mutisamples(img_vec, latent_vec, num_samples)

    # mu = (img_cate * mu_k).to(device)
    # sigma2 = ((img_cate ** 2) * (sigma_k ** 2)).to(device)
    # mu = torch.sum(mu, dim=0, keepdim=True)
    # sigma2 = torch.sum(sigma2, dim=0, keepdim=True)  # 加权求和 (1,150)
    #
    # # mu = mu.expand([num_samples, 150])
    # # sigma2 = sigma2.expand([num_samples, 150])
    #
    # latent_vec = torch.randn(mu.size()).to(device)  # 隐变量随机采样
    # latent_vec *= torch.sqrt(sigma2)  # scaled
    # latent_vec += mu  # shifted

    # latent_vec = torch.randn(config.latent_dim).to(device)
    print(img_cate_temp.shape)

    seq = model.generate_onesample(img_vec, img_cate_temp, latent_vec)
    # seq0, seq = model.p_decoder.beam_search(img_vec, latent_vec)
    print(' '.join(list(map(lambda index: itow[str(index)], seq[1:-1]))) + '.')


def predict2():
    #word_map = json.load(open(os.path.join(config.data_folder, 'WORD_MAP.json')))
    #itow = word_map['itow']
    file_name = 'COCO_train2014_000000' + str(config.img) + '.npy'
    img_dir_coco = os.path.join(config.resnet_feat_dir, 'coco')
    img_path = os.path.join(img_dir_coco, file_name)

    print(img_path)
    num_samples = 20
    img_vec = torch.Tensor(np.load(img_path)).to(device)  # 根据数据集中给出的路径直接加载图像特征作为输入
    print(img_vec.shape)
    img_vec = img_vec.unsqueeze(0).expand([num_samples, img_vec.size(0)])
    print(img_vec.shape)

    # cate = [72, 82]
    # cate_len = len(cate)  # 一个样本中的对象类别总数
    # img_cate = torch.zeros(90)

    # for k in cate:
    #     img_cate[k - 1] += 1
    # img_cate /= cate_len

    # img_cate = img_cate.to(device)
    # img_cate_temp = img_cate.to(device)
    # # img_cate = torch.sum(img_cate, dim=0)  # (1,90) ->90
    # print(img_cate.shape)
    # img_cate = img_cate.unsqueeze(1).expand([90, 150])  # 90->(90,150)
    # print(img_cate.shape)

    latent_vec = shift_latent_vec(num_samples, config.latent_dim, config.test_length, w.cuda())

    # mu = (img_cate * mu_k).to(device)
    # sigma2 = ((img_cate ** 2) * (sigma_k ** 2)).to(device)
    # mu = torch.sum(mu, dim=0, keepdim=True)
    # sigma2 = torch.sum(sigma2, dim=0, keepdim=True)  # 加权求和 (1,150)
    #
    # mu = mu.expand([num_samples, 150])
    # sigma2 = sigma2.expand([num_samples, 150])
    #
    # latent_vec = torch.randn(mu.size()).to(device)  # 隐变量随机采样
    # latent_vec *= torch.sqrt(sigma2)  # scaled
    # latent_vec += mu  # shifted

    # latent_vec = torch.randn(config.latent_dim).to(device)
    # img_cate_temp = img_cate_temp.unsqueeze(0).expand([num_samples, img_cate_temp.size(0)])  # ([10, 90])
    # print(img_cate_temp.shape)

    seq = model.generate_mutisamples(img_vec, latent_vec, num_samples)
    # seq0, seq = model.p_decoder.beam_search(img_vec, latent_vec)
    # seq = model.predict_beam_search(img_feat, word_map, beam_size=params['beam_size'], max_len=params['max_len'])
    # print(' '.join(list(map(lambda index: itow[str(index)], seq[1:-1]))) + '.')
    for s in seq:
        print(vocab.idList_to_sent(s))

with torch.no_grad():
    latent_vecs = torch.load('./latent_vec.pt', map_location='cpu')
    lengths = torch.load('./length.pt', map_location='cpu')
    w = torch.load('./weight.pt', map_location='cpu')

    latent_vecs_dim = latent_vecs.size(-1)
    v = (latent_vecs.view(-1, latent_vecs_dim))
    l = np.array(lengths.flatten())
    w = w/torch.norm(w)
    w = w.t()

    p = np.array(v @ w)

    u = set(l)

    dic_m = {}
    for i in u:
        dic_m[i] = p[l==i].mean()

print("predict some")
predict2()

print("one")
#predict()
