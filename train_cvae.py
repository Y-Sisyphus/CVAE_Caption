# 训练
import torch
import random
import numpy as np

import time
import sys
import os
import shutil
import json
import pickle
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from config import config
from data_load import data_load
from models.CVAE.vae_framework import VAE_Framework
from loss import CE_KL
from eval_metrics import generate_sen, eval_pycoco
from utils.tensorboard_writer import write_scalar, write_metrics
from utils.log_print import train_print
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置随机种子等确保结果可以复现
seed = config.seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

# 设置保存路径，保存参数、中间输出、模型以及tensorboard结果
log_path = config.log_dir.format(config.id)
if not os.path.exists(log_path):
    os.makedirs(log_path)
para_path = os.path.join(log_path, 'para.json')
with open(para_path, 'w') as f:
    json.dump(sys.argv, f)
shutil.copy('./config.py', log_path)

epochs = config.epoch
global_step = 0
writer = SummaryWriter(log_path)

with open(config.vocab, 'rb') as f:
    vocab = pickle.load(f)

train_loader = data_load(config, 'train', config.train)

model = VAE_Framework(config).to(device)

optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': config.lr}], betas=(0.9, 0.98), eps=1e-9)

# 训练CVAE最核心的两个损失：重构损失和KL散度
criterion = CE_KL()

loss_ce_average = 0
loss_kl_average = 0
loss_ml_average = 0
kl_rate_init = config.kl_rate
sl_rate = config.sl_rate

for epoch in range(epochs):

    model.train()
    total_step = len(train_loader)
    epoch_time = time.time()
    step_time = time.time()
    for step, (cap, cap_len, img_vec) in enumerate(train_loader):

        global_step += 1
        cap = cap.to(device)
        cap_len = cap_len.to(device)
        img_vec = img_vec.to(device)

        cap_len = cap_len + 2  # 开始符结束符

        logit, mu, sigma2, latent_vec, pre_length = model(cap, cap_len, img_vec)

        mse_l = torch.nn.functional.mse_loss(pre_length, cap_len.unsqueeze(1).float())

        loss_ce, loss_kl = criterion(logit, mu, sigma2, cap, cap_len)

        # kl_rate = kl_rate_init*(0+global_step/50000) if global_step <= 50000 else kl_rate_init  # 可以考虑从0递增的KL项权重，也就是kl annealing技巧
        kl_rate = kl_rate_init

        loss = loss_ce + kl_rate*loss_kl + sl_rate*mse_l  # 进行backward的损失是重构损失和KL散度的加权
        loss_ce_average += loss_ce.item()
        loss_kl_average += loss_kl.item()
        loss_ml_average += mse_l.item()

        model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), config.grad_clip)
        optimizer.step()

        # 在tensorboard中记录训练中两项损失的变化
        if global_step % config.save_loss_freq == 0:
            write_scalar(writer, 'loss_ce', (loss_ce_average/config.save_loss_freq), global_step)
            write_scalar(writer, 'loss_kl', (loss_kl_average/config.save_loss_freq), global_step)
            write_scalar(writer, 'loss_ml', (loss_ml_average/config.save_loss_freq), global_step)
            loss_ce_average = 0
            loss_kl_average = 0
            loss_ml_average = 0

        # print training information
        train_print(loss.item(), step, total_step, epoch, time.time() - step_time, time.time() - epoch_time)
        step_time = time.time()

        # 保存模型并val
        if global_step % config.save_model_freq == 0:
            print("Evaluating...")
            model.eval()

            # 保存模型
            model_path = os.path.join(log_path, 'model')
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            save_path = os.path.join(model_path, f'model_{global_step}.pt')
            torch.save(model.state_dict(), save_path)

            # val
            generate_sen(config, model, global_step, 'val')  # 生成测试集的图像描述
            pycoco_scores = eval_pycoco(config, global_step, 'val')  # 使用改写的pycoco代码计算指标
            write_metrics(writer, pycoco_scores, global_step)  # 在tensorboard中记录val的结果

            model.train()
