import torch
import os
import json
import pickle
from config import config
from data_load import data_load
from models.CVAE.vae_framework import VAE_Framework
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

log_path = config.log_dir.format(config.id)
dic_path = './data/length_dic.json'

model = VAE_Framework(config).to(device)
test_step = config.step
ckpts_path = log_path + '/model/model_' + str(test_step) + '.pt'
model.load_state_dict(torch.load(ckpts_path))

train_loader = data_load(config, 'train', config.train)

dic = {}

w = list(model.parameters())[26]
bias = list(model.parameters())[27]

latent_vec = []
length = []

with torch.no_grad():
    for step, (cap, cap_len, img_vec) in tqdm(enumerate(train_loader)):
        cap = cap.to(device)
        cap_len = cap_len.to(device)
        img_vec = img_vec.to(device)

        cap_len = cap_len + 2
        length.append(cap_len)

        _, _, _, vec, _ = model(cap, cap_len, img_vec)
        latent_vec.append(vec)

latent_vec = torch.stack(latent_vec)
l = torch.stack(length)

torch.save(latent_vec, './latent_vec.pt', _use_new_zipfile_serialization=False)
torch.save(l, './length.pt', _use_new_zipfile_serialization=False)
torch.save(w, './weight.pt', _use_new_zipfile_serialization=False)

# latent_vecs_dim = latent_vec.size(-1)
# X = latent_vecs.view(-1, latent_vecs_dim)
# y = lengths.flatten()
# p = X@w
#
# torch.save(p, './projection.pt', _use_new_zipfile_serialization=False)
# torch.save(l, './length.pt', _use_new_zipfile_serialization=False)
print(l.size(), latent_vec.size(), w.size())

# dic_m = {}
# for key in dic:
#     dic_m[key] = dic[key][1]/dic[key][0]
#
# with open(dic_path, 'w') as f:
#     json.dump(dic_m, f)

# i = 0
# vec = []
# length = []
# with torch.no_grad():
#     for step, (cap, cap_len, img_vec) in tqdm(enumerate(train_loader)):
#         i += 1
#         cap = cap.to(device)
#         cap_len = cap_len.to(device)
#         img_vec = img_vec.to(device)
#
#         cap_len = cap_len + 2
#         length.append(cap_len)
#
#         _, _, _, latent_vec, _ = model(cap, cap_len, img_vec)
#         vec.append(latent_vec)
#
#         if(i == 200): break
#
#     vec = torch.stack(vec)
#     l = torch.stack(length)
#
#     torch.save(vec, './latent_vec.pt', _use_new_zipfile_serialization=False)
#     torch.save(l, './length.pt', _use_new_zipfile_serialization=False)
#     print(l.size(), vec.size())
