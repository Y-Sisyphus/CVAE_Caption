import torch
import os
import json
import pickle
import numpy as np

device = 'cpu'

with torch.no_grad():
    latent_vecs = torch.load('./latent_vec.pt', map_location=device)
    lengths = torch.load('./length.pt', map_location=device)
    w = torch.load('./weight.pt', map_location=device)

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

with open('./data/mean_p.json', 'w') as f:
    json.dump(dic_m, f)

# u = torch.unique(l)
# dic_m = {}
# for i in u:
#     p_i = torch.where(l==i, p, torch.tensor(0))
#     n = torch.sum(torch.where(p_i==0, torch.tensor(1), torch.tensor(0)))
#     dic[i] = torch.sum(p_i)/n

# dic = {}
# for i in range(v.shape[0]):
#     if (l[i] in dic.keys()):
#         dic[l[i]][0] += 1
#         dic[l[i]][1] += p[i]
#     else:
#         dic[l[i]] = [1, p[i]]
#
# dic_m = {}
# for key in dic:
#     dic_m[key] = dic[key][1] / dic[key][0]

print(dic_m)
print(v.shape, l.shape, w.shape)