# 测试
# 具体见每个指标对应函数的注释
import torch
import os
import json
import pickle
from config import config
from data_load import data_load
from models.CVAE.vae_framework import VAE_Framework
from tqdm import tqdm
from eval_metrics import eval_oracle, eval_distinct, eval_novel, eval_ngram_diversity, eval_mBleu4
from pycocoevalcap.eval import COCOEvalCap
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

print(dic_m)

log_path = config.log_dir.format(config.id)
result_dir = os.path.join(log_path, 'generated')

with open(config.vocab, 'rb') as f:
    vocab = pickle.load(f)

# 加载ref用于计算指标
ref_dir = './data/pycocoref_test.json'
ref_data = json.load(open(ref_dir, 'r'))

# 加载训练好的模型
model = VAE_Framework(config).to(device)
test_step = config.step
ckpts_path = log_path + '/model/model_' + str(test_step) + '.pt'
model.load_state_dict(torch.load(ckpts_path))

def generate_latent_vec(model, num_samples, latent_dim, test_length):
    latent_vec = []
    i = num_samples
    while i > 0:
        vec = torch.randn(latent_dim).to(device)
        pre_l = float(model.predict_sentence_length(vec))
        if((pre_l > 7) & (pre_l < 15)):
            i = i-1
            latent_vec.append(vec)
    latent_vec = torch.stack(latent_vec)
    assert(latent_vec.size()[0] == num_samples)
    return latent_vec

def shift_latent_vec(num_samples, latent_dim, length, w):
    latent_vec = []
    for i in range(10,20):
        l_v = torch.randn(2, latent_dim).to(device)
        l_vec = l_v + (torch.tensor(dic_m[i]) - l_v @ w) @ w.t()
        latent_vec.append(l_vec)
    latent_vec = torch.cat(latent_vec,0)
    return latent_vec

model.eval()

# 为了计算oracle evaluation和diversity的指标，需要首先为每张图片生成一组句子（num_samples个），最终共5000*num_samples个
num_samples = config.num_samples
all_captions = [{} for _ in range(num_samples)]

real_l = []
data_loader = data_load(config, "test", config.test)
for step, (img_vec) in tqdm(enumerate(data_loader)):
    img_vec = img_vec.to(device)
    img_vec = img_vec.expand([num_samples, img_vec.size(1)])
    latent_vec = shift_latent_vec(num_samples, config.latent_dim, config.test_length, w.cuda())
    #latent_vec = torch.randn(num_samples, config.latent_dim).to(device)
    #latent_vec = generate_latent_vec(model, num_samples, config.latent_dim, config.test_length).to(device)
    sentences = model.generate_mutisamples(img_vec, latent_vec, num_samples)
    r_l = np.array(sentences.cpu())
    r_l = np.where(r_l==2, 0, 1).sum(axis=1)
    real_l.extend(r_l)
    for i in range(num_samples):
        refs = []   # 按照pycoco需要的格式记录
        ref = {}
        ref["image_id"] = step
        ref["caption"] = vocab.idList_to_sent(sentences[i])
        ref["id"] = step
        refs.append(ref)
        all_captions[i][str(step)] = refs
print(len(real_l))
np.savetxt('./data/real_l.csv', real_l, fmt='%d', delimiter=',')

print("Oracle evaluation...")
best_score_captions, pycoco_scores = eval_oracle(all_captions)
filename = 'test_oracle_' + str(test_step) + '.json'
test_oracle_dir = os.path.join(result_dir, filename)
with open(test_oracle_dir, 'w') as f:  # 保存oracle evaluation最终得到的5000个句子
    json.dump(best_score_captions, f)

print("Diversity evaluation")
distinct_ratio = eval_distinct(all_captions)
novel_ratio = eval_novel(all_captions, json.load(open(config.train, 'r')))
mBleu4 = eval_mBleu4(all_captions)
ngram_ratio1 = eval_ngram_diversity(all_captions, 1)
ngram_ratio2 = eval_ngram_diversity(all_captions, 2)

print("### Result ###")
print("Oracle")
pycoco_list = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"]
for item in pycoco_list:
    print(item + ": " + str(pycoco_scores[item]))
print("Diversity")
print("Ratio of distinct captions: " + str(distinct_ratio))
print("Ratio of novel captions: " + str(novel_ratio))
print("mBleu-4: " + str(mBleu4))
print("1-gram diversity: " + str(ngram_ratio1))
print("2-gram diversity: " + str(ngram_ratio2))
