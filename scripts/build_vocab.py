# 从训练数据集中生成单词表
import json
import pickle
from tqdm import tqdm

from utils.vocab import Vocabulary

vocab = Vocabulary()

dataset_train = json.load(open('../data/dataset_train.json', 'r'))

counter = {}
for item in tqdm(dataset_train):
    sentence_token = item["caption"]
    for token in sentence_token:
        counter[token] = counter.get(token, 0)+1
cand_word = [token for token, f in counter.items() if f > 5]
print("Vocab size coco: "+str(len(cand_word)))

for w in cand_word:
    vocab.add_word(w)
vocab_path = '../data/vocab.pkl'
with open(vocab_path, 'wb') as f:
    pickle.dump(vocab, f)
print(f'vocab size: {vocab.get_size()}, saved to {vocab_path}')
