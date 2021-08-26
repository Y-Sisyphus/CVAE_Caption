# 用于评价（计算指标的代码）
import torch
import os
import pickle
import json
from data_load import data_load
from tqdm import tqdm
import random
from pycocoevalcap.eval import COCOEvalCap
from nltk import ngrams

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_sen(config, model, step, mode):
    # val时每个图片生成一个描述
    print("Generating sentence...")
    with open(config.vocab, 'rb') as f:
        vocab = pickle.load(f)
    log_path = config.log_dir.format(config.id)
    result_dir = os.path.join(log_path, 'generated')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    filename = mode + '_' + str(step) + '.json'
    out_dir = os.path.join(result_dir, filename)

    data_loader = data_load(config, mode, config.val if mode == 'val' else config.test)
    model.eval()
    cocoeval_out = {}

    for i, (img_vec) in tqdm(enumerate(data_loader)):

        img_vec = img_vec.to(device)
        latent_vec = torch.randn(1, config.latent_dim).to(device)  # 隐变量标准正态中随机采样得到
        sentence_id = model.generate_onesample(img_vec, latent_vec)
        sentence = vocab.idList_to_sent(sentence_id)

        refs = []
        ref = {}
        ref["image_id"] = i
        ref["caption"] = sentence
        ref["id"] = i
        refs.append(ref)
        cocoeval_out[i] = refs

    with open(out_dir, 'w') as f:
        json.dump(cocoeval_out, f)


def eval_pycoco(config, step, mode):
    # val时计算指标
    print("Calculating pycoco...")
    ref_dir = './data/pycocoref_'+mode+'.json'
    ref_data = json.load(open(ref_dir, 'r'))

    log_path = config.log_dir.format(config.id)
    result_dir = os.path.join(log_path, 'generated')
    filename = mode + '_' + str(step) + '.json'
    res_dir = os.path.join(result_dir, filename)
    res_data = json.load(open(res_dir, 'r'))

    cocoEval = COCOEvalCap('nothing', 'nothing')
    pycoco_scores = cocoEval.evaluate_diy(ref_data, res_data)

    return pycoco_scores  # 返回字典 key为["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE"]


def eval_oracle(all_captions):
    """oracle evaluation：每张图片生成num_samples个描述，之后使用ref、选出每张图片对应生成的一组描述中CIDEr得分最高的作为此图片对应的描述，代表模型性能的上限"""
    num_samples = len(all_captions)
    num_sentences = len(all_captions[0])
    all_scores = [[] for _ in range(num_sentences)]  # 记录每个句子的CIDEr得分
    ref_dir = './data/pycocoref_test.json'
    ref_data = json.load(open(ref_dir, 'r'))
    for i in range(num_samples):
        cocoEval = COCOEvalCap('nothing', 'nothing')
        scores = cocoEval.evaluate_diy_test(ref_data, all_captions[i])
        for step in range(len(scores)):
            all_scores[step].append(scores[step])

    best_score_captions = {}  # 每张图片对应的num_samples个句子中选出其中得分最高的
    for i in range(num_sentences):
        max_score_index = all_scores[i].index(max(all_scores[i]))
        best_score_captions[str(i)] = all_captions[max_score_index][str(i)]

    cocoEval = COCOEvalCap('nothing', 'nothing')
    pycoco_scores = cocoEval.evaluate_diy(ref_data, best_score_captions)  # 最终测试得到oracle evaluation结果

    return best_score_captions, pycoco_scores


def eval_distinct(all_captions):
    """计算生成的一组句子中不同caption的占比"""
    num_samples = len(all_captions)
    num_sentences = len(all_captions[0])
    ratio = 0
    for step in tqdm(range(num_sentences)):
        sentences = [item[str(step)][0]["caption"] for item in all_captions]
        distinct_sentences = set(sentences)
        ratio += (len(distinct_sentences)/len(sentences))

    ratio_mean = ratio / num_sentences
    return ratio_mean


def eval_novel(all_captions, dataset_train):
    """计算生成的一组句子中训练集中没出现过的比例"""
    num_samples = len(all_captions)
    num_sentences = len(all_captions[0])
    train_captions = [item["caption"] for item in dataset_train]
    num_samples = num_samples if num_samples < 5 else 5  # 按照论文，一般选5个
    num_novel = 0
    for step in tqdm(range(num_sentences)):
        for i in range(num_samples):
            sentence = all_captions[i][str(step)][0]["caption"].split()
            if sentence not in train_captions:
                num_novel += 1

    ratio = num_novel / (num_sentences*num_samples)
    return ratio


def eval_ngram_diversity(all_captions, n):
    """计算生成的一组句子中不同的n-garm的占所有n-gram的比例"""
    num_samples = len(all_captions)
    num_sentences = len(all_captions[0])
    ratio = 0
    for step in tqdm(range(num_sentences)):
        sentences = [item[str(step)][0]["caption"].split() for item in all_captions]
        all_ngram = []
        for item in sentences:
            if n == 1:
                all_ngram += item
            elif n == 2:
                all_ngram += ngrams(item, 2)
        distinct_ngram = set(all_ngram)
        ratio += (len(distinct_ngram)/len(all_ngram))

    ratio_mean = ratio / num_sentences
    return ratio_mean


def eval_mBleu4(all_captions):
    # mBleu4：生成一组num_samples个描述，对于其中的每个句子，计算该句子以剩余num_samples-1个句子为参考时的Bleu4，得分越高说明句子间越相似
    num_samples = len(all_captions)
    num_sentences = len(all_captions[0])
    num_samples = num_samples if num_samples < 5 else 5  # 按照论文，一般选5个
    mBleu4 = 0
    for i in range(num_samples):
        res_data = all_captions[i]
        ref_data = {}
        for step in range(num_sentences):
            refs = [all_captions[j][str(step)][0] for j in range(num_samples) if j != i]
            for k, item in enumerate(refs):
                item["id"] = k
            ref_data[str(step)] = refs
        cocoEval = COCOEvalCap('nothing', 'nothing')
        scores = cocoEval.evaluate_diy(ref_data, res_data)
        mBleu4 += scores["Bleu_4"]

    mBleu4_mean = mBleu4 / num_samples
    return mBleu4_mean



