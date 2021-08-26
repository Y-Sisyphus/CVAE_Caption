# 为测试时方便，使用自己改写后pycoco的代码，所以需要把生成的句子和ref参考句子都转换为指定格式
# 将ref的格式转化为便于pycoco直接计算指标的格式，用于val和test时计算指标
import json
from tqdm import tqdm

eval_text_dir = '../data/dataset_test.json'
cocoeval_out_dir = '../data/pycocoref_test.json'

eval_dataset = json.load(open(eval_text_dir, 'r'))

cocoeval_out = {}
for step, item in tqdm(enumerate(eval_dataset)):
    refs = []
    for step_new, item_new in enumerate(item["caption"]):
        ref = {}
        ref['image_id'] = step
        ref['id'] = step_new
        caption = ''
        sentence = item_new
        for word in sentence:
            caption += word
            caption += ' '
        ref['caption'] = caption[:-1]
        refs.append(ref)
    cocoeval_out[step] = refs

print(len(cocoeval_out))
with open(cocoeval_out_dir, 'w') as f:
    json.dump(cocoeval_out, f)