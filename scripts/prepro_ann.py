# 预处理coco2014数据集（目标检测标注部分），得到object categories的标注
# 保存为一个字典，key是图片的cocoid，val是其包含的object categories（一个list）
import json
from tqdm import tqdm

ann_train = json.load(open('../data/instances_train2014.json', 'r'))["annotations"]
ann_val = json.load(open('../data/instances_val2014.json', 'r'))["annotations"]

ann_dict = {}
print("Processing ann_train...")
for item in tqdm(ann_train):
    cocoid = item["image_id"]
    if cocoid in ann_dict:
        ann_dict[cocoid].append(item["category_id"])
    else:
        ann_dict[cocoid] = [item["category_id"]]

print("Processing ann_val...")
for item in tqdm(ann_val):
    cocoid = item["image_id"]
    if cocoid in ann_dict:
        ann_dict[cocoid].append(item["category_id"])
    else:
        ann_dict[cocoid] = [item["category_id"]]

print("Num of ann_dict: " + str(len(ann_dict)))
with open('../data/ann_dict.json', 'w') as f:
    json.dump(ann_dict, f)
